import datetime
import json
import time
from copy import deepcopy
from datetime import timezone
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from chibi.config import gpt_settings
from chibi.exceptions import NoProviderSelectedError
from chibi.models import Message, SelectedModel, TelegramFileMeta, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, VisionResultSchema
from chibi.services.interface import EditorContextProvider, UserInterface
from chibi.services.lock_manager import LockManager
from chibi.services.usage_cache import UsageCacheStore
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider
    from chibi.services.providers.tools import ToolResponseSchema


@inject_database
async def get_chibi_user(db: Database, user_id: int) -> User:
    return await db.get_or_create_user(user_id=user_id)


@inject_database
async def set_active_model(db: Database, interface: UserInterface, model: ModelChangeSchema) -> None:
    user = await db.get_or_create_user(user_id=interface.storage_id)
    thread_id = interface.thread_id
    if model.video_generation:
        user.thread_selected_video_model[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    elif model.image_generation:
        user.thread_selected_image_model[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    else:
        user.thread_selected_llm[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    await db.save_user(user)


@inject_database
async def reset_chat_history(db: Database, storage_id: int, thread_id: int) -> None:
    user = await db.get_or_create_user(user_id=storage_id)
    await db.drop_messages(user=user, thread_id=thread_id)


# Share of gpt_settings.max_history_tokens used as the token budget for the
# input of the emergency summarization request itself. At 100k history tokens
# the raw full-history prompt used to overflow the summarization request;
# 60% leaves room for the system prompt, tool schemas and the generated
# summary output while discarding as little context as possible.
EMERGENCY_SUMMARIZATION_INPUT_BUDGET = 0.6


def _is_tool_response_message(message: Message) -> bool:
    """Detect background tool-response turns stored as role="user" JSON blobs.

    Args:
        message: The message to inspect.

    Returns:
        True when the message content is a JSON object with "type": "tool response".
    """
    if message.role != "user" or not message.content:
        return False
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("type") == "tool response"


def _fit_messages_to_budget(messages: list[Message], budget: int) -> list[Message]:
    """Build a transient summarizer input under the given token budget.

    The input is a separate construction: the stored conversation history is
    never mutated. Shape-(i) tool-call pairs (assistant ``tool_calls`` +
    role="tool" results) are already excluded from the summarizer input by the
    caller and never appear here. Dropping order when over budget:

    1. Background tool-response turns (role="user" JSON blobs with
       ``"type": "tool response"``), LARGEST first — they are the bulk of the
       volume and the least valuable as summarization context.
    2. The oldest remaining turns.

    The most recent message is never dropped so the summarizer always keeps
    the latest context anchor.

    Args:
        messages: Candidate messages (already filtered from tool-result turns).
        budget: Maximum total estimated tokens for the kept messages.

    Returns:
        The kept messages, in original order, fitting the budget.
    """
    total = sum(msg.estimate_tokens for msg in messages)
    if total <= budget:
        return messages

    dropped: set[int] = set()

    # (1) Background tool-response JSON blobs, largest first.
    tool_responses = sorted(
        (msg for msg in messages if _is_tool_response_message(msg)),
        key=lambda msg: msg.estimate_tokens,
        reverse=True,
    )
    for msg in tool_responses:
        if total <= budget:
            break
        dropped.add(id(msg))
        total -= msg.estimate_tokens

    # (2) Oldest remaining turns; never drop the most recent message.
    if total > budget:
        for msg in messages[:-1]:
            if total <= budget:
                break
            if id(msg) in dropped:
                continue
            dropped.add(id(msg))
            total -= msg.estimate_tokens

    return [msg for msg in messages if id(msg) not in dropped]


@inject_database
async def emergency_summarization(db: Database, storage_id: int, thread_id: int) -> None:
    user = await db.get_or_create_user(user_id=storage_id)

    chat_history = await db.get_conversation_messages(user=user, thread_id=thread_id)
    # In-loop tool results (shape-(i) pairs) are excluded from the summarizer
    # input exactly as before: assistant tool_calls and their role="tool"
    # results are not useful summarization context and their pairing is
    # provider-protocol sensitive.
    summarizable = [msg for msg in chat_history if not msg.tool_calls and not msg.tool_call_id]
    budget = int(gpt_settings.max_history_tokens * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    kept = _fit_messages_to_budget(summarizable, budget)
    chat_history_string = "\n".join(msg.content for msg in kept)
    user_messages: list[Message] = [Message(role="user", content=chat_history_string)]

    response, _ = await user.get_active_llm_provider(thread_id=thread_id).get_chat_response(
        messages=user_messages,
        user=user,
        model=user.get_active_llm_model(thread_id=thread_id),
        max_tokens=int(gpt_settings.max_tokens * 1.3),
        system_prompt="Summarize this conversation, keeping the most important and useful information using English.",
        caller_storage_id=storage_id,
        caller_thread_id=thread_id,
    )
    initial_message = Message(role="user", content="What we were talking about?")
    answer_message = Message(role="assistant", content=response.answer)
    await reset_chat_history(storage_id=storage_id, thread_id=thread_id)
    await db.add_message(user=user, message=initial_message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
    # The cached prompt size reflects the pre-reset history; without this the
    # first post-summary turn would instantly re-trigger summarization.
    UsageCacheStore().invalidate(user_id=storage_id, thread_id=thread_id)


@inject_database
async def save_telegram_document_metadata(db: Database, storage_id: int, file_metadata: dict[str, Any]) -> str:
    user = await db.get_or_create_user(user_id=storage_id)
    file_meta = TelegramFileMeta(**file_metadata)
    files_update = {
        file_meta.file_unique_id: file_meta,
    }
    if user.telegram_files:
        user.telegram_files.update(files_update)
    else:
        user.telegram_files = files_update

    await db.save_user(user)
    return file_meta.file_unique_id


@inject_database
async def get_telegram_documents(db: Database, storage_id: int, limit: int = 0) -> dict[str, TelegramFileMeta]:
    user = await db.get_or_create_user(user_id=storage_id)
    files = user.telegram_files
    if limit == 0:
        return files

    last_files = dict(islice(files.items(), max(len(files) - limit, 0), None))
    return last_files


@inject_database
async def get_telegram_document(db: Database, storage_id: int, file_unique_id: str) -> TelegramFileMeta | None:
    user = await db.get_or_create_user(user_id=storage_id)
    return user.telegram_files.get(file_unique_id)


@inject_database
async def get_llm_chat_completion_answer(
    db: Database,
    storage_id: int,
    interface: UserInterface,
    user_text_message: str | None = None,
    user_voice_message: BytesIO | None = None,
    user_caption: str | None = None,
    tool_message: Optional["ToolResponseSchema"] = None,
    scheduler_trigger_text: str | None = None,
    scheduler_job_id: str | None = None,
) -> ChatResponseSchema:
    """Produce an LLM answer for any incoming turn of the given thread.

    Builds the JSON prompt for the turn according to its kind — user message,
    tool response, file caption, or scheduler trigger (the third system
    message type) — appends it to the thread history, and asks the active
    provider for a completion. All new messages (prompt and answer) are
    persisted to the thread history.

    Args:
        db: Database instance (injected).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        interface: Interface of the current turn, used for the provider call
            context (storage/thread ids, editor context).
        user_text_message: Raw text typed by the user, if any.
        user_voice_message: Voice message audio, if any.
        user_caption: Caption of an uploaded file, if any.
        tool_message: Result of a background tool execution, if any.
        scheduler_trigger_text: Trigger text from a fired scheduler job
            (action type ``self``), if any.
        scheduler_job_id: Identifier of the scheduler job that fired; only
            meaningful together with ``scheduler_trigger_text``.

    Returns:
        The provider's answer for this turn.

    Raises:
        ValueError: If no prompt data of any kind is provided, or a voice
            message is given while the user has no STT provider.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    thread_id = interface.thread_id
    lock = await LockManager().get_lock(key=f"{storage_id}_{thread_id}")

    if (
        not user_text_message
        and not user_voice_message
        and not tool_message
        and not user_caption
        and not scheduler_trigger_text
    ):
        raise ValueError("No prompt data provided")

    if user_voice_message and not user.stt_provider:
        raise ValueError("Can't compute voice message: no STT provide available.")

    prompt: dict[str, Any]
    editor_ctx = interface.editor_context if isinstance(interface, EditorContextProvider) else None

    if scheduler_trigger_text:
        prompt = {
            "type": "scheduled_trigger",
            "desc": (
                "Service trigger from the Chibi scheduler. This is NOT a user message and not a tool "
                "response: the agent was woken up by a scheduled job in this thread. Handle the "
                "trigger_text below using the thread context. Reply with <chibi>ACK</chibi> (and "
                "nothing else) if there is nothing worth reporting to the user."
            ),
            "job_id": scheduler_job_id,
            "trigger_text": scheduler_trigger_text,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    elif tool_message:
        prompt = {
            "type": "tool response",
            "desc": "background task is done",
            "tool_name": tool_message.tool_name,
            "tool_response": tool_message.model_dump(),
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    elif user_caption:
        prompt = {
            "type": "system message",
            "desc": "user uploaded a file",
            "caption": user_caption,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    else:
        user_message = (
            await user.stt_provider.transcribe(audio=user_voice_message) if user_voice_message else user_text_message
        )
        assert user_message
        prompt = {
            "user": interface.user_data,
            "prompt": user_message,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "type": "user message",
            "transcribed_from_voice_message": bool(user_voice_message),
        }

    if editor_ctx:
        selection = editor_ctx.get("selection")
        editor_context: dict[str, Any] = {
            "active_file": editor_ctx.get("active_file"),
            "language_id": editor_ctx.get("language_id"),
            "cursor_position": editor_ctx.get("cursor_position"),
        }
        workspace_root = editor_ctx.get("workspace_root")
        if isinstance(workspace_root, str):
            editor_context["workspace_root"] = workspace_root.rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if isinstance(selection, dict) and isinstance(selection.get("text"), str) and selection["text"]:
            editor_context["selection"] = {
                "start_line": selection.get("start_line"),
                "end_line": selection.get("end_line"),
                "text": selection["text"][:4096],
            }
        prompt["editor_context"] = editor_context

    async with lock:
        conversation_messages: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)
        new_message_to_llm = Message(role="user", content=json.dumps(prompt))
        conversation_messages.append(new_message_to_llm)

        active_provider = user.get_active_llm_provider(thread_id=thread_id)
        active_model = user.get_active_llm_model(thread_id=thread_id)

        chat_response, new_messages = await active_provider.get_chat_response(
            messages=conversation_messages,
            user=user,
            model=active_model,
            interface=interface,
            track_prompt_size=True,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
        )
        await db.add_message(user=user, message=new_message_to_llm, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
        for message in new_messages:
            await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
        return chat_response


@inject_database
async def check_history_and_summarize(db: Database, storage_id: int, thread_id: int) -> bool:
    """Decide whether the conversation has grown large enough to auto-summarize.

    The primary signal is the real provider-reported prompt size cached in
    ``UsageCacheStore`` for this ``(user_id, thread_id)`` — the same key the
    write side (provider chat completion) and the read side
    (``prepare_system_prompt``) use. This figure reflects the *entire* outgoing
    request (system prompt, skills, tool schemas, tool-call arguments,
    structural overhead) rather than only the conversation ``content+role`` the
    old heuristic measured, so it is ~4.8x larger for an identical conversation.

    On cold start — the first turn after a process restart, when the store is
    empty for this key — the real size is unknown, so we fall back to the
    existing ``estimate_tokens`` heuristic over the conversation messages. This
    keeps summarization functional even when no provider data is available yet,
    at the cost of being the old (history-only) approximation for that single
    turn. ``estimate_tokens`` itself is left untouched and remains the
    cold-start fallback.

    Args:
        db: Database instance (injected).
        storage_id: The user/chat identifier used as the store key's ``user_id``.
        thread_id: The message thread identifier (0 for the main thread).

    Returns:
        True if summarization was triggered, False otherwise.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    messages: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)

    real_prompt_size = UsageCacheStore().get(user_id=storage_id, thread_id=thread_id)
    if real_prompt_size is not None:
        tokens = real_prompt_size
    else:
        # Cold start: the store has no value for this key yet (e.g. first turn
        # after a process restart). Fall back to the history-only heuristic so
        # summarization still functions. estimate_tokens is intentionally left
        # intact and used here as-is.
        tokens = sum(msg.estimate_tokens for msg in messages)

    if tokens >= gpt_settings.max_history_tokens:
        await emergency_summarization(storage_id=storage_id, thread_id=thread_id)
        return True
    return False


@inject_database
async def generate_image(
    db: Database, interface: UserInterface, prompt: str, model: str | None = None, provider_name: str | None = None
) -> list[str] | list[BytesIO]:
    user = await db.get_or_create_user(user_id=interface.user_id)

    if provider_name:
        provider = user.providers.get(provider_name)
        selected_model = model
    else:
        provider = user.get_active_image_provider(thread_id=interface.thread_id)
        selected_model = user.get_active_image_model(thread_id=interface.thread_id)

    if not provider:
        raise NoProviderSelectedError("No image provider available")
    images = await provider.get_images(prompt=prompt, model=selected_model)
    if interface.user_id not in gpt_settings.image_generations_whitelist:
        await db.count_image(interface.user_id)
    return images


@inject_database
async def describe_image(
    db: Database,
    user_id: int,
    image: bytes,
    mime_type: str,
    model: str | None = None,
    prompt: str | None = None,
) -> VisionResultSchema:
    """Analyze an image using the user's vision provider.

    Args:
        db: Database instance.
        user_id: User ID.
        image: Image data as bytes.
        mime_type: MIME type of the image.
        model: Optional vision model override.
        prompt: Optional prompt to guide the vision analysis.

    Returns:
        Vision analysis result.
    """
    user = await db.get_or_create_user(user_id=user_id)
    provider = user.vision_provider

    return await provider.vision(image=image, model=model, mime_type=mime_type, prompt=prompt)


@inject_database
async def ocr_pdf(
    db: Database,
    user_id: int,
    pdf: bytes,
    model: str | None = None,
) -> VisionResultSchema:
    """Extract text from a PDF using the user's OCR-capable provider.

    Args:
        db: Database instance.
        user_id: User ID.
        pdf: PDF data as bytes.
        model: Optional model override.

    Returns:
        OCR result as VisionResultSchema.

    Raises:
        ValueError: If no OCR-capable provider is available.
    """
    user = await db.get_or_create_user(user_id=user_id)
    provider = user.ocr_provider
    return await provider.ocr(pdf=pdf, model=model)


# @cached(ttl=3600)
@inject_database
async def get_user_cached_models(
    db: Database, user_id: int, image_generation: bool = False, video_generation: bool = False
) -> list[ModelChangeSchema]:
    user = await db.get_or_create_user(user_id=user_id)
    return await user.get_available_models(image_generation=image_generation, video_generation=video_generation)


@inject_database
async def get_models_available(
    db: Database, user_id: int, image_generation: bool = False, video_generation: bool = False, thread_id: int = 0
) -> list[ModelChangeSchema]:
    user = await db.get_or_create_user(user_id=user_id)
    user_models = await get_user_cached_models(
        user_id=user_id, image_generation=image_generation, video_generation=video_generation
    )

    if not user_models:
        return []

    available_models = deepcopy(user_models)

    if video_generation:
        active_provider = user.get_active_video_provider(thread_id=thread_id)
        active_model = user.get_active_video_model(thread_id=thread_id) or active_provider.default_video_model
    elif image_generation:
        active_provider = user.get_active_image_provider(thread_id=thread_id)
        active_model = user.get_active_image_model(thread_id=thread_id) or active_provider.default_image_model
    else:
        active_provider = user.get_active_llm_provider(thread_id=thread_id)
        active_model = user.get_active_llm_model(thread_id=thread_id) or active_provider.default_model

    for model in available_models:
        if model.name == active_model and model.provider == active_provider.name:
            model.display_name = f"🟢 {model.display_name}️"
    return available_models


@inject_database
async def user_has_reached_images_generation_limit(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    return user.has_reached_image_limits


@inject_database
async def set_api_key(db: Database, user_id: int, api_key: str, provider_name: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.tokens[provider_name] = api_key
    await db.save_user(user)
    return None


@inject_database
async def get_info(db: Database, user_id: int) -> str:
    user = await db.get_or_create_user(user_id=user_id)
    return user.info


@inject_database
async def set_info(db: Database, user_id: int, new_info: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.info = new_info
    await db.save_user(user)


@inject_database
async def set_thread_notes(db: Database, user_id: int, thread_id: int, notes: str) -> None:
    """Replace the thread-scoped notes for a specific thread.

    The notes text is stored per thread on the user object and persisted with
    the whole-user save. Full-replace semantics: the stored text is swapped
    entirely, there is no merging. An empty string is allowed and clears the
    notes for the thread. Users loaded from older persisted records are
    handled naturally by the pydantic default of the dict field.

    Args:
        db: The database instance.
        user_id: The storage ID of the user.
        thread_id: The ID of the thread to set the notes for.
        notes: The complete new notes text; an empty string clears the notes.
    """
    user = await db.get_or_create_user(user_id=user_id)
    user.thread_notes[thread_id] = notes
    await db.save_user(user)


@inject_database
async def activate_llm_skill(db: Database, user_id: int, skill_name: str, skill_payload: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.llm_skills[skill_name] = skill_payload
    await db.save_user(user)


@inject_database
async def deactivate_llm_skill(db: Database, user_id: int, skill_name: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    if skill_name not in user.llm_skills.keys():
        raise ValueError(f"The skill {skill_name} seems never been activated")
    user.llm_skills.pop(skill_name)
    await db.save_user(user)


@inject_database
async def set_working_dir(db: Database, user_id: int, new_wd: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.working_dir = new_wd
    await db.save_user(user)


@inject_database
async def get_cwd(db: Database, user_id: int) -> str:
    user = await db.get_or_create_user(user_id=user_id)
    return user.working_dir


@inject_database
async def set_thread_working_dir(db: Database, user_id: int, thread_id: int, new_wd: str) -> None:
    """Set the working directory override for a specific thread.

    The path is normalized via ``Path(...).expanduser()`` only — no existence
    check is performed. Users loaded from older persisted records are handled
    naturally by the pydantic default of the new dict field.

    Args:
        db: The database instance.
        user_id: The storage ID of the user.
        thread_id: The ID of the thread to set the working directory for.
        new_wd: The new working directory path.
    """
    user = await db.get_or_create_user(user_id=user_id)
    user.thread_working_dirs[thread_id] = str(Path(new_wd).expanduser())
    await db.save_user(user)


@inject_database
async def get_moderation_provider(db: Database, user_id: int) -> "Provider":
    user = await db.get_or_create_user(user_id=user_id)
    return user.moderation_provider


@inject_database
async def drop_tool_call_history(db: Database, storage_id: int, thread_id: int) -> None:
    user = await db.get_or_create_user(user_id=storage_id)
    chat_history: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)
    await reset_chat_history(storage_id=storage_id, thread_id=thread_id)
    for message in chat_history:
        if message.role == "tool":
            continue
        message.tool_calls = None
        message.tool_call_id = None
        await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)


# @inject_database
# async def drop_history(db: Database, storage_id: int, thread_id: int) -> None:
#     user = await db.get_or_create_user(user_id=storage_id)
#     chat_history: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)
#     await reset_chat_history(storage_id=storage_id, thread_id=thread_id)
#     await db.add_message(user=user, message=chat_history[0], ttl=gpt_settings.messages_ttl, thread_id=thread_id)


@inject_database
async def save_thread_name(db: Database, storage_id: int, thread_id: int, name: str) -> None:
    """Save the name of a specific thread for the user.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        thread_id: The ID of the thread.
        name: The name to save for the thread.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    user.thread_names[thread_id] = name
    await db.save_user(user)


@inject_database
async def delete_thread_from_map(db: Database, storage_id: int, thread_id: int) -> None:
    """Delete a thread mapping from the user's saved thread names.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        thread_id: The ID of the thread to delete.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    user.thread_names.pop(thread_id)
    await db.save_user(user)


@inject_database
async def clone_thread_messages(
    db: Database,
    storage_id: int,
    old_thread_id: int,
    new_thread_id: int,
    name: str | None = None,
) -> int:
    """Clone messages and settings from an old thread to a new thread.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        old_thread_id: The ID of the thread to clone from.
        new_thread_id: The ID of the thread to clone to.
        name: The name of the new thread. Defaults to None.

    Returns:
        The number of messages that were cloned.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    existing_messages = await db.get_conversation_messages(user=user, thread_id=old_thread_id)

    for i, message in enumerate(existing_messages):
        cloned = deepcopy(message)
        cloned.id = time.time_ns() + i
        await db.add_message(user=user, message=cloned, thread_id=new_thread_id)

    # Re-fetch the user: LocalStorage.add_message persists into a reloaded copy, so
    # the object above is stale and saving it would wipe the just-cloned messages.
    user = await db.get_or_create_user(user_id=storage_id)

    if old_thread_id in user.thread_selected_llm:
        user.thread_selected_llm[new_thread_id] = user.thread_selected_llm[old_thread_id]
    if old_thread_id in user.thread_selected_image_model:
        user.thread_selected_image_model[new_thread_id] = user.thread_selected_image_model[old_thread_id]
    if old_thread_id in user.thread_working_dirs:
        user.thread_working_dirs[new_thread_id] = user.thread_working_dirs[old_thread_id]

    user.thread_names[new_thread_id] = name or str(new_thread_id)

    await db.save_user(user)
    return len(existing_messages)
