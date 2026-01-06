import datetime
import json
from datetime import timezone
from io import BytesIO
from typing import Any, cast

from aiocache import cached
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.lock_manager import LockManager
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database


@inject_database
async def set_active_model(db: Database, user_id: int, model: ModelChangeSchema) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    if model.image_generation:
        user.selected_image_model_name = model.name
        user.selected_image_provider_name = model.provider
    else:
        user.selected_gpt_model_name = model.name
        user.selected_gpt_provider_name = model.provider
    await db.save_user(user)


@inject_database
async def reset_chat_history(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    await db.drop_messages(user=user)


@inject_database
async def emergency_summarization(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)

    chat_history = await db.get_messages(user=user)
    chat_history_string = str(msg for msg in chat_history if not any((msg.get("tool_calls"), msg.get("tool_call_id"))))
    user_messages: list[Message] = [Message(role="user", content=chat_history_string)]

    response, _ = await user.active_gpt_provider.get_chat_response(
        messages=user_messages,
        user=user,
        system_prompt="Summarize this conversation, keeping the most important and useful information using English.",
    )
    initial_message = Message(role="user", content="What we were talking about?")
    answer_message = Message(role="assistant", content=response.answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=initial_message, ttl=gpt_settings.messages_ttl)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)


@inject_database
async def get_gtp_chat_answer(
    db: Database,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text_prompt: str | None = None,
    voice_prompt: BytesIO | None = None,
) -> ChatResponseSchema:
    user = await db.get_or_create_user(user_id=user_id)
    lock = await LockManager().get_lock(key=user_id)

    async with lock:
        user_content = None
        if text_prompt:
            user_content = text_prompt
        elif voice_prompt:
            user_content = await cast(Any, user.stt_provider).transcribe(audio=voice_prompt)

        if not user_content:
            raise ValueError("No text or voice provided")

        conversation_messages: list[Message] = await db.get_conversation_messages(user=user)

        user_prompt = {
            "prompt": user_content,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "transcribed_from_voice_message": bool(voice_prompt),
        }

        user_message = Message(role="user", content=json.dumps(user_prompt))
        conversation_messages.append(user_message)

        chat_response, new_messages = await user.active_gpt_provider.get_chat_response(
            messages=conversation_messages,
            user=user,
            model=user.selected_gpt_model_name,
            update=update,
            context=context,
        )
        await db.add_message(user=user, message=user_message, ttl=gpt_settings.messages_ttl)
        for message in new_messages:
            await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl)
        return chat_response


@inject_database
async def check_history_and_summarize(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    messages = await db.get_messages(user=user)
    # Roughly estimating how many tokens the current conversation history will comprise. It is possible to calculate
    # this accurately, but the modules that can be used for this need to be separately built for armv7, which is
    # difficult to do right now (but will be done further, I hope).
    if len(str(messages)) / 4 >= gpt_settings.max_history_tokens:
        await emergency_summarization(user_id=user_id)
        return True
    return False


@inject_database
async def generate_image(
    db: Database, user_id: int, prompt: str, model: str | None = None, provider_name: str | None = None
) -> list[str] | list[BytesIO]:
    user = await db.get_or_create_user(user_id=user_id)

    if provider_name:
        provider = user.providers.get(provider_name)
        selected_model = model
    elif user.selected_image_provider_name:
        provider = user.active_image_provider
        selected_model = model or user.selected_image_model_name
    else:
        provider = user.active_image_provider
        selected_model = None
    if not provider:
        raise ValueError(f"User {user_id}: no image provider available.")
    images = await provider.get_images(prompt=prompt, model=selected_model)
    if user_id not in gpt_settings.image_generations_whitelist:
        await db.count_image(user_id)
    return images


@cached(ttl=6000)
@inject_database
async def get_models_available(db: Database, user_id: int, image_generation: bool = False) -> list[ModelChangeSchema]:
    user = await db.get_or_create_user(user_id=user_id)
    return await user.get_available_models(image_generation=image_generation)


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
async def set_working_dir(db: Database, user_id: int, new_wd: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.working_dir = new_wd
    await db.save_user(user)


@inject_database
async def get_cwd(db: Database, user_id: int) -> str:
    user = await db.get_or_create_user(user_id=user_id)
    return user.working_dir


@inject_database
async def drop_tool_call_history(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    chat_history: list[Message] = await db.get_conversation_messages(user=user)
    await reset_chat_history(user_id=user_id)
    for message in chat_history:
        if message.role == "tool":
            continue
        message.tool_calls = None
        message.tool_call_id = None
        await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl)


@inject_database
async def summarize_history(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    chat_history: list[Message] = await db.get_conversation_messages(user=user)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=chat_history[0], ttl=gpt_settings.messages_ttl)
