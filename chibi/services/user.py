from typing import Optional

from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.services.gpt import (
    get_chat_response,
    get_images_by_prompt,
    retrieve_available_models,
)
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database


@inject_database
async def get_api_key(db: Database, user_id: int, raise_on_absence: bool = True) -> Optional[str]:
    if api_key := gpt_settings.api_key:
        return api_key
    user = await db.get_or_create_user(user_id=user_id)
    if hasattr(user, "api_token") and user.api_token:
        return user.api_token
    if raise_on_absence:
        raise ValueError(f"User {user_id} does not have active OpenAI API Key.")
    return None


@inject_database
async def set_api_key(db: Database, user_id: int, api_key: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.api_token = api_key
    await db.save_user(user)


@inject_database
async def set_active_model(db: Database, user_id: int, model_name: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.gpt_model = model_name
    await db.save_user(user)


@inject_database
async def reset_chat_history(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    await db.drop_messages(user=user)


@inject_database
async def summarize(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = await get_api_key(user_id=user_id)

    chat_history = await db.get_messages(user=user)

    assistant_message = ChatCompletionAssistantMessageParam(
        role="assistant", content="Summarize this conversation in 700 characters or less"
    )

    user_message = ChatCompletionUserMessageParam(role="user", content=str(chat_history))

    query_messages: list[ChatCompletionMessageParam] = [assistant_message, user_message]
    answer, usage = await get_chat_response(
        api_key=openai_api_key, messages=query_messages, model=user.model, max_tokens=200
    )
    answer_message = Message(role="assistant", content=answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)


@inject_database
async def get_gtp_chat_answer(db: Database, user_id: int, prompt: str) -> tuple[str, CompletionUsage | None]:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = await get_api_key(user_id=user_id)

    query_message = Message(role="user", content=prompt)
    await db.add_message(user=user, message=query_message, ttl=gpt_settings.messages_ttl)
    conversation_messages: list[ChatCompletionMessageParam] = await db.get_conversation_messages(user=user)

    answer, usage = await get_chat_response(api_key=openai_api_key, messages=conversation_messages, model=user.model)
    answer_message = Message(role="assistant", content=answer)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)

    return answer, usage


@inject_database
async def check_history_and_summarize(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)

    # Roughly estimating how many tokens the current conversation history will comprise. It is possible to calculate
    # this accurately, but the modules that can be used for this need to be separately built for armv7, which is
    # difficult to do right now (but will be done further, I hope).
    if len(str(user.messages)) / 4 >= gpt_settings.max_history_tokens:
        await summarize(user_id=user_id)
        return True
    return False


async def generate_image(user_id: int, prompt: str) -> list[str]:
    openai_api_key = await get_api_key(user_id=user_id)
    return await get_images_by_prompt(api_key=openai_api_key, prompt=prompt)


async def get_models_available(user_id: int, include_gpt4: bool) -> list[str]:
    openai_api_key = await get_api_key(user_id=user_id)
    return await retrieve_available_models(api_key=openai_api_key, include_gpt4=include_gpt4)
