import asyncio
import logging

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.services.gpt import (
    get_chat_response,
    get_images_by_prompt,
    retrieve_available_models,
)
from chibi.storage.abc import Database
from chibi.storage.database import inject_database


@inject_database
async def set_api_key(db: Database, user_id: int, api_key=str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.api_token = api_key
    await db.save_user(user)


@inject_database
async def set_active_model(db: Database, user_id: int, model_name=str) -> None:
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
    openai_api_key = user.api_token or gpt_settings.api_key
    logging.info(f"[User ID {user_id}] History is too long. Summarizing...")
    chat_history = await db.get_messages(user=user)
    query_messages = [
        {
            "role": "assistant",
            "content": "Summarize this conversation in 700 characters or less",
        },
        {"role": "user", "content": str(chat_history)},
    ]
    answer, usage = await get_chat_response(
        api_key=openai_api_key, messages=query_messages, model=user.model, max_tokens=200
    )
    answer_message = Message(role="assistant", content=answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)
    logging.info(f"[User ID {user_id}] History successfully summarized.")


@inject_database
async def get_gtp_chat_answer(db: Database, user_id: int, prompt: str) -> tuple[str, dict[str, int]]:
    user = await db.get_or_create_user(user_id=user_id)
    logging.warning(await db.get_messages(user=user))
    openai_api_key = user.api_token or gpt_settings.api_key

    query_message = Message(role="user", content=prompt)
    await db.add_message(user=user, message=query_message, ttl=gpt_settings.messages_ttl)
    conversation_messages = await db.get_messages(user=user)
    answer, usage = await get_chat_response(api_key=openai_api_key, messages=conversation_messages, model=user.model)
    answer_message = Message(role="assistant", content=answer)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)

    # Roughly estimating how many tokens the current conversation history will comprise. It is possible to calculate
    # this accurately, but the modules that can be used for this need to be separately built for armv7, which is
    # difficult to do right now (but will be done further).
    if len(str(user.messages)) / 4 >= gpt_settings.max_history_tokens:
        asyncio.create_task(summarize(user_id=user_id))
    return answer, usage


@inject_database
async def generate_image(db: Database, user_id: int, prompt: str) -> list[str]:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = user.api_token or gpt_settings.api_key
    return await get_images_by_prompt(api_key=openai_api_key, prompt=prompt)


@inject_database
async def get_models_available(db: Database, user_id: int, include_gpt4: bool) -> list[str]:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = user.api_token or gpt_settings.api_key
    return await retrieve_available_models(api_key=openai_api_key, include_gpt4=include_gpt4)
