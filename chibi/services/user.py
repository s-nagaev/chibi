import asyncio
import logging

from chibi.config import application_settings, gpt_settings
from chibi.models import Message
from chibi.services.gpt import get_chat_response, get_images_by_prompt
from chibi.storage.local import LocalStorage as Database

db = Database(application_settings.local_data_path)


TTL = gpt_settings.max_conversation_age_minutes * 60


async def set_token(user_id: int, token=str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.api_token = token
    await db.save_user(user)


async def reset_chat_history(user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    initial_message = Message(role="system", content=gpt_settings.assistant_prompt)
    user.messages = [
        initial_message,
    ]
    await db.save_user(user=user)


async def summarize(user_id: int) -> None:
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
    answer, usage = await get_chat_response(api_key=openai_api_key, messages=query_messages)
    answer_message = Message(role="assistant", content=answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=answer_message, ttl=TTL)


async def get_gtp_chat_answer(user_id: int, prompt: str) -> tuple[str, dict[str, int]]:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = user.api_token or gpt_settings.api_key

    query_message = Message(role="user", content=prompt)
    await db.add_message(user=user, message=query_message, ttl=TTL)
    conversation_messages = await db.get_messages(user=user)

    answer, usage = await get_chat_response(api_key=openai_api_key, messages=conversation_messages)

    answer_message = Message(role="assistant", content=answer)
    await db.add_message(user=user, message=answer_message, ttl=TTL)

    if len(str(user.messages)) / 4 >= 2000:
        asyncio.create_task(summarize(user_id=user_id))

    return answer, usage


async def generate_image(user_id: int, prompt: str) -> list[str]:
    user = await db.get_or_create_user(user_id=user_id)
    openai_api_key = user.api_token or gpt_settings.api_key
    return await get_images_by_prompt(api_key=openai_api_key, prompt=prompt)
