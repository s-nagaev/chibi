from openai.types.chat import ChatCompletionUserMessageParam

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database
from chibi.types import ChatCompletionMessageSchema, UserMessageSchema


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

    chat_history = await db.get_messages(user=user)

    task_message = ChatCompletionUserMessageParam(
        role="user", content="Summarize this conversation in 1000 characters or less"
    )

    user_messages = ChatCompletionUserMessageParam(role="user", content=str(chat_history))

    query_messages: list[ChatCompletionMessageSchema] = [user_messages, task_message]
    answer, usage = await user.active_provider.get_chat_response(messages=query_messages, max_tokens=250)
    answer_message = Message(role="assistant", content=answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)


@inject_database
async def get_gtp_chat_answer(db: Database, user_id: int, prompt: str) -> ChatResponseSchema:
    user = await db.get_or_create_user(user_id=user_id)
    conversation_messages: list[ChatCompletionMessageSchema] = await db.get_conversation_messages(user=user)
    conversation_messages.append(UserMessageSchema(role="user", content=prompt))

    chat_response = await user.active_provider.get_chat_response(messages=conversation_messages)

    answer_message = Message(role="assistant", content=chat_response.answer)
    query_message = Message(role="user", content=prompt)
    await db.add_message(user=user, message=query_message, ttl=gpt_settings.messages_ttl)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)
    return chat_response


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


@inject_database
async def generate_image(db: Database, user_id: int, prompt: str) -> list[str]:
    user = await db.get_or_create_user(user_id=user_id)
    if not user.openai:
        raise Exception  # TODO
    images = await user.openai.get_images(prompt=prompt)
    if user_id not in gpt_settings.image_generations_whitelist:
        await db.count_image(user_id)
    return images


@inject_database
async def get_models_available(db: Database, user_id: int) -> list[str]:
    user = await db.get_or_create_user(user_id=user_id)
    return await user.get_available_models()


@inject_database
async def user_has_reached_images_generation_limit(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    return user.has_reached_image_limits


@inject_database
async def set_api_key(db: Database, user_id: int, api_key: str) -> None:
    # TODO: just very fast & unreliable solution, will be updated in near future.
    user = await db.get_or_create_user(user_id=user_id)

    if 28 < len(api_key) < 36:
        user.mistralai_token = api_key
    elif 45 < len(api_key) < 56:
        user.openai_token = api_key
    elif len(api_key) > 96:
        user.anthropic_token = api_key
