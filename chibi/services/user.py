from io import BytesIO

from openai.types.chat import ChatCompletionUserMessageParam

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database
from chibi.types import ChatCompletionMessageSchema, UserMessageSchema


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
async def summarize(db: Database, user_id: int) -> None:
    user = await db.get_or_create_user(user_id=user_id)

    chat_history = await db.get_messages(user=user)

    user_messages: list[ChatCompletionMessageSchema] = [
        ChatCompletionUserMessageParam(role="user", content=str(chat_history))
    ]

    response = await user.active_gpt_provider.get_chat_response(
        messages=user_messages,
        system_prompt="Summarize this conversation in 700 characters or less using English.",
    )
    initial_message = Message(role="user", content="What we were talking about?")
    answer_message = Message(role="assistant", content=response.answer)
    await reset_chat_history(user_id=user_id)
    await db.add_message(user=user, message=initial_message, ttl=gpt_settings.messages_ttl)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)


@inject_database
async def get_gtp_chat_answer(db: Database, user_id: int, prompt: str) -> ChatResponseSchema:
    user = await db.get_or_create_user(user_id=user_id)
    conversation_messages: list[ChatCompletionMessageSchema] = await db.get_conversation_messages(user=user)
    conversation_messages.append(UserMessageSchema(role="user", content=prompt))
    chat_response = await user.active_gpt_provider.get_chat_response(
        messages=conversation_messages, model=user.selected_gpt_model_name
    )

    answer_message = Message(role="assistant", content=chat_response.answer)
    query_message = Message(role="user", content=prompt)
    await db.add_message(user=user, message=query_message, ttl=gpt_settings.messages_ttl)
    await db.add_message(user=user, message=answer_message, ttl=gpt_settings.messages_ttl)
    return chat_response


@inject_database
async def check_history_and_summarize(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    messages = await db.get_messages(user=user)
    # Roughly estimating how many tokens the current conversation history will comprise. It is possible to calculate
    # this accurately, but the modules that can be used for this need to be separately built for armv7, which is
    # difficult to do right now (but will be done further, I hope).
    if len(str(messages)) / 4 >= gpt_settings.max_history_tokens:
        await summarize(user_id=user_id)
        return True
    return False


@inject_database
async def generate_image(db: Database, user_id: int, prompt: str) -> list[str] | list[BytesIO]:
    user = await db.get_or_create_user(user_id=user_id)
    images = await user.active_image_provider.get_images(prompt=prompt, model=user.selected_image_model_name)
    if user_id not in gpt_settings.image_generations_whitelist:
        await db.count_image(user_id)
    return images


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
