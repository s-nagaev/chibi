import asyncio
from typing import cast

from loguru import logger
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
    constants,
)
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.constants import UserAction, UserContext
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers import registered_providers
from chibi.services.user import (
    check_history_and_summarize,
    generate_image,
    get_gtp_chat_answer,
    get_models_available,
    reset_chat_history,
    set_active_model,
    set_api_key,
    user_has_reached_images_generation_limit,
)
from chibi.utils import (
    chat_data,
    download_image,
    get_telegram_chat,
    get_telegram_message,
    get_telegram_user,
    handle_gpt_exceptions,
    send_gpt_answer_message,
    send_message,
    set_user_action,
    set_user_context,
    user_data,
)


@handle_gpt_exceptions
async def handle_model_selection(
    update: Update, context: ContextTypes.DEFAULT_TYPE, model: ModelChangeSchema, query: CallbackQuery
) -> None:
    telegram_user = get_telegram_user(update=update)
    await set_active_model(user_id=telegram_user.id, model=model)
    logger.info(f"{user_data(update)} switched to model '{model.name} ({model.provider})'")
    await query.edit_message_text(text=f"Selected model: '{model.name} ({model.provider})'")


@handle_gpt_exceptions
async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    prompt = telegram_message.text

    if not prompt:
        return None

    if prompt.startswith("/ask"):
        prompt = prompt.replace("/ask", "", 1).strip()

    prompt_to_log = prompt.replace("\r", " ").replace("\n", " ")

    logger.info(
        f"{user_data(update)} sent a new message in the {chat_data(update)}"
        f"{': ' + prompt_to_log if application_settings.log_prompt_data else ''}"
    )

    get_gtp_chat_answer_task = asyncio.ensure_future(get_gtp_chat_answer(user_id=telegram_user.id, prompt=prompt))

    while not get_gtp_chat_answer_task.done():
        await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.TYPING)
        await asyncio.sleep(2.5)

    chat_response: ChatResponseSchema = await get_gtp_chat_answer_task
    # usage_message = (
    #     f"Tokens used: {str(usage.get('total_tokens', 'n/a'))} "
    #     f"({str(usage.get('prompt_tokens', 'n/a'))} prompt, "
    #     f"{str(usage.get('completion_tokens', 'n/a'))} completion)"
    # )
    answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
    logged_answer = f"Answer: {answer_to_log}" if application_settings.log_prompt_data else ""

    logger.info(
        f"{user_data(update)} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {chat_data(update)}. {logged_answer}"
    )
    await send_gpt_answer_message(gpt_answer=chat_response.answer, update=update, context=context)
    history_is_summarized = await check_history_and_summarize(user_id=telegram_user.id)
    if history_is_summarized:
        logger.info(f"{user_data(update)}: history successfully summarized.")


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_chat = get_telegram_chat(update=update)
    telegram_user = get_telegram_user(update=update)
    logger.info(f"{user_data(update)}: conversation history reset.")

    await reset_chat_history(user_id=telegram_user.id)
    await context.bot.send_message(chat_id=telegram_chat.id, text="Done!")


@handle_gpt_exceptions
async def handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    set_user_action(context=context, action=UserAction.NONE)
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    if not telegram_message.text:
        return None

    if await user_has_reached_images_generation_limit(user_id=telegram_user.id):
        await context.bot.send_message(
            chat_id=telegram_chat.id,
            reply_to_message_id=telegram_message.message_id,
            text=(
                f"Sorry, you have reached your monthly images generation limit "
                f"({gpt_settings.image_generations_monthly_limit}). Please, try again later."
            ),
        )
        return None

    logger.info(
        f"{user_data(update)} sent image generation request in the {chat_data(update)}"
        f"{': ' + prompt if application_settings.log_prompt_data else ''}"
    )
    generate_image_task = asyncio.ensure_future(generate_image(user_id=telegram_user.id, prompt=prompt))

    # The user finds it psychologically easier to wait for a response from the chatbot when they see its activity
    # during the entire waiting time.
    while not generate_image_task.done():
        await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)
        await asyncio.sleep(2.5)

    image_data = await generate_image_task
    if isinstance(image_data[0], str):
        image_files = [await download_image(image_url=cast(str, url)) for url in image_data]
        try:
            await context.bot.send_media_group(
                chat_id=telegram_chat.id,
                media=[InputMediaPhoto(url) for url in image_files],
                reply_to_message_id=telegram_message.message_id,
            )
        except Exception as e:
            logger.exception(
                f"{user_data(update)} image generation request succeeded, but we couldn't send the image "
                f"due to exception: {e}. Trying to send if via text message..."
            )
            image_urls = cast(list[str], image_data)
            await send_message(
                update=update, context=context, text="\n".join(image_urls), disable_web_page_preview=False
            )
    else:
        await context.bot.send_media_group(
            chat_id=telegram_chat.id,
            media=[InputMediaPhoto(img) for img in image_data],
            reply_to_message_id=telegram_message.message_id,
        )
    log_message = f"{user_data(update)} got a successfully generated image(s)"
    if application_settings.log_prompt_data and isinstance(image_data[0], str):
        log_message += f": {image_data}"
    logger.info(log_message)


async def handle_provider_api_key_set(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    provider_name: str,
) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    logger.info(f"{telegram_user.name} provides API Key for provider '{provider_name}'.")

    api_key = telegram_message.text.strip() if telegram_message.text else None
    if not api_key:
        return None
    provider = registered_providers.get(provider_name)
    if not provider:
        return None

    if not await provider(token=api_key).api_key_is_valid():
        error_msg = "Sorry, but API key you have provided does not seem correct."
        await send_message(update=update, context=context, text=error_msg)
        logger.warning(f"{user_data(update)} provided invalid key.")
        return None

    await set_api_key(user_id=telegram_user.id, api_key=api_key, provider_name=provider_name)
    msg = f"Your {provider_name} API Key successfully set! 🦾\n\n" "Now you may check available models in /gpt_models."
    await send_message(update=update, context=context, reply=False, text=msg)
    try:
        await context.bot.delete_message(chat_id=telegram_chat.id, message_id=telegram_message.message_id)
    except Exception:
        pass
    logger.info(f"{user_data(update)} successfully set up {provider_name} Key.")


@handle_gpt_exceptions
async def handle_available_model_options(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_generation: bool = False,
) -> InlineKeyboardMarkup:
    telegram_user = get_telegram_user(update=update)
    models_available = await get_models_available(user_id=telegram_user.id, image_generation=image_generation)
    mapped_models: dict[str, ModelChangeSchema] = {str(k): model for k, model in enumerate(models_available)}
    set_user_context(context=context, key=UserContext.MAPPED_MODELS, value=mapped_models)
    keyboard = [
        [InlineKeyboardButton(f"{model.name.lower()} ({model.provider})", callback_data=key)]
        for key, model in mapped_models.items()
    ]
    keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
    return InlineKeyboardMarkup(keyboard)


async def handle_available_provider_options() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(name, callback_data=name)]
        for name, klass in registered_providers.items()
        if name != "Cloudflare"
        # Temporary removing the Cloudflare provider from the "public mode"
        # because we need to handle account id setting first. Will provide
        # such a support in one of the following releases.
    ]
    keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
    return InlineKeyboardMarkup(keyboard)
