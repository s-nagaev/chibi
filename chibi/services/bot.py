import asyncio
from io import BytesIO

from loguru import logger
from telegram import (
    CallbackQuery,
    File,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
    constants,
)
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings
from chibi.constants import UserAction, UserContext
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers import RegisteredProviders
from chibi.services.providers.tools import ToolResponse
from chibi.services.providers.utils import get_usage_msg
from chibi.services.user import (
    check_history_and_summarize,
    generate_image,
    get_llm_chat_completion_answer,
    get_models_available,
    reset_chat_history,
    set_active_model,
    set_api_key,
    user_has_reached_images_generation_limit,
)
from chibi.utils.app import handle_gpt_exceptions
from chibi.utils.telegram import (
    chat_data,
    get_telegram_chat,
    get_telegram_message,
    get_telegram_user,
    send_gpt_answer_message,
    send_images,
    send_message,
    set_user_action,
    set_user_context,
    user_data,
)


@handle_gpt_exceptions
async def handle_model_selection(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    model: ModelChangeSchema,
    query: CallbackQuery,
) -> None:
    telegram_user = get_telegram_user(update=update)
    await set_active_model(user_id=telegram_user.id, model=model)
    logger.info(f"{user_data(update)} switched to model '{model.name} ({model.provider})'")
    await query.edit_message_text(text=f"Selected model: '{model.name} ({model.provider})'")


async def handle_tool_response(tool_response: ToolResponse, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    chat_response: ChatResponseSchema = await get_llm_chat_completion_answer(
        user_id=telegram_user.id, tool_message=tool_response, context=context, update=update
    )
    usage_message = get_usage_msg(chat_response.usage)

    if len(chat_response.answer) <= 5 and "ACK" in chat_response.answer:
        logger.info(
            f"[{user_data(update)}-{chat_data(update)}] LLM silently received tool result "
            f"(answer: {chat_response.answer}). No user notification required. {usage_message}"
        )
        return None

    if application_settings.log_prompt_data:
        answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
        logged_answer = f"Answer: {answer_to_log}"
    else:
        logged_answer = ""

    logger.info(
        f"{user_data(update)} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {chat_data(update)}. {logged_answer} {usage_message}"
    )

    await send_gpt_answer_message(gpt_answer=chat_response.answer, update=update, context=context)


@handle_gpt_exceptions
async def handle_user_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    text_prompt = telegram_message.text

    if telegram_message.voice:
        file_id = telegram_message.voice.file_id
        file: File = await context.bot.get_file(file_id)
        voice_prompt = BytesIO()
        await file.download_to_memory(out=voice_prompt)
        voice_prompt.seek(0)
    else:
        voice_prompt = None

    if not text_prompt and not voice_prompt:
        return None

    if text_prompt:
        if text_prompt.startswith("/ask"):
            text_prompt = text_prompt.replace("/ask", "", 1).strip()

    prompt_to_log = text_prompt.replace("\r", " ").replace("\n", " ") if text_prompt else "voice message"

    logger.info(
        f"{user_data(update)} sent a new message in the {chat_data(update)}"
        f"{': ' + prompt_to_log if application_settings.log_prompt_data else ''}"
    )

    get_gtp_chat_answer_task = asyncio.ensure_future(
        get_llm_chat_completion_answer(
            user_id=telegram_user.id,
            user_text_message=text_prompt,
            user_voice_message=voice_prompt,
            context=context,
            update=update,
        )
    )

    while not get_gtp_chat_answer_task.done():
        await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.TYPING)
        await asyncio.sleep(2.5)

    chat_response: ChatResponseSchema = await get_gtp_chat_answer_task
    usage = chat_response.usage
    usage_message = get_usage_msg(usage)

    if application_settings.log_prompt_data:
        answer_to_log = chat_response.answer.replace("\r", " ").replace("\n", " ")
        logged_answer = f"Answer: {answer_to_log}"
    else:
        logged_answer = ""

    if len(chat_response.answer) <= 5 and "ACK" in chat_response.answer:
        logger.info(
            f"[{user_data(update)}-{chat_data(update)}] LLM silently received user request "
            f"(answer: {chat_response.answer}). No user notification required. {usage_message}"
        )
        return None

    logger.info(
        f"{user_data(update)} got {chat_response.provider} ({chat_response.model}) answer in "
        f"the {chat_data(update)}. {logged_answer} {usage_message}"
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
    await send_images(images=image_data, update=update, context=context)
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
    provider = RegisteredProviders.get_class(provider_name)
    if not provider:
        return None

    if not await provider(token=api_key).api_key_is_valid():
        error_msg = "Sorry, but API key you have provided does not seem correct."
        await send_message(update=update, context=context, text=error_msg)
        logger.warning(f"{user_data(update)} provided invalid key.")
        return None

    await set_api_key(user_id=telegram_user.id, api_key=api_key, provider_name=provider_name)
    RegisteredProviders.register_as_available(provider=provider)

    msg = f"Your {provider_name} API Key successfully set! 🦾\n\nNow you may check available models in /gpt_models."
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
        [InlineKeyboardButton(f"{model.display_name.title()} ({model.provider})", callback_data=key)]
        for key, model in mapped_models.items()
    ]
    for model in models_available:
        logger.debug(f"{model.provider}: {model.name}")
    keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
    return InlineKeyboardMarkup(keyboard)


async def handle_available_provider_options() -> InlineKeyboardMarkup:
    keyboard = [
        [InlineKeyboardButton(name, callback_data=name)]
        for name, klass in RegisteredProviders.all.items()
        if name != "Cloudflare"
        # Temporary removing the Cloudflare provider from the "public mode"
        # because we need to handle account id setting first. Will provide
        # such a support in one of the following releases.
    ]
    keyboard.append([InlineKeyboardButton(text="CLOSE (SELECT NOTHING)", callback_data="-1")])
    return InlineKeyboardMarkup(keyboard)
