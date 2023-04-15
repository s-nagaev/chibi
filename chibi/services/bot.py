import asyncio

from loguru import logger
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
    constants,
)
from telegram.ext import ContextTypes

from chibi.config import application_settings
from chibi.services.gpt import api_key_is_valid
from chibi.services.user import (
    check_history_and_summarize,
    generate_image,
    get_gtp_chat_answer,
    get_models_available,
    reset_chat_history,
    set_active_model,
    set_api_key,
)
from chibi.utils import (
    api_key_is_plausible,
    get_telegram_chat,
    get_telegram_message,
    get_telegram_user,
    handle_gpt_exceptions,
    send_gpt_answer_message,
    send_message,
    user_can_use_gpt4,
)


async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    model_name = query.data
    telegram_user = get_telegram_user(update=update)
    await set_active_model(user_id=telegram_user.id, model_name=model_name)
    logger.info(f"{telegram_user.name} switched to model '{model_name}'")
    await query.edit_message_text(text=f"Selected model: {query.data}")


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
        f"{telegram_user.name} sent a new message{': ' + prompt_to_log if application_settings.log_prompt_data else ''}"
    )

    get_gtp_chat_answer_task = asyncio.ensure_future(get_gtp_chat_answer(user_id=telegram_user.id, prompt=prompt))

    while not get_gtp_chat_answer_task.done():
        await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.TYPING)
        await asyncio.sleep(2.5)

    gpt_answer, usage = await get_gtp_chat_answer_task
    usage_message = (
        f"Tokens used: {str(usage.get('total_tokens', 'n/a'))} "
        f"({str(usage.get('prompt_tokens', 'n/a'))} prompt, "
        f"{str(usage.get('completion_tokens', 'n/a'))} completion)"
    )
    answer_to_log = gpt_answer.replace("\r", " ").replace("\n", " ")
    logged_answer = f"Answer: {answer_to_log}" if application_settings.log_prompt_data else ""

    logger.info(f"{telegram_user.name} got GPT answer. {usage_message}. {logged_answer}")
    await send_gpt_answer_message(gpt_answer=gpt_answer, update=update, context=context)
    history_is_summarized = await check_history_and_summarize(user_id=telegram_user.id)
    if history_is_summarized:
        logger.info(f"{telegram_user.name}'s history successfully summarized.")


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_chat = get_telegram_chat(update=update)
    telegram_user = get_telegram_user(update=update)
    logger.info(f"{telegram_user.name} conversation history reset.")

    await reset_chat_history(user_id=telegram_user.id)
    await context.bot.send_message(chat_id=telegram_chat.id, text="Done!")


@handle_gpt_exceptions
async def handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    if not telegram_message.text:
        return None

    prompt = telegram_message.text.replace("/imagine", "", 1).strip()
    if not prompt:
        await context.bot.send_message(
            chat_id=telegram_chat.id,
            reply_to_message_id=telegram_message.message_id,
            text="Please provide some description (e.g. /imagine cat)",
        )
        return None

    logger.info(
        f"{telegram_user.name} sent image generation request"
        f"{': ' + prompt if application_settings.log_prompt_data else ''}"
    )
    generate_image_task = asyncio.ensure_future(generate_image(user_id=telegram_user.id, prompt=prompt))

    # The user finds it psychologically easier to wait for a response from the chatbot when they see its activity
    # during the entire waiting time.
    while not generate_image_task.done():
        await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)
        await asyncio.sleep(2.5)

    image_urls = await generate_image_task

    await context.bot.send_media_group(
        chat_id=telegram_chat.id,
        media=[InputMediaPhoto(url) for url in image_urls],
        reply_to_message_id=telegram_message.message_id,
    )


async def handle_openai_key_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    logger.info(f"{telegram_user.name} provides ones API Key.")

    if not telegram_message.text:
        return None

    api_key = telegram_message.text.replace("/set_openai_key", "", 1).strip()
    error_msg = (
        "Sorry, but incorrect API key provided. You can find your API key at "
        "https://platform.openai.com/account/api-keys"
    )
    if not api_key_is_plausible(api_key=api_key):
        await send_message(update=update, context=context, text=error_msg)
        logger.warning(f"{telegram_user.username} provided improbable key.")
        return

    if not await api_key_is_valid(api_key=api_key):
        await send_message(update=update, context=context, text=error_msg)
        logger.warning(f"{telegram_user.username} provided incorrect API key.")
        return

    await set_api_key(user_id=telegram_user.id, api_key=api_key)
    msg = (
        "Your OpenAI API Key successfully set, my functionality unlocked! 🦾\n\n"
        "Now you also may check available models in /menu."
    )
    await send_message(update=update, context=context, reply=False, text=msg)
    await context.bot.delete_message(chat_id=telegram_chat.id, message_id=telegram_message.message_id)
    logger.info(f"{telegram_user.name} successfully set up OpenAPI Key.")


async def handle_available_model_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    telegram_user = get_telegram_user(update=update)
    can_use_gpt4 = user_can_use_gpt4(tg_user=telegram_user)
    models_available = await get_models_available(user_id=telegram_user.id, include_gpt4=can_use_gpt4)
    keyboard = [[InlineKeyboardButton(model.upper(), callback_data=model)] for model in models_available]
    return InlineKeyboardMarkup(keyboard)
