import asyncio
import logging

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
    constants,
)
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.services.gpt import api_key_is_valid
from chibi.services.user import (
    generate_image,
    get_gtp_chat_answer,
    get_models_available,
    reset_chat_history,
    set_active_model,
    set_api_key,
)
from chibi.utils import api_key_is_plausible, handle_gpt_exceptions, send_message


async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    model_name = query.data
    user = update.callback_query.from_user
    await set_active_model(user_id=user.id, model_name=model_name)
    logging.info(f"{user.name} switched to model '{model_name}'")
    await query.edit_message_text(text=f"Selected model: {query.data}")


@handle_gpt_exceptions
async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    prompt = update.message.text
    if prompt.startswith("/ask"):
        prompt = prompt.replace("/ask", "", 1).strip()

    prompt_to_log = prompt.replace("\r", " ").replace("\n", " ")
    logging.info(f"{user.name} sent a new message: {prompt_to_log}")

    get_gtp_chat_answer_task = asyncio.ensure_future(
        get_gtp_chat_answer(user_id=update.message.from_user.id, prompt=prompt)
    )

    while not get_gtp_chat_answer_task.done():
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
        await asyncio.sleep(2.5)

    gpt_answer, usage = await get_gtp_chat_answer_task
    usage_message = (
        f"Tokens used: {str(usage.get('total_tokens', 'n/a'))} "
        f"({str(usage.get('prompt_tokens', 'n/a'))} prompt, "
        f"{str(usage.get('completion_tokens', 'n/a'))} completion)"
    )
    answer_to_log = gpt_answer.replace("\r", " ").replace("\n", " ")
    logging.info(f"{user.name} got GPT answer. {usage_message}. Answer: {answer_to_log}")

    try:
        await send_message(update=update, context=context, text=gpt_answer, parse_mode=constants.ParseMode.MARKDOWN)
    except BadRequest as e:
        # Trying to handle a exception connected with markdown parsing: just re-sending the message in a text mode.
        logging.error(
            f"{user.name} got a Telegram Bad Request error while receiving GPT answer: {e}. "
            f"Trying to re-send it in plain text mode."
        )
        await send_message(update=update, context=context, text=gpt_answer)


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.info(f"{update.message.from_user.name} conversation history reset.")
    await reset_chat_history(user_id=update.message.from_user.id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


@handle_gpt_exceptions
async def handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    prompt = update.message.text.replace("/imagine", "", 1).strip()
    if not prompt:
        return await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
            text="Please provide some description (e.g. /imagine cat)",
        )

    logging.info(f"{update.message.from_user.name} sent image generation request: {prompt}")

    generate_image_task = asyncio.ensure_future(generate_image(user_id=update.message.from_user.id, prompt=prompt))

    # The user finds it psychologically easier to wait for a response from the chatbot when they see its activity
    # during the entire waiting time.
    while not generate_image_task.done():
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)
        await asyncio.sleep(2.5)

    image_urls = await generate_image_task

    await context.bot.send_media_group(
        chat_id=chat_id,
        media=[InputMediaPhoto(url) for url in image_urls],
        reply_to_message_id=update.message.message_id,
    )


async def handle_openai_key_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logging.info(f"{user.name} provides ones API Key.")

    api_key = update.message.text.replace("/set_openai_key", "", 1).strip()
    error_msg = (
        "Sorry, but incorrect API key provided. You can find your API key at "
        "https://platform.openai.com/account/api-keys"
    )
    if not api_key_is_plausible(api_key=api_key):
        await send_message(update=update, context=context, text=error_msg)
        logging.warning(f"{user.username} provided improbable key.")
        return

    if not await api_key_is_valid(api_key=api_key):
        await send_message(update=update, context=context, text=error_msg)
        logging.warning(f"{user.username} provided incorrect API key.")
        return

    await set_api_key(user_id=user.id, api_key=api_key)
    msg = (
        "Your OpenAI API Key successfully set, my functionality unlocked! 🦾\n\n "
        "Now you also may check available models in /menu."
    )
    await send_message(update=update, context=context, reply=False, text=msg)
    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
    logging.info(f"{user.username} successfully set up OpenAPI Key.")


async def handle_available_model_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    user = update.message.from_user
    can_use_gpt4 = gpt_settings.gpt4_enabled or (
        gpt_settings.gpt4_whitelist and user.username in gpt_settings.gpt4_whitelist
    )
    models_available = await get_models_available(user_id=user.id, include_gpt4=can_use_gpt4)
    keyboard = [[InlineKeyboardButton(model.upper(), callback_data=model)] for model in models_available]
    return InlineKeyboardMarkup(keyboard)
