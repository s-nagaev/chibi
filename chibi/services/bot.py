import logging

from openai.error import InvalidRequestError, RateLimitError, TryAgain
from telegram import InputMediaPhoto, Update, constants
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.services.gpt import api_key_is_valid
from chibi.services.user import (
    generate_image,
    get_gtp_chat_answer,
    reset_chat_history,
    set_active_model,
    set_api_key,
)
from chibi.utils import api_key_is_plausible, send_message


async def handle_model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    model_name = query.data
    user = update.callback_query.from_user
    await set_active_model(user_id=user.id, model_name=model_name)
    logging.info(f"{user.name} switched to model '{model_name}'")
    await query.edit_message_text(text=f"Selected option: {query.data}")


async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user = update.message.from_user
    prompt = update.message.text
    if prompt.startswith("/ask"):
        prompt = prompt.replace("/ask", "", 1).strip()

    prompt_to_log = prompt.replace("\r", " ").replace("\n", " ")
    logging.info(f"{user.name} sent a new message: {prompt_to_log}")

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    try:
        gpt_answer, usage = await get_gtp_chat_answer(user_id=update.message.from_user.id, prompt=prompt)
        usage_message = (
            f"Tokens used: {str(usage.get('total_tokens', 'n/a'))} "
            f"({str(usage.get('prompt_tokens', 'n/a'))} prompt, "
            f"{str(usage.get('completion_tokens', 'n/a'))} completion)"
        )
        answer_to_log = gpt_answer.replace("\r", " ").replace("\n", " ")
        logging.info(f"{user.name} got GPT answer. {usage_message}. Answer: {answer_to_log}")
    except TryAgain as e:
        logging.error(f"{user.name} didn't get a GPT answer due to exception: {e}")
        await send_message(update=update, context=context, text="Service is overloaded. Please, try again later.")
        return
    except InvalidRequestError as e:
        logging.error(f"{user.name} got a InvalidRequestError: {e}")
        await send_message(update=update, context=context, text=f"🤐{e}")
        return
    except RateLimitError as e:
        logging.error(f"{user.name} reached a Rate Limit: {e}")
        await send_message(update=update, context=context, text=f"🤐Rate Limit exceeded: {e}")
        return
    except Exception as e:
        logging.error(f"{user.name} got an error: {e}")
        msg = (
            "I'm sorry, but there seems to be a little hiccup with your request at the moment. Would you mind "
            "trying again later? Don't worry, I'll be here to assist you whenever you're ready!"
        )
        await send_message(update=update, context=context, text=msg)
        return
    try:
        await send_message(update=update, context=context, text=gpt_answer, parse_mode=constants.ParseMode.MARKDOWN)
    except BadRequest as e:
        logging.error(
            f"{user.name} got a Telegram Bad Request error while receiving GPT answer: {e}. Trying to re-send it."
        )
        await send_message(update=update, context=context, text=gpt_answer)


async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.info(f"{update.message.from_user.name} conversation history reset.")
    await reset_chat_history(user_id=update.message.from_user.id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Done!")


async def handle_image_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    prompt = update.message.text.replace("/imagine", "", 1).strip()
    if not prompt:
        await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
            text="Please provide some description (e.g. /imagine cat)",
        )
        return

    logging.info(f"{update.message.from_user.name} sent image generation request: {prompt}")
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.UPLOAD_PHOTO)
    try:
        image_urls = await generate_image(user_id=update.message.from_user.id, prompt=prompt)
    except RateLimitError as e:
        logging.error(f"{update.message.from_user.name} reached a Rate Limit: {e}")
        await send_message(update=update, context=context, text=f"🤐Rate Limit exceeded: {e}")
        return
    except InvalidRequestError as e:
        logging.error(f"{update.message.from_user.name} got a InvalidRequestError: {e}")
        await send_message(update=update, context=context, text=f"🤐{e}")
        return
    except Exception as e:
        logging.error(f"{update.message.from_user.name} got an error: {e}")
        msg = (
            "I'm sorry, but there seems to be a little hiccup with your request at the moment. Would you mind "
            "trying again later? Don't worry, I'll be here to assist you whenever you're ready!"
        )
        await send_message(update=update, context=context, text=msg)
        return

    await context.bot.send_media_group(
        chat_id=chat_id,
        media=[InputMediaPhoto(url) for url in image_urls],
        reply_to_message_id=update.message.message_id,
    )


async def handle_openai_key_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logging.info(f"{user.name} provides ones API Key.")

    api_key = update.message.text.replace("/set_token", "", 1).strip()
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

    await set_api_key(user_id=user.id, token=api_key)
    msg = "Your OpenAI API Key successfully set, my functionality unlocked."
    await send_message(update=update, context=context, text=msg)
    logging.info(f"{user.username} successfully set up OpenAPI Key.")
