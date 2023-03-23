import logging

from openai.error import InvalidRequestError, RateLimitError, TryAgain
from telegram import InputMediaPhoto, Update, constants
from telegram.ext import ContextTypes

from chibi.services.user import generate_image, get_gtp_chat_answer, reset_chat_history
from chibi.utils import send_message


async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user = update.message.from_user
    logging.info(f"{user.name} sent a new message: {update.message.text}")

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    prompt = update.message.text
    if prompt.startswith("/ask"):
        prompt = prompt.replace("/ask", "", 1).strip()

    try:
        gpt_answer, usage = await get_gtp_chat_answer(user_id=update.message.from_user.id, prompt=prompt)
        usage_message = (
            f"Tokens used: {str(usage.get('total_tokens', 'n/a'))} "
            f"({str(usage.get('prompt_tokens', 'n/a'))} prompt, "
            f"{str(usage.get('completion_tokens', 'n/a'))} completion)"
        )
        logging.info(f"{user.name} got GPT answer. {usage_message}")
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
    await send_message(update=update, context=context, text=gpt_answer, parse_mode=constants.ParseMode.MARKDOWN)


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
    image_urls = await generate_image(user_id=update.message.from_user.id, prompt=prompt)
    await context.bot.send_media_group(
        chat_id=chat_id,
        media=[InputMediaPhoto(url) for url in image_urls],
        reply_to_message_id=update.message.message_id,
    )
