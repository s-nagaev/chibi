import io
from functools import wraps
from typing import Any, Callable

from loguru import logger
from openai.error import InvalidRequestError, RateLimitError, TryAgain
from telegram import Chat as TelegramChat
from telegram import Message as TelegramMessage
from telegram import Update
from telegram import User as TelegramUser
from telegram import constants
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.services.user import get_api_key

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]


def get_telegram_user(update: Update) -> TelegramUser:
    if user := update.effective_user:
        return user
    raise ValueError(f"Telegram incoming update does not contain valid user data. Update ID: {update.update_id}")


def get_telegram_chat(update: Update) -> TelegramChat:
    if chat := update.effective_chat:
        return chat
    raise ValueError(f"Telegram incoming update does not contain valid chat data. Update ID: {update.update_id}")


def get_telegram_message(update: Update) -> TelegramMessage:
    if message := update.effective_message:
        return message
    raise ValueError(f"Telegram incoming update does not contain valid message data. Update ID: {update.update_id}")


async def send_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, reply: bool = True, **kwargs: Any
) -> TelegramMessage:
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)

    if reply:
        return await context.bot.send_message(
            chat_id=telegram_chat.id, reply_to_message_id=telegram_message.message_id, **kwargs
        )
    return await context.bot.send_message(chat_id=telegram_chat.id, **kwargs)


async def send_gpt_answer_message(gpt_answer: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_user = get_telegram_user(update=update)
    telegram_chat = get_telegram_chat(update=update)
    try:
        await send_message(update=update, context=context, text=gpt_answer, parse_mode=constants.ParseMode.MARKDOWN)
    except BadRequest as e:
        # Trying to handle an exception connected with markdown parsing: just re-sending the message in a text mode.
        logger.error(
            f"{telegram_user.name} got a Telegram Bad Request error while receiving GPT answer: {e}. "
            f"Trying to re-send it in plain text mode."
        )
        await send_message(update=update, context=context, text=gpt_answer)

        if "```" in gpt_answer:
            logger.info(
                f"{telegram_user.name} got and answer containing some code, but face with a markdown parse error. "
                f"Additionally sending answer as an attachment..."
            )
            file = io.BytesIO()
            file.write(gpt_answer.encode())
            file.seek(0)
            explain_message_text = (
                "Oops! 😯It looks like your answer contains some code, but Telegram can't display it properly. "
                "I'll additionally add your answer to the markdown file. 👇"
            )

            explain_message = await send_message(update=update, context=context, text=explain_message_text, reply=False)
            await context.bot.send_document(
                chat_id=telegram_chat.id,
                reply_to_message_id=explain_message.message_id,
                document=file,
                filename="answer.md",
            )


def user_is_allowed(tg_user: TelegramUser) -> bool:
    if not telegram_settings.users_whitelist:
        return True
    return any(identifier in telegram_settings.users_whitelist for identifier in (tg_user.id, tg_user.name))


def group_is_allowed(tg_chat: TelegramChat) -> bool:
    if not telegram_settings.groups_whitelist:
        return True
    return tg_chat.id in telegram_settings.groups_whitelist


def user_can_use_gpt4(tg_user: TelegramUser) -> bool:
    if gpt_settings.gpt4_enabled:
        return True
    if gpt_settings.gpt4_whitelist:
        return any(identifier in gpt_settings.gpt4_whitelist for identifier in (tg_user.id, tg_user.name))
    return False


def check_user_allowance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator controlling access to the chatbot.

    This deco checks:
        - if the specific user is allowed to interact with the chatbot, using ALLOW_BOTS and USERS_WHITELIST;
        - if the chatbot is allowed to be in a specific group, using a GROUPS_WHITELIST.

    If the specific user is disallowed to interact with the chatbot, the corresponding message will be sent.
    If the chatbot is disallowed to be in a specific group, it will send the corresponding message
    and leave it immediately.

    Args:
        func: async function that may rise openai exception.

    Returns:
        Wrapper function object.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        telegram_chat = get_telegram_chat(update=update)
        telegram_user = get_telegram_user(update=update)
        user_name = telegram_user.name or f"{telegram_user.first_name} ({telegram_user.id})"

        if telegram_user.is_bot and not telegram_settings.allow_bots:
            logger.warning(f"Bots are not allowed. {user_name}'s request ignored.")
            return None

        if telegram_chat.type in PERSONAL_CHAT_TYPES and not user_is_allowed(tg_user=telegram_user):
            logger.warning(f"{user_name} is not allowed to work with me. Request rejected.")
            await send_message(update=update, context=context, text=telegram_settings.message_for_disallowed_users)
            return None

        if telegram_chat.type in GROUP_CHAT_TYPES and not group_is_allowed(tg_chat=telegram_chat):
            message = (
                f"The group {telegram_chat.effective_name} (id: {telegram_chat.id}, link: {telegram_chat.link}) "
                f"does not exist in the whitelist. Leaving it..."
            )
            logger.warning(message)
            await context.bot.send_message(chat_id=telegram_chat.id, text=message, disable_web_page_preview=True)
            await telegram_chat.leave()
            return None

        return await func(*args, **kwargs)

    return wrapper


def handle_gpt_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator handling openai module's exceptions.

    If the specific exception occurred, handles it and sends the corresponding message.

    Args:
        func: async function that may rise openai exception.

    Returns:
        Wrapper function object.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        telegram_user = get_telegram_user(update=update)
        try:
            return await func(*args, **kwargs)
        except TryAgain as e:
            logger.error(f"{telegram_user.name} didn't get a GPT answer due to exception: {e}")
            await send_message(update=update, context=context, text="🥴Service is overloaded. Please, try again later.")
            return None

        except InvalidRequestError as e:
            logger.error(f"{telegram_user.name} got a InvalidRequestError: {e}")
            await send_message(update=update, context=context, text=f"😲{e}")
            return None

        except RateLimitError as e:
            logger.error(f"{telegram_user.name} reached a Rate Limit: {e}")
            await send_message(update=update, context=context, text=f"🤐Rate Limit exceeded: {e}")
            return None

        except Exception as e:
            logger.error(f"{telegram_user.name} got an error: {e}")
            msg = (
                "I'm sorry, but there seems to be a little hiccup with your request at the moment 😥 Would you mind "
                "trying again later? Don't worry, I'll be here to assist you whenever you're ready! 😼"
            )
            await send_message(update=update, context=context, text=msg)
            return None

    return wrapper


def api_key_is_plausible(api_key: str) -> bool:
    """Dummy pre-validator for OpenAI token.

    Just to discard obviously inappropriate data.

    Args:
        api_key: OpenAI token string.

    Returns:
        True if token provided looks like a token :) False otherwise.
    """

    if len(api_key) > 55 or len(api_key) < 51:
        return False
    if " " in api_key:
        return False
    return True


def check_openai_api_key(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator checking if the specific user has an OpenAI API Key.

    Args:
        func: async function that does any OpenAI API service interaction.

    Returns:
        Wrapper function object.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        telegram_user = get_telegram_user(update=update)
        user_name = telegram_user.name or f"{telegram_user.first_name} ({telegram_user.id})"

        if not await get_api_key(user_id=telegram_user.id, raise_on_absence=False):
            logger.info(f"{user_name} has no OpenAI API Key. Suggesting to apply one...")
            text = (
                "Oops! It looks like you didn't set your OpenAI API Key yet. "
                "Please, use /set_openai_key command, i.e.\n\n /set_openai_key sk-XXXXXXXXXXXXXXXXXXXXX\n\n"
                "You may find your key in your user settings after creating an OpenAI account "
                "(https://platform.openai.com/account/api-keys)."
            )
            await send_message(update=update, context=context, text=text, reply=False)
            return None

        return await func(*args, **kwargs)

    return wrapper


def log_application_settings() -> None:
    mode = (
        "PRIVATE (master OpenAI API Key is provided)"
        if gpt_settings.api_key
        else "PUBLIC (no master OpenAI API Key is provided)"
    )
    gpt4_state = "ENABLED" if gpt_settings.gpt4_enabled else "DISABLED"
    storage = "REDIS" if application_settings.redis else "LOCAL"

    message = (
        f"Application is initialized in the {mode} mode using {storage} storage. \n"
        f"Bot name is {telegram_settings.bot_name}. \n"
        f"Initial assistant prompt: {gpt_settings.assistant_prompt}. \n"
        f"Access to GPT-4 models is {gpt4_state}. \n"
        f"Proxy is {telegram_settings.proxy or 'UNSET'}. \n"
        f"Messages TTL: {gpt_settings.max_conversation_age_minutes} minutes. \n"
        f"Maximum conversation history size: {gpt_settings.max_history_tokens} tokens. \n"
        f"Maximum answer size: {gpt_settings.max_tokens} tokens. \n"
        f"GPT-4 access whitelist: {gpt_settings.gpt4_whitelist or 'UNSET'}. \n"
        f"Users whitelist: {telegram_settings.users_whitelist or 'UNSET'}. \n"
        f"Groups whitelist: {telegram_settings.groups_whitelist or 'UNSET'}. \n"
    )
    logger.info(message)
