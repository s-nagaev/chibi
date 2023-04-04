from functools import wraps
from typing import Any, Callable

from loguru import logger
from openai.error import InvalidRequestError, RateLimitError, TryAgain
from telegram import Chat as TelegramChat
from telegram import Update
from telegram import User as TelegramUser
from telegram import constants
from telegram.ext import ContextTypes

from chibi.config import telegram_settings

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]


async def send_message(update: Update, context: ContextTypes.DEFAULT_TYPE, reply: bool = True, **kwargs: Any) -> None:
    if reply:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, reply_to_message_id=update.message.message_id, **kwargs
        )
        return
    await context.bot.send_message(chat_id=update.effective_chat.id, **kwargs)


def user_is_allowed(tg_user: TelegramUser) -> bool:
    if not telegram_settings.users_whitelist:
        return True
    return any(identifier in telegram_settings.users_whitelist for identifier in (tg_user.id, tg_user.name))


def group_is_allowed(tg_chat: TelegramChat) -> bool:
    if not telegram_settings.groups_whitelist:
        return True
    return tg_chat.id in telegram_settings.groups_whitelist


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

    async def wrapper(*args, **kwargs) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        chat = update.effective_chat
        telegram_user = update.message.from_user
        user_name = telegram_user.name or f"{telegram_user.first_name} ({telegram_user.id})"

        if telegram_user.is_bot and not telegram_settings.allow_bots:
            logger.warning(f"Bots are not allowed. {user_name}'s request ignored.")
            return None

        if chat.type in PERSONAL_CHAT_TYPES and not user_is_allowed(tg_user=telegram_user):
            logger.warning(f"{user_name} is not allowed to work with me. Request rejected.")
            await send_message(update=update, context=context, text=telegram_settings.message_for_disallowed_users)
            return None

        if chat.type in GROUP_CHAT_TYPES and not group_is_allowed(tg_chat=chat):
            message = (
                f"The group {chat.effective_name} (id: {chat.id}, link: {chat.link}) does not exist in the whitelist. "
                "Leaving it..."
            )
            logger.warning(message)
            await context.bot.send_message(chat_id=chat.id, text=message, disable_web_page_preview=True)
            await chat.leave()
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
    async def wrapper(*args, **kwargs) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        user = update.message.from_user
        try:
            return await func(*args, **kwargs)
        except TryAgain as e:
            logger.error(f"{user.name} didn't get a GPT answer due to exception: {e}")
            await send_message(update=update, context=context, text="🥴Service is overloaded. Please, try again later.")
            return None

        except InvalidRequestError as e:
            logger.error(f"{user.name} got a InvalidRequestError: {e}")
            await send_message(update=update, context=context, text=f"😲{e}")
            return None

        except RateLimitError as e:
            logger.error(f"{user.name} reached a Rate Limit: {e}")
            await send_message(update=update, context=context, text=f"🤐Rate Limit exceeded: {e}")
            return None

        except Exception as e:
            logger.error(f"{user.name} got an error: {e}")
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
