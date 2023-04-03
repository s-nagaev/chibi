import logging
from functools import wraps
from typing import Any, Callable

from openai.error import InvalidRequestError, RateLimitError, TryAgain
from telegram import Update, constants
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


def check_user_allowance(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator controlling access to the chatbot.

    This deco checks:
        - if the specific user is allowed to interact with the chatbot, using USERS_WHITELIST.
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
        user = update.message.from_user.username

        if not telegram_settings.allow_bots and user.endswith("Bot"):
            logging.warning(f"Bots are not allowed. {user}'s request ignored.")
            return

        if chat.type in PERSONAL_CHAT_TYPES:
            if not telegram_settings.users_whitelist or (user in telegram_settings.users_whitelist):
                return await func(*args, **kwargs)

        if chat.type in GROUP_CHAT_TYPES:
            if not telegram_settings.groups_whitelist or (chat.id in telegram_settings.groups_whitelist):
                return await func(*args, **kwargs)

            message = (
                f"The group {chat.effective_name} (id: {chat.id}, link: {chat.link}) does not exist in the whitelist. "
                "Leaving it..."
            )
            logging.warning(message)
            await context.bot.send_message(chat_id=chat.id, text=message, disable_web_page_preview=True)
            await chat.leave()

        await context.bot.send_message(
            chat_id=chat.id,
            text=telegram_settings.message_for_disallowed_users,
            disable_web_page_preview=True,
        )
        logging.warning(f"{user} is not allowed to work with me. Request rejected.")

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
            logging.error(f"{user.name} didn't get a GPT answer due to exception: {e}")
            await send_message(update=update, context=context, text="🥴Service is overloaded. Please, try again later.")
            return
        except InvalidRequestError as e:
            logging.error(f"{user.name} got a InvalidRequestError: {e}")
            await send_message(update=update, context=context, text=f"😲{e}")
            return
        except RateLimitError as e:
            logging.error(f"{user.name} reached a Rate Limit: {e}")
            await send_message(update=update, context=context, text=f"🤐Rate Limit exceeded: {e}")
            return
        except Exception as e:
            logging.error(f"{user.name} got an error: {e}")
            msg = (
                "I'm sorry, but there seems to be a little hiccup with your request at the moment 😥 Would you mind "
                "trying again later? Don't worry, I'll be here to assist you whenever you're ready! 😼"
            )
            await send_message(update=update, context=context, text=msg)
            return

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
