import logging
from typing import Any, Callable

from telegram import Update, constants
from telegram.ext import ContextTypes

from chibi.config import telegram_settings

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]


async def send_message(update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs: Any) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id, reply_to_message_id=update.message.message_id, **kwargs
    )


def check_user_allowance(func: Callable[..., Any]) -> Callable[..., Any]:
    async def wrapper(*args, **kwargs) -> Any:
        update: Update = kwargs.get("update") or args[1]
        context: ContextTypes.DEFAULT_TYPE = kwargs.get("context") or args[2]
        chat = update.effective_chat
        user = update.message.from_user.username

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
        logging.warning(f"{update.message.from_user.username} is not allowed to work with me. Request rejected.")

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
