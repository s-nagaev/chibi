import io
from functools import wraps
from typing import Any, Callable, Type
from urllib.parse import parse_qs, urlparse

import httpx
from loguru import logger
from telegram import Chat as TelegramChat
from telegram import Message as TelegramMessage
from telegram import Update
from telegram import User as TelegramUser
from telegram import constants
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.constants import GROUP_CHAT_TYPES, PERSONAL_CHAT_TYPES, SupportedProviders
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.mistralai import MistralAI
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import Provider


def get_telegram_user(update: Update) -> TelegramUser:
    if user := update.effective_user:
        return user
    raise ValueError(f"Telegram incoming update does not contain valid user data. Update ID: {update.update_id}")


def get_telegram_chat(update: Update) -> TelegramChat:
    if chat := update.effective_chat:
        return chat
    raise ValueError(f"Telegram incoming update does not contain valid chat data. Update ID: {update.update_id}")


def user_data(update: Update) -> str:
    user = get_telegram_user(update=update)
    return f"{user.name} ({user.id})"


def chat_data(update: Update) -> str:
    chat = get_telegram_chat(update=update)
    return f"{chat.type.upper()} chat ({chat.id})"


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


async def send_long_message(
    message: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    parse_mode: str | None = None,
) -> None:
    chunk_size = constants.MessageLimit.MAX_TEXT_LENGTH
    text_chunks = [message[i : i + chunk_size] for i in range(0, len(message), chunk_size)]
    for chunk in text_chunks:
        await send_message(update=update, context=context, text=chunk, parse_mode=parse_mode)


async def send_gpt_answer_message(gpt_answer: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_chat = get_telegram_chat(update=update)
    try:
        await send_long_message(
            message=gpt_answer, update=update, context=context, parse_mode=constants.ParseMode.MARKDOWN
        )
    except BadRequest as e:
        # Trying to handle an exception connected with markdown parsing: just re-sending the message in a text mode.
        logger.error(
            f"{user_data(update)} got a Telegram Bad Request error in the {chat_data(update)} "
            f"while receiving GPT answer: {e}. Trying to re-send it in plain text mode."
        )
        await send_long_message(message=gpt_answer, update=update, context=context)

        if "```" in gpt_answer:
            logger.info(
                f"{user_data(update)} got and answer containing some code in the {chat_data(update)}, "
                f"but face with a markdown parse error. Additionally sending answer as an attachment..."
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
    return any(identifier in telegram_settings.users_whitelist for identifier in (str(tg_user.id), tg_user.username))


def group_is_allowed(tg_chat: TelegramChat) -> bool:
    if not telegram_settings.groups_whitelist:
        return True
    return tg_chat.id in telegram_settings.groups_whitelist


def user_interacts_with_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    telegram_message = get_telegram_message(update=update)
    prompt = telegram_message.text

    if not prompt:
        return False

    if context.bot.first_name in prompt or context.bot.username in prompt:
        return True

    reply_message = telegram_message.reply_to_message
    if not reply_message or not reply_message.from_user:
        return False

    return reply_message.from_user.id == context.bot.id


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

        if telegram_user.is_bot and not telegram_settings.allow_bots:
            logger.warning(f"Bots are not allowed. Request from {user_data(update)} was ignored.")
            return None

        if telegram_chat.type in PERSONAL_CHAT_TYPES and not user_is_allowed(tg_user=telegram_user):
            logger.warning(f"{user_data(update)} is not allowed to work with me. Request rejected.")
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
        error_msg_prefix = f"{user_data(update)} didn't get a GPT answer in the {chat_data(update)}"
        try:
            return await func(*args, **kwargs)
        except NoApiKeyProvidedError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            await send_message(
                update=update,
                context=context,
                text="Oops! It looks like you didn't set the API key for this provider.",
            )
            return None

        except NotAuthorizedError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            await send_message(
                update=update,
                context=context,
                text=(
                    "We encountered an authorization problem when interacting with a remote service.\n"
                    f"Please check your {e.provider} API key."
                ),
            )
            return None

        except ServiceResponseError as e:
            logger.error(f"{error_msg_prefix}: {e}")

            await send_message(
                update=update,
                context=context,
                text=(
                    f"😲Lol... we got an unexpected response from the {e.provider} service! \n"
                    f"Please, try again a bit later."
                ),
            )
            return None

        except ServiceRateLimitError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            await send_message(
                update=update, context=context, text=f"🤐Rate Limit exceeded for {e.provider}. We should back off a bit."
            )
            return None

        except Exception as e:
            logger.error(f"{error_msg_prefix}: {e!r}")
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
    if not api_key:
        return False
    if " " in api_key:
        return False
    if len(api_key) < 30:
        return False
    if len(api_key) > 150:
        return False

    return True


async def download_image(image_url: str) -> bytes:
    parsed_url = urlparse(image_url)
    params = parse_qs(parsed_url.query)
    image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    response = await httpx.AsyncClient().get(url=image_url, params=params)
    response.raise_for_status()
    return response.content


def get_provider_class_by_name(provider_name: SupportedProviders) -> Type[Provider]:
    match provider_name:
        case SupportedProviders.ANTHROPIC:
            return Anthropic
        case SupportedProviders.MISTRALAI:
            return MistralAI
        case SupportedProviders.OPENAI:
            return OpenAI
    raise ValueError("Wrong provider name provided.")


def log_application_settings() -> None:
    mode = "<yellow>PUBLIC</yellow>" if gpt_settings.public_mode else "<blue>PRIVATE</blue>"
    is_set = "<green>SET</green>"
    unset = "<red>UNSET</red>"
    storage = "<red>REDIS</red>" if application_settings.redis else "<yellow>LOCAL</yellow>"
    proxy = f"<blue>{telegram_settings.proxy}</blue>" if telegram_settings.proxy else unset
    users_whitelist = (
        f"<blue>{','.join(telegram_settings.users_whitelist)}</blue>" if telegram_settings.users_whitelist else unset
    )
    groups_whitelist = (
        f"<blue>{telegram_settings.groups_whitelist}</blue>" if telegram_settings.groups_whitelist else unset
    )
    images_whitelist = (
        f"<blue>{','.join(gpt_settings.image_generations_whitelist)}</blue>"
        if gpt_settings.image_generations_whitelist
        else unset
    )

    messages = (
        f"Application is initialized in the {mode} mode using {storage} storage.",
        f"Anthropic (Claude-3) client set: {is_set if bool(gpt_settings.anthropic_key) else unset }",
        f"Mistral AI client: {is_set if bool(gpt_settings.mistralai_key) else unset }",
        f"OpenAI client set: {is_set if bool(gpt_settings.openai_key) else unset }",
        f"Bot name is <blue>{telegram_settings.bot_name}</blue>",
        f"Initial assistant prompt: <blue>{gpt_settings.assistant_prompt}</blue>",
        f"Proxy is {proxy}",
        f"Messages TTL: <blue>{gpt_settings.max_conversation_age_minutes} minutes</blue>",
        f"Maximum conversation history size: <blue>{gpt_settings.max_history_tokens}</blue> tokens",
        f"Maximum answer size: <blue>{gpt_settings.max_tokens}</blue> tokens",
        f"Users whitelist: {users_whitelist}",
        f"Groups whitelist: {groups_whitelist}",
        f"Images generation limit: <blue>{gpt_settings.image_generations_monthly_limit}</blue>",
        f"Images limit whitelist: {images_whitelist}",
    )
    for message in messages:
        logger.opt(colors=True).info(message)

    if application_settings.redis_password:
        logger.opt(colors=True).warning(
            "`REDIS_PASSWORD` environment variable is <red>deprecated</red>. Use `REDIS` instead, i.e. "
            "`redis://:password@localhost:6379/0`"
        )
