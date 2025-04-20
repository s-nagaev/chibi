import io
from collections import deque
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, Type, TypeVar, cast
from urllib.parse import parse_qs, urlparse

import httpx
import telegramify_markdown
from loguru import logger
from telegram import Chat as TelegramChat
from telegram import Message as TelegramMessage
from telegram import Update, constants
from telegram import User as TelegramUser
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.constants import (
    GROUP_CHAT_TYPES,
    MARKDOWN_TOKENS,
    PERSONAL_CHAT_TYPES,
    SETTING_SET,
    SETTING_UNSET,
    UserAction,
    UserContext,
)
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoProviderSelectedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.services.providers import registered_providers

R = TypeVar("R")
P = ParamSpec("P")

CLOSER = {"```": "```", "`": "`", "*": "*", "_": "_", "~": "~"}


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


def _get_next_token(text: str, pos: int, escaped: bool) -> tuple[str | None, int]:
    """Find the next Markdown token at the given position.

    Checks if the text starting from `pos` begins with any known Markdown token,
    unless the preceding character was an escape character '\'.

    Args:
        text: The string to search within.
        pos: The starting position in the text to check for a token.
        escaped: True if the character immediately preceding `pos` was an
                 escape character ('\'), False otherwise.

    Returns:
        A tuple containing:
        - The found Markdown token (str) or None if no token is found
          at the position or if it's escaped.
        - The length of the found token (int), or 0 if no token is found.
    """
    if escaped:
        return None, 0
    for token in MARKDOWN_TOKENS:
        if text.startswith(token, pos):
            return token, len(token)
    return None, 0


def split_markdown_v2(
    text: str,
    limit: int = constants.MessageLimit.MAX_TEXT_LENGTH,
    recommended_margin: int = 400,
    safety_margin: int = 50,
) -> list[str]:
    """Split a Markdown text into chunks while trying to preserve formatting.

    Attempts to split the text into chunks smaller than `limit`. It tracks
    opening and closing Markdown tokens using a stack. When a chunk needs to be
    split, it appends the necessary closing tokens to the end of the current
    chunk and prepends the corresponding opening tokens to the beginning of the
    next chunk.

    Splitting priority:
    1. At a newline character when the buffer size is close to the limit
       (within `recommended_margin`).
    2. Anywhere when the buffer size is very close to the limit
       (within `safety_margin`).

    Args:
        text: The Markdown string to split.
        limit: The maximum desired length for each chunk. Defaults to
               `constants.MessageLimit.MAX_TEXT_LENGTH`.
        recommended_margin: The preferred distance from the `limit` at which
               to split, ideally looking for a newline.
        safety_margin: The absolute minimum distance from the `limit` at which
               a split must occur, regardless of the character.

    Returns:
        A list of strings, where each string is a chunk of the original text,
        with formatting tokens adjusted to maintain validity across chunks.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    buffer: list[str] = []
    stack: deque[str] = deque()

    i = 0
    escaped: bool = False

    while i < len(text):
        token, shift = _get_next_token(text=text, pos=i, escaped=escaped)
        escaped = text[i] == "\\"
        if token:
            buffer.append(token)
            if stack and stack[-1] == token:
                stack.pop()
            else:
                stack.append(token)
            i += shift
        else:
            buffer.append(text[i])
            i += 1

        if len("".join(buffer)) >= limit - recommended_margin:
            if text[i] == "\n":
                closing_sequence = "\n".join(reversed(stack))
                chunks.append("".join(buffer) + closing_sequence)
                buffer = list(stack)

        elif len("".join(buffer)) >= limit - safety_margin:
            closing_sequence = "\n".join(reversed(stack))
            chunks.append("".join(buffer) + closing_sequence)
            buffer = list(stack)

    if buffer:
        chunks.append("".join(buffer))
    return chunks


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
    normalize_md: bool = True,
) -> None:
    if normalize_md:
        message = telegramify_markdown.markdownify(message)
        chunks = split_markdown_v2(message)
    else:
        chunks = [
            message[i : i + constants.MessageLimit.MAX_TEXT_LENGTH]
            for i in range(0, len(message), constants.MessageLimit.MAX_TEXT_LENGTH)
        ]

    for chunk_number, chunk in enumerate(chunks):
        await send_message(update=update, context=context, text=chunk, parse_mode=parse_mode, reply=chunk_number == 0)


async def send_message_in_plain_text_and_file(message: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_chat = get_telegram_chat(update=update)

    await send_long_message(message=message, update=update, context=context, normalize_md=False)
    file = io.BytesIO()
    file.write(message.encode())
    file.seek(0)
    explain_message_text = (
        "Oops! 😯It looks like your answer contains some code, but Telegram can't display it properly. "
        "I'll additionally add your answer to the markdown file. 👇"
    )

    await send_message(update=update, context=context, text=explain_message_text, reply=False)
    await context.bot.send_document(
        chat_id=telegram_chat.id,
        document=file,
        filename="answer.md",
    )


async def send_gpt_answer_message(gpt_answer: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await send_long_message(
            message=gpt_answer, update=update, context=context, parse_mode=constants.ParseMode.MARKDOWN_V2
        )
    except BadRequest as e:
        # Trying to handle an exception connected with markdown parsing: just re-sending the message in a text mode.
        logger.error(
            f"{user_data(update)} got a Telegram Bad Request error in the {chat_data(update)} "
            f"while receiving GPT answer: {e}. Trying to re-send it in plain text mode."
        )
        await send_message_in_plain_text_and_file(message=gpt_answer, update=update, context=context)


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


def check_user_allowance(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R | None]]:
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

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        update: Update = cast(Update, kwargs.get("update", None) or args[1])
        context: ContextTypes.DEFAULT_TYPE = cast(ContextTypes.DEFAULT_TYPE, kwargs.get("context") or args[2])
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
                update=update, context=context, text=f"Rate Limit exceeded for {e.provider}. We should back off a bit."
            )
            return None

        except NoModelSelectedError as e:
            logger.error(f"{error_msg_prefix}: {e}")

            await send_message(
                update=update,
                context=context,
                text="Please, select your model first.",
            )
            return None

        except NoProviderSelectedError as e:
            logger.error(f"{error_msg_prefix}: {e}")

            await send_message(
                update=update,
                context=context,
                text="Please, select your provider first.",
            )
            return None

        except Exception as e:
            logger.error(f"{error_msg_prefix}: {e!r}")
            msg = (
                "I'm sorry, but there seems to be a little hiccup with your request at the moment 😥 Would you mind "
                "trying again later? Don't worry, I'll be here to assist you whenever you're ready! 😼"
            )
            await send_message(update=update, context=context, text=msg)
            raise

    return wrapper


def get_user_context(context: ContextTypes.DEFAULT_TYPE, key: UserContext, expected_type: Type[R]) -> R | None:
    """Retrieve a specific value from the user's context data.

    Safely access the `user_data` dictionary associated with the current user
    in the Telegram bot context and return the value associated with the given key,
    cast to the expected type.

    Args:
        context: The update context provided by the `python-telegram-bot` library.
        key: An enum member (UserContext) representing the key for the data to retrieve.
        expected_type: The Python type the retrieved value is expected to conform to.
                       Used for casting the result.

    Returns:
        The value associated with the key, cast to `expected_type`, if it exists
        and `user_data` is available. Otherwise, returns None.
    """
    if context.user_data is not None:
        return cast(R, context.user_data.get(key, None))
    return None


def set_user_context(context: ContextTypes.DEFAULT_TYPE, key: UserContext, value: object | None) -> None:
    """Set or update a specific value in the user's context data.

    Safely access the `user_data` dictionary associated with the current user
    and store the provided value under the given key. If `user_data` does not
    exist, do nothing.

    Args:
        context: The update context provided by the `python-telegram-bot` library.
        key: An enum member (UserContext) representing the key under which to store the value.
        value: The value to store in the user's context data. Can be any object or None.
    """
    if context.user_data is not None:
        context.user_data[key] = value
    return None


def current_user_action(context: ContextTypes.DEFAULT_TYPE) -> UserAction:
    """Get the current action state associated with the user.

    Retrieve the action stored under the `UserContext.ACTION` key in the user's
    context data. If `user_data` is missing or the action is not set, it defaults
    to `UserAction.NONE`.

    Args:
        context: The update context provided by the `python-telegram-bot` library.

    Returns:
        The current `UserAction` enum member associated with the user. Defaults to
        `UserAction.NONE` if no action is set or `user_data` is unavailable.
    """
    if context.user_data is None:
        return UserAction.NONE
    return context.user_data.get(UserContext.ACTION, UserAction.NONE)


def set_user_action(context: ContextTypes.DEFAULT_TYPE, action: UserAction) -> None:
    """Set the current action state for the user.

    Store the provided `UserAction` under the `UserContext.ACTION` key in the user's
    context data. If `user_data` does not exist, do nothing.

    Args:
        context: The update context provided by the `python-telegram-bot` library.
        action: The `UserAction` enum member to set as the user's current action state.
    """
    if context.user_data is not None:
        context.user_data[UserContext.ACTION] = action
    return None


async def download_image(image_url: str) -> bytes:
    parsed_url = urlparse(image_url)
    params = parse_qs(parsed_url.query)
    image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    response = await httpx.AsyncClient().get(url=image_url, params=params)
    response.raise_for_status()
    return response.content


def _provider_statuses() -> list[str]:
    """Prepare a provider clients statuses data for logging.

    Returns:
        list of string containing the provider clients statuses data.
    """
    statuses = [
        "<magenta>Provider clients:</magenta>",
    ]
    for provider_name in registered_providers:
        status = SETTING_SET if getattr(gpt_settings, f"{provider_name.lower()}_key", None) else SETTING_UNSET
        statuses.append(f"{provider_name} client: {status}")
    return statuses


def log_application_settings() -> None:
    mode = "<yellow>PUBLIC</yellow>" if gpt_settings.public_mode else "<blue>PRIVATE</blue>"
    storage = "<red>REDIS</red>" if application_settings.redis else "<yellow>LOCAL</yellow>"
    proxy = f"<blue>{telegram_settings.proxy}</blue>" if telegram_settings.proxy else SETTING_UNSET
    users_whitelist = (
        f"<blue>{','.join(telegram_settings.users_whitelist)}</blue>"
        if telegram_settings.users_whitelist
        else SETTING_UNSET
    )
    groups_whitelist = (
        f"<blue>{telegram_settings.groups_whitelist}</blue>" if telegram_settings.groups_whitelist else SETTING_UNSET
    )
    models_whitelist = (
        f"<blue>{', '.join(gpt_settings.models_whitelist)}</blue>" if gpt_settings.models_whitelist else SETTING_UNSET
    )
    images_whitelist = (
        f"<blue>{','.join(gpt_settings.image_generations_whitelist)}</blue>"
        if gpt_settings.image_generations_whitelist
        else SETTING_UNSET
    )

    messages = [
        "<magenta>General Settings:</magenta>",
        f"Application is initialized in the {mode} mode using {storage} storage.",
        f"Proxy is {proxy}",
        "<magenta>LLM Settings:</magenta>",
        f"Bot name is <blue>{telegram_settings.bot_name}</blue>",
        f"Initial assistant prompt: <blue>{gpt_settings.assistant_prompt}</blue>",
        f"Messages TTL: <blue>{gpt_settings.max_conversation_age_minutes} minutes</blue>",
        f"Maximum conversation history size: <blue>{gpt_settings.max_history_tokens}</blue> tokens",
        f"Maximum answer size: <blue>{gpt_settings.max_tokens}</blue> tokens",
        f"Images generation limit: <blue>{gpt_settings.image_generations_monthly_limit}</blue>",
        "<magenta>Whitelists:</magenta>",
        f"Images limit whitelist: {images_whitelist}",
        f"Users whitelist: {users_whitelist}",
        f"Groups whitelist: {groups_whitelist}",
        f"Models whitelist: {models_whitelist}",
    ]
    messages += _provider_statuses()

    for message in messages:
        logger.opt(colors=True).info(message)

    if application_settings.redis_password:
        logger.opt(colors=True).warning(
            "`REDIS_PASSWORD` environment variable is <red>deprecated</red>. Use `REDIS` instead, i.e. "
            "`redis://:password@localhost:6379/0`"
        )
