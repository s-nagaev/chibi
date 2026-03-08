import sys
from collections import deque
from io import BytesIO
from typing import Any, Callable, Coroutine, Literal, ParamSpec, Type, TypeVar, cast
from urllib.parse import parse_qs, urlparse

import click
import httpx
import telegramify_markdown
from loguru import logger
from telegram import (
    Chat as TelegramChat,
)
from telegram import (
    InputMediaDocument,
    InputMediaPhoto,
    Update,
    constants,
)
from telegram import (
    Message as TelegramMessage,
)
from telegram import (
    User as TelegramUser,
)
from telegram.constants import FileSizeLimit
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from chibi.config import gpt_settings, telegram_settings
from chibi.constants import (
    FILE_UPLOAD_TIMEOUT,
    GROUP_CHAT_TYPES,
    IMAGE_UPLOAD_TIMEOUT,
    MARKDOWN_TOKENS,
    PERSONAL_CHAT_TYPES,
    UserAction,
    UserContext,
)

R = TypeVar("R")
P = ParamSpec("P")


def get_telegram_user(update: Update) -> TelegramUser:
    """Retrieve the Telegram user from the update.

    Args:
        update: The incoming Telegram update.

    Returns:
        The Telegram user associated with the update.

    Raises:
        ValueError: If the update does not contain valid user data.
    """
    if user := update.effective_user:
        return user
    raise ValueError(f"Telegram incoming update does not contain valid user data. Update ID: {update.update_id}")


def get_telegram_chat(update: Update) -> TelegramChat:
    """Retrieve the Telegram chat from the update.

    Args:
        update: The incoming Telegram update.

    Returns:
        The Telegram chat associated with the update.

    Raises:
        ValueError: If the update does not contain valid chat data.
    """
    if chat := update.effective_chat:
        return chat
    raise ValueError(f"Telegram incoming update does not contain valid chat data. Update ID: {update.update_id}")


def user_data(update: Update) -> str:
    """Get a string representation of the user for logging.

    Args:
        update: The incoming Telegram update.

    Returns:
        A string containing the user's name and ID.
    """
    user = get_telegram_user(update=update)
    return f"{user.name} ({user.id})"


def chat_data(update: Update) -> str:
    """Get a string representation of the chat for logging.

    Args:
        update: The incoming Telegram update.

    Returns:
        A string containing the chat type and ID.
    """
    chat = get_telegram_chat(update=update)
    return f"{chat.type.upper()} chat ({chat.id})"


def get_telegram_message(update: Update) -> TelegramMessage:
    """Retrieve the Telegram message from the update.

    Args:
        update: The incoming Telegram update.

    Returns:
        The Telegram message associated with the update.

    Raises:
        ValueError: If the update does not contain valid message data.
    """
    if message := update.effective_message:
        return message
    raise ValueError(f"Telegram incoming update does not contain valid message data. Update ID: {update.update_id}")


def _get_next_token(text: str, pos: int, escaped: bool) -> tuple[str | None, int]:
    """Find the next Markdown token at the given position.

    Args:
        text: The string to search within.
        pos: The starting position in the text.
        escaped: Whether the character at pos is escaped.

    Returns:
        A tuple with the found token and its length.
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
    """Split a Markdown text into chunks.

    Args:
        text: The Markdown string to split.
        limit: The maximum desired length for each chunk.
        recommended_margin: The preferred distance from the limit at which to split.
        safety_margin: The absolute minimum distance from the limit at which a split must occur.

    Returns:
        A list of string chunks.
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
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    reply: bool = True,
    **kwargs: Any,
) -> TelegramMessage:
    """Send a message via Telegram.

    Args:
        update: The incoming Telegram update.
        context: The update context.
        reply: Whether to reply to the message.
        **kwargs: Additional arguments for send_message.

    Returns:
        The sent Telegram message.
    """
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)

    if reply:
        return await context.bot.send_message(
            chat_id=telegram_chat.id,
            reply_to_message_id=telegram_message.message_id,
            **kwargs,
        )
    return await context.bot.send_message(chat_id=telegram_chat.id, **kwargs)


async def send_long_message(
    message: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    parse_mode: str | None = None,
    normalize_md: bool = True,
    reply: bool = True,
) -> None:
    """Send a long message, splitting it if necessary.

    Args:
        message: The message text.
        update: The incoming Telegram update.
        context: The update context.
        parse_mode: The parse mode for the message.
        normalize_md: Whether to normalize Markdown.
        reply: Whether to reply to the message.
    """
    if normalize_md:
        message = telegramify_markdown.markdownify(message)
        chunks = split_markdown_v2(message)
    else:
        chunks = [
            message[i : i + constants.MessageLimit.MAX_TEXT_LENGTH]
            for i in range(0, len(message), constants.MessageLimit.MAX_TEXT_LENGTH)
        ]

    for chunk_number, chunk in enumerate(chunks):
        await send_message(
            update=update,
            context=context,
            text=chunk,
            parse_mode=parse_mode,
            reply=chunk_number == 0 if reply else False,
        )


async def send_audio(
    audio: bytes,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Send audio to the user.

    Args:
        audio: The audio data.
        update: The incoming Telegram update.
        context: The update context.
    """
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.RECORD_VOICE)

    await context.bot.send_audio(
        chat_id=telegram_chat.id, audio=audio, title="voice", reply_to_message_id=telegram_message.message_id
    )


async def download_image(image_url: str) -> bytes:
    """Download an image from a URL.

    Args:
        image_url: The URL of the image.

    Returns:
        The image data as bytes.
    """
    parsed_url = urlparse(image_url)
    params = parse_qs(parsed_url.query)
    image_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    response = await httpx.AsyncClient().get(url=image_url, params=params)
    response.raise_for_status()
    return response.content


async def send_images(
    images: list[str] | list[BytesIO],
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Send a list of images to the user.

    Args:
        images: A list of image URLs or BytesIO objects.
        update: The incoming Telegram update.
        context: The update context.
    """
    telegram_chat = get_telegram_chat(update=update)
    telegram_message = get_telegram_message(update=update)
    await context.bot.send_chat_action(chat_id=telegram_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)

    if isinstance(images[0], str):
        logger.info(f"Downloading {len(images)} images for {user_data(update)} via URLs...")
        image_files = [await download_image(image_url=cast(str, url)) for url in images]
        try:
            logger.info(f"Uploading {len(images)} images to {user_data(update)} in the {chat_data(update)}")
            await context.bot.send_media_group(
                chat_id=telegram_chat.id,
                media=[InputMediaPhoto(url) for url in image_files],
                reply_to_message_id=telegram_message.message_id,
                read_timeout=IMAGE_UPLOAD_TIMEOUT,
                write_timeout=IMAGE_UPLOAD_TIMEOUT,
            )
        except Exception as e:
            logger.error(
                f"{user_data(update)} image generation request succeeded, but we couldn't send the image "
                f"due to exception: {e}. Trying to send it via text message..."
            )
            image_urls = cast(list[str], images)
            await send_message(
                update=update,
                context=context,
                text="\n".join(image_urls),
                disable_web_page_preview=False,
            )
        return None

    logger.info(f"Uploading {len(images)} image(s) to {user_data(update)} in the {chat_data(update)}")
    media_photos: list[BytesIO] = []
    media_docs: list[BytesIO] = []

    for file in images:
        if not isinstance(file, BytesIO):
            continue

        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size < FileSizeLimit.PHOTOSIZE_UPLOAD:
            media_photos.append(file)
        elif size < FileSizeLimit.FILESIZE_UPLOAD:
            media_docs.append(file)
        else:
            logger.error(f"{user_data(update)} File size ({size}) exceeds file size limit, skipping it..")
            continue

    if media_photos:
        await context.bot.send_media_group(
            chat_id=telegram_chat.id,
            media=[InputMediaPhoto(img) for img in media_photos],
            reply_to_message_id=telegram_message.message_id,
            write_timeout=IMAGE_UPLOAD_TIMEOUT,
        )

    if media_docs:
        logger.info(f"Uploading {len(images)} image(s) as file(s) to {user_data(update)} in the {chat_data(update)}")

        await context.bot.send_media_group(
            chat_id=telegram_chat.id,
            media=[InputMediaDocument(media=img, filename="file.jpeg") for img in media_docs],
            reply_to_message_id=telegram_message.message_id,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )


async def send_text_file(file_content: str, file_name: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a text file to the user.

    Args:
        file_content: The content of the file.
        file_name: The name of the file.
        update: The incoming Telegram update.
        context: The update context.
    """
    telegram_chat = get_telegram_chat(update=update)
    text_file = BytesIO(file_content.encode("utf-8"))
    text_file.name = file_name

    await context.bot.send_document(
        chat_id=telegram_chat.id,
        document=text_file,
        filename=file_name,
    )


async def send_message_in_plain_text_and_file(
    message: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    reply: bool = True,
) -> None:
    """Send a message as plain text and as a file.

    Args:
        message: The message text.
        update: The incoming Telegram update.
        context: The update context.
        reply: Whether to reply to the message.
    """
    telegram_chat = get_telegram_chat(update=update)

    await send_long_message(message=message, update=update, context=context, normalize_md=False, reply=reply)
    file = BytesIO()
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


async def send_answer_message(
    message: str, update: Update, context: ContextTypes.DEFAULT_TYPE, reply: bool = True
) -> None:
    """Send an answer message, handling potential Markdown errors.

    Args:
        message: The message text.
        update: The incoming Telegram update.
        context: The update context.
        reply: Whether to reply to the message.
    """
    try:
        await send_long_message(
            message=message,
            update=update,
            context=context,
            parse_mode=constants.ParseMode.MARKDOWN_V2,
            reply=reply,
        )
    except BadRequest as e:
        logger.error(
            f"{user_data(update)} got a Telegram Bad Request error in the {chat_data(update)} "
            f"while receiving GPT answer: {e}. Trying to re-send it in plain text mode."
        )
        await send_message_in_plain_text_and_file(message=message, update=update, context=context, reply=reply)


def current_user_action(context: ContextTypes.DEFAULT_TYPE) -> UserAction:
    """Get the current action state associated with the user.

    Args:
        context: The update context.

    Returns:
        The current UserAction.
    """
    if context.user_data is None:
        return UserAction.NONE
    return context.user_data.get(UserContext.ACTION, UserAction.NONE)


def set_user_action(context: ContextTypes.DEFAULT_TYPE, action: UserAction) -> None:
    """Set the current action state for the user.

    Args:
        context: The update context.
        action: The UserAction to set.
    """
    if context.user_data is not None:
        context.user_data[UserContext.ACTION] = action
    return None


def user_interacts_with_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if the user is interacting with the bot.

    Args:
        update: The incoming Telegram update.
        context: The update context.

    Returns:
        True if the user is interacting with the bot, False otherwise.
    """
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


def get_user_context(context: ContextTypes.DEFAULT_TYPE, key: UserContext, expected_type: Type[R]) -> R | None:
    """Retrieve a specific value from the user's context data.

    Args:
        context: The update context.
        key: The key for the data to retrieve.
        expected_type: The expected type of the value.

    Returns:
        The value associated with the key, or None.
    """
    if context.user_data is not None:
        return cast(R, context.user_data.get(key, None))
    return None


def set_user_context(context: ContextTypes.DEFAULT_TYPE, key: UserContext, value: object | None) -> None:
    """Set or update a specific value in the user's context data.

    Args:
        context: The update context.
        key: The key under which to store the value.
        value: The value to store.
    """
    if context.user_data is not None:
        context.user_data[key] = value
    return None


def user_is_allowed(tg_user: TelegramUser) -> bool:
    """Check if the user is allowed to interact with the bot.

    Args:
        tg_user: The Telegram user.

    Returns:
        True if the user is allowed, False otherwise.
    """
    if not telegram_settings.users_whitelist:
        return True
    return any(identifier in telegram_settings.users_whitelist for identifier in (str(tg_user.id), tg_user.username))


def group_is_allowed(tg_chat: TelegramChat) -> bool:
    """Check if the group is allowed.

    Args:
        tg_chat: The Telegram chat.

    Returns:
        True if the group is allowed, False otherwise.
    """
    return tg_chat.id in telegram_settings.groups_whitelist


def check_user_allowance(
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R | None]]:
    """Decorator controlling access to the chatbot.

    Args:
        func: The async function to decorate.

    Returns:
        The wrapper function.
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
            await send_message(
                update=update,
                context=context,
                text=telegram_settings.message_for_disallowed_users,
            )
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


def show_message(header: str, message: str, message_type: Literal["err", "warn", "inf"]) -> None:
    """Show a formatted message to the user.

    Args:
        header: The message header.
        message: The message text.
        message_type: The type of the message ("err", "warn", "inf").
    """
    if message_type == "err":
        text_color = "red"
    elif message_type == "warn":
        text_color = "yellow"
    else:
        text_color = "green"

    click.echo()
    click.secho(f" {header.upper()} ".center(80, "="), fg=text_color, bold=True)
    click.echo(message)
    click.echo()
    click.echo("If you're using Chibi installed via pip, please update settings using")
    click.secho("$ chibi config", fg="green", bold=True)
    click.echo()
    click.echo(
        "Otherwise, please check the config file manually  or ensure "
        "that you've exported\nenvironment variables properly.",
    )
    click.secho("=" * 80, fg=text_color, bold=True)
    click.echo()


def _var(env_name: str) -> str:
    """Format an environment variable name.

    Args:
        env_name: The environment variable name.

    Returns:
        The formatted environment variable name.
    """
    return click.style(env_name, fg="yellow", bold=True)


def telegram_security_pre_start_check() -> None:
    """Perform security checks before starting the bot."""
    security_error_header = "SECURITY CHECK FAILURE"
    security_warning_header = "SECURITY CHECK WARNING"

    security_error: bool = False
    msg = ""

    # Private mode requires users whitelist
    if not gpt_settings.public_mode and not telegram_settings.users_whitelist:
        security_error = True
        msg = (
            f"Chibi is running in PRIVATE mode, but the {_var('USERS_WHITELIST')} setting\nis not configured.  "
            "This is EXTREMELY dangerous, as it allows ANY Telegram user\nto use YOUR bot with your API tokens.  "
            f"Please specify your Telegram username or\nID in {_var('USERS_WHITELIST')} by running the command:"
        )

    # Public mode is not compatible with the file system access
    if gpt_settings.filesystem_access and gpt_settings.public_mode:
        security_error = True
        msg = (
            "Chibi is running in PUBLIC mode with access to the computer’s file system!\nThis is an  EXTREMELY  "
            "dangerous combination of settings, allowing ANY Telegram\nuser to interact with data on your computer "
            "via the Agent.\n\nYou must either disable public mode "
            f"({_var('PUBLIC_MODE=false')}) or disable the Agent's\naccess to the file system "
            f"({_var('FILESYSTEM_ACCESS=false')})."
        )

    if security_error:
        show_message(header=security_error_header, message=msg, message_type="err")
        sys.exit(1)

    # Having an Agent with access to the file system in a Telegram group can be dangerous
    if gpt_settings.filesystem_access and telegram_settings.groups_whitelist:
        msg = (
            "Having an Agent with access to the file system in a Telegram group can be\ndangerous. "
            "We hope you know what you are doing!\nThe settings involved: "
            f"{_var('FILESYSTEM_ACCESS')}, {_var('GROUPS_WHITELIST')}."
        )
        show_message(header=security_warning_header, message=msg, message_type="warn")


def telegram_setting_pre_start_check() -> None:
    """Perform configuration checks before starting the bot."""
    if not telegram_settings.token:
        header = "CONFIGURATION ERROR"
        msg = f"Telegram token not set. Setting name: {_var('TELEGRAM_BOT_TOKEN')}."
        show_message(header=header, message=msg, message_type="err")
        sys.exit(1)
