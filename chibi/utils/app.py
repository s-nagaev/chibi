import json
from abc import ABCMeta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import httpx
from loguru import logger
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.constants import SETTING_DISABLED, SETTING_ENABLED, SETTING_SET, SETTING_UNSET
from chibi.exceptions import (
    ContextLengthExceededError,
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoProviderSelectedError,
    NoResponseError,
    NotAuthorizedError,
    RecursionLimitExceeded,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.interface import UserInterface


@runtime_checkable
class IDEErrorInterface(Protocol):
    """Optional error-state contract used by the IDE interface."""

    error_code: str | None
    error_message: str | None


def _set_ide_error(interface: UserInterface, code: str, message: str) -> None:
    """Record a sanitized error only for interfaces that opt into IDE state.

    Args:
        interface: User interface receiving the fallback response.
        code: Machine-readable IDE error code.
        message: Sanitized user-facing IDE error message.
    """
    if isinstance(interface, IDEErrorInterface):
        interface.error_code = code
        interface.error_message = message


class SingletonMeta(ABCMeta):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


async def run_heartbeat(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a heartbeat GET request to a configured monitoring URL.

    This function is designed to be called periodically by a scheduler
    (python-telegram-bot's JobQueue) to signal the bot's operational status
    to an external monitoring service (e.g., Healthchecks.io, Uptime Kuma, etc.).

    Args:
        context: The callback context provided by the JobQueue.
    """
    if not application_settings.heartbeat_url:
        return None

    transport = httpx.AsyncHTTPTransport(
        retries=application_settings.heartbeat_retry_calls,
        proxy=application_settings.heartbeat_proxy,
    )

    async with httpx.AsyncClient(transport=transport, proxy=application_settings.heartbeat_proxy) as client:
        try:
            await context.bot.get_me()

            result = await client.get(application_settings.heartbeat_url)
        except Exception as error:
            logger.error(f"Uptime Checker failed with an Exception: {error}")
            return
        if result.is_error:
            logger.error(f"Uptime Checker failed, status_code: {result.status_code}, msg: {result.text}")


def _provider_statuses() -> list[str]:
    """Prepare a provider clients statuses data for logging.

    Returns:
        list of string containing the provider clients statuses data.
    """
    from chibi.services.providers import RegisteredProviders

    statuses = [
        "<magenta> Provider clients </magenta>".center(90, "="),
    ]
    for provider_name in RegisteredProviders.all.keys():
        status = SETTING_SET if provider_name in RegisteredProviders.available else SETTING_UNSET
        statuses.append(f"{provider_name.capitalize()} client: {status}")
    statuses.append("=" * 71)
    return statuses


def log_application_settings() -> None:
    mode = "<yellow>PUBLIC</yellow>" if gpt_settings.public_mode else "<cyan>PRIVATE</cyan>"
    storage = "<red>REDIS</red>" if application_settings.redis else "<yellow>LOCAL</yellow>"
    chromadb_storage = "<cyan>REMOTE</cyan>" if application_settings.chroma_host else "<yellow>LOCAL</yellow>"
    proxy = f"<cyan>{telegram_settings.proxy}</cyan>" if telegram_settings.proxy else SETTING_UNSET
    users_whitelist = (
        f"<cyan>{','.join(telegram_settings.users_whitelist)}</cyan>"
        if telegram_settings.users_whitelist
        else SETTING_UNSET
    )
    groups_whitelist = (
        f"<cyan>{telegram_settings.groups_whitelist}</cyan>" if telegram_settings.groups_whitelist else SETTING_UNSET
    )
    models_whitelist = (
        f"<cyan>{', '.join(gpt_settings.models_whitelist)}</cyan>" if gpt_settings.models_whitelist else SETTING_UNSET
    )
    models_blacklist = (
        f"<cyan>{', '.join(gpt_settings.models_blacklist)}</cyan>" if gpt_settings.models_blacklist else SETTING_UNSET
    )
    images_whitelist = (
        f"<cyan>{','.join(gpt_settings.image_generations_whitelist)}</cyan>"
        if gpt_settings.image_generations_whitelist
        else SETTING_UNSET
    )
    embedding_provider = (
        application_settings.embedding_function.upper() if application_settings.is_chroma_configured else SETTING_UNSET
    )
    embedding_model = (
        application_settings.embedding_model
        if application_settings.is_chroma_configured and application_settings.embedding_model
        else SETTING_UNSET
    )

    messages = [
        "<magenta> General Settings </magenta>".center(90, "="),
        f"Application is initialized in the {mode} mode using {storage} storage.",
        f"Proxy is {proxy}",
        "<magenta> LLM Settings </magenta>".center(90, "="),
        f"Bot name is <cyan>{telegram_settings.bot_name}</cyan>",
        f"Messages TTL: <cyan>{gpt_settings.max_conversation_age_minutes} minutes</cyan>",
        f"Maximum conversation history size: <cyan>{gpt_settings.max_history_tokens}</cyan> tokens",
        f"Maximum answer size: <cyan>{gpt_settings.max_tokens}</cyan> tokens",
        f"Images generation limit: <cyan>{gpt_settings.image_generations_monthly_limit}</cyan>",
        f"Filesystem access: {SETTING_ENABLED if gpt_settings.filesystem_access else SETTING_DISABLED}",
        "<magenta> Whitelists </magenta>".center(90, "="),
        f"Images limit whitelist: {images_whitelist}",
        f"Users whitelist: {users_whitelist}",
        f"Groups whitelist: {groups_whitelist}",
        f"Models whitelist: {models_whitelist}",
        f"Models blacklist: {models_blacklist}",
        "<magenta> Heartbeat: </magenta>".center(90, "="),
        f"Heartbeat mechanism: {SETTING_SET if application_settings.heartbeat_url else SETTING_UNSET}",
        "<magenta> Infinite Context </magenta>".center(90, "="),
        f"ChromaDB configuration: {SETTING_SET if application_settings.is_chroma_configured else SETTING_UNSET}",
        f"ChromaDB storage: {chromadb_storage}",
        f"Embedding provider: {embedding_provider}",
        f"Embedding model: {embedding_model}",
    ]
    messages += _provider_statuses()

    if gpt_settings.models_whitelist and gpt_settings.models_blacklist:
        logger.opt(colors=True).warning(
            "Both models_whitelist and models_blacklist are set. Blacklist will be ignored; using whitelist instead."
        )

    for message in messages:
        logger.opt(colors=True).info(message)

    if application_settings.redis_password:
        logger.opt(colors=True).warning(
            "`REDIS_PASSWORD` environment variable is <red>deprecated</red>. Use `REDIS` instead, i.e. "
            "`redis://:password@localhost:6379/0`"
        )


async def _try_reactive_context_recovery(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    storage_id: int,
    thread_id: int,
) -> Any:
    """Summarize history and retry the wrapped function once after context overflow.

    Args:
        func: The wrapped function to retry.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        storage_id: User/chat storage identifier to summarize.
        thread_id: Thread identifier to summarize.

    Returns:
        The result of the retried function call.

    Raises:
        ContextLengthExceededError: If the retry also overflows.
        Exception: Any exception raised by emergency_summarization or the retry.
    """
    from chibi.services.user import emergency_summarization  # Circular import avoidance

    await emergency_summarization(storage_id=storage_id, thread_id=thread_id)
    return await func(*args, **kwargs)


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
        interface: UserInterface | None = kwargs.get("interface")
        if not interface:
            logger.warning(
                f"The 'handle_gpt_exceptions' decorator couldn't get the interface object "
                f"from function {func.__name__}."
            )
            return await func(*args, **kwargs)
        error_msg_prefix = f"{interface.user_data} didn't get a GPT answer in the {interface.chat_data}"
        text = (
            "I'm sorry, but there seems to be a little hiccup with your request at the moment 😥 Would you mind "
            "trying again later? Don't worry, I'll be here to assist you whenever you're ready! 😼"
        )

        try:
            return await func(*args, **kwargs)

        except NoResponseError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface,
                "provider_error",
                "The provider returned no response. Try again or check provider availability.",
            )
            return None

        except NoApiKeyProvidedError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface, "provider_configuration", "This provider is not configured. Set its API key, then try again."
            )
            text = "Oops! It looks like you didn't set the API key for this provider."

        except NotAuthorizedError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface,
                "provider_authorization",
                f"The {e.provider} provider rejected its credentials. Check the provider API key.",
            )
            text = (
                "We encountered an authorization problem when interacting with a remote service.\n"
                f"Please check your {e.provider} API key."
            )

        except ServiceResponseError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface,
                "provider_error",
                f"The {e.provider} provider returned an unexpected response. Try again later.",
            )
            text = (
                f"😲Lol... we got an unexpected response from the {e.provider} service! \n"
                f"Please, try again a bit later."
            )

        except ContextLengthExceededError as e:
            if gpt_settings.reactive_context_recovery:
                try:
                    result = await _try_reactive_context_recovery(
                        func=func,
                        args=args,
                        kwargs=kwargs,
                        storage_id=interface.storage_id,
                        thread_id=interface.thread_id,
                    )
                    if isinstance(result, ChatResponseSchema):
                        result = result.model_copy(
                            update={"answer": result.answer + "\n\n_(Context was compressed to stay within limits.)_"}
                        )
                    return result
                except ContextLengthExceededError:
                    logger.warning(f"{error_msg_prefix}: context still too large after reactive summarization")
                except Exception as recovery_error:
                    logger.exception(f"{error_msg_prefix}: reactive context recovery failed: {recovery_error!r}")
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface,
                "context_length_exceeded",
                "The conversation context is too large for the model. Start a new thread or reset history.",
            )
            text = (
                "I'm sorry, but this conversation has grown too large for the model's context window. "
                "Please start a new thread or reset the chat history with /reset."
            )

        except ServiceRateLimitError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(
                interface, "provider_rate_limit", f"The {e.provider} provider rate limit was reached. Try again later."
            )
            text = f"Rate Limit exceeded for {e.provider}. We should back off a bit."

        except NoModelSelectedError as e:
            logger.error(f"{error_msg_prefix}: {e}")

            _set_ide_error(interface, "missing_model", "Select a model before sending a request.")
            text = "Please, select your model first."

        except NoProviderSelectedError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            _set_ide_error(interface, "missing_provider", "Select a provider before sending a request.")
            text = "Please, select your provider first."

        except RecursionLimitExceeded as e:
            logger.error(f"{error_msg_prefix}: {e}")
            text = (
                f"{e.provider} ({e.model}) exceeded the limit on the maximum number of consecutive tool calls "
                f"({e.exceeded_limit}) and was stopped. The model has likely entered an infinite loop of tool "
                f"calls. Please check the logs. If the model was functioning as intended, you should either "
                f"rephrase the task or increase the value of the `MAX_CONSECUTIVE_TOOL_CALLS` setting."
            )

        except Exception as e:
            logger.exception(f"{error_msg_prefix}: {e!r}")
            _set_ide_error(
                interface,
                "runtime_error",
                "Chibi could not complete the provider operation. Check the output channel for details.",
            )

        await interface.send_message(message=text)

    return wrapper


def get_builtin_skill_names() -> dict[str, str]:
    path = Path(application_settings.skills_dir)
    result = {}
    for f in path.iterdir():
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            with f.open(encoding="utf-8") as fh:
                first_line = fh.readline()
            desc = first_line.lstrip("# ").strip() if first_line.startswith("#") else f.stem
            result[f.name] = desc
        except (UnicodeDecodeError, OSError):
            continue
    return result


def convert_list_of_models_to_str(models: list[ModelChangeSchema]) -> str:
    available_llm_models = {}

    for llm in models:
        if llm.provider not in available_llm_models:
            available_llm_models[llm.provider] = llm.name
        else:
            available_llm_models[llm.provider] += f", {llm.name}"
    return json.dumps(available_llm_models)
