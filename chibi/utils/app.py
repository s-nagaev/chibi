from functools import wraps
from typing import Any, Callable

import httpx
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import application_settings, gpt_settings, telegram_settings
from chibi.constants import SETTING_DISABLED, SETTING_ENABLED, SETTING_SET, SETTING_UNSET
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoProviderSelectedError,
    NoResponseError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.utils.telegram import chat_data, send_message, user_data


class SingletonMeta(type):
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
        context: The callback context provided by the JobQueue. Although
                 required by the JobQueue signature, it's not directly used
                 in this function's logic.
    """
    if not application_settings.heartbeat_url:
        return None

    transport = httpx.AsyncHTTPTransport(
        retries=application_settings.heartbeat_retry_calls,
        proxy=application_settings.heartbeat_proxy,
    )

    async with httpx.AsyncClient(transport=transport, proxy=application_settings.heartbeat_proxy) as client:
        try:
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
        "<magenta>Provider clients:</magenta>",
    ]
    for provider_name in RegisteredProviders.all.keys():
        status = SETTING_SET if provider_name in RegisteredProviders.available else SETTING_UNSET
        statuses.append(f"{provider_name} client: {status}")
    return statuses


def log_application_settings() -> None:
    mode = "<yellow>PUBLIC</yellow>" if gpt_settings.public_mode else "<cyan>PRIVATE</cyan>"
    storage = "<red>REDIS</red>" if application_settings.redis else "<yellow>LOCAL</yellow>"
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
    images_whitelist = (
        f"<cyan>{','.join(gpt_settings.image_generations_whitelist)}</cyan>"
        if gpt_settings.image_generations_whitelist
        else SETTING_UNSET
    )

    messages = [
        "<magenta>General Settings:</magenta>",
        f"Application is initialized in the {mode} mode using {storage} storage.",
        f"Proxy is {proxy}",
        "<magenta>LLM Settings:</magenta>",
        f"Bot name is <cyan>{telegram_settings.bot_name}</cyan>",
        # f"Initial assistant prompt: <cyan>{gpt_settings.assistant_prompt}</cyan>",
        f"Messages TTL: <cyan>{gpt_settings.max_conversation_age_minutes} minutes</cyan>",
        f"Maximum conversation history size: <cyan>{gpt_settings.max_history_tokens}</cyan> tokens",
        f"Maximum answer size: <cyan>{gpt_settings.max_tokens}</cyan> tokens",
        f"Images generation limit: <cyan>{gpt_settings.image_generations_monthly_limit}</cyan>",
        f"Filesystem access: {SETTING_ENABLED if gpt_settings.filesystem_access else SETTING_DISABLED}",
        "<magenta>Whitelists:</magenta>",
        f"Images limit whitelist: {images_whitelist}",
        f"Users whitelist: {users_whitelist}",
        f"Groups whitelist: {groups_whitelist}",
        f"Models whitelist: {models_whitelist}",
        "<magenta>Heartbeat:</magenta>",
        f"Heartbeat mechanism: {SETTING_SET if application_settings.heartbeat_url else SETTING_UNSET}",
    ]
    messages += _provider_statuses()

    for message in messages:
        logger.opt(colors=True).info(message)

    if application_settings.redis_password:
        logger.opt(colors=True).warning(
            "`REDIS_PASSWORD` environment variable is <red>deprecated</red>. Use `REDIS` instead, i.e. "
            "`redis://:password@localhost:6379/0`"
        )


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

        except NoResponseError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            return None

        except ServiceRateLimitError as e:
            logger.error(f"{error_msg_prefix}: {e}")
            await send_message(
                update=update,
                context=context,
                text=f"Rate Limit exceeded for {e.provider}. We should back off a bit.",
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
            logger.exception(f"{error_msg_prefix}: {e!r}")
            msg = (
                "I'm sorry, but there seems to be a little hiccup with your request at the moment 😥 Would you mind "
                "trying again later? Don't worry, I'll be here to assist you whenever you're ready! 😼"
            )
            await send_message(update=update, context=context, text=msg)
            # raise

    return wrapper
