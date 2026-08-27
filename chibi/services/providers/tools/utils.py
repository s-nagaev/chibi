import datetime
import json
import urllib.parse
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ParamSpec, TypedDict, TypeVar, Unpack

import httpx
from cachetools import TTLCache
from fake_useragent import UserAgent
from httpx import Response
from loguru import logger

from chibi.config import gpt_settings
from chibi.constants import get_sub_executor_prompt
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.interface import UserInterface
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database
from chibi.utils.app import SingletonMeta

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider

P = ParamSpec("P")
R = TypeVar("R")

ua_generator = UserAgent()


class AdditionalOptions(TypedDict, total=False):
    user_id: int | None
    caller_model: str
    caller_provider: str
    caller_storage_id: int | None
    caller_thread_id: int | None
    interface: UserInterface | None


def resolve_session_context(**kwargs: Unpack[AdditionalOptions]) -> tuple[int, int]:
    """Resolve the current session identity from a tool-invocation payload.

    Resolution order: ``interface.storage_id``/``interface.thread_id`` first,
    then the scalar ``caller_storage_id``/``caller_thread_id`` options injected
    by providers (used when no interface travels with the call, e.g. inside
    sub-agent requests).

    Args:
        kwargs: The additional-options payload of a tool invocation.

    Returns:
        A ``(storage_id, thread_id)`` pair identifying the originating session.

    Raises:
        ValueError: If neither an interface nor both caller identity options
            are available.
    """
    interface = kwargs.get("interface")
    if interface is not None:
        return interface.storage_id, interface.thread_id

    storage_id = kwargs.get("caller_storage_id")
    thread_id = kwargs.get("caller_thread_id")
    if storage_id is None or thread_id is None:
        raise ValueError(
            "No session context available: this function requires either an interface "
            "or caller_storage_id/caller_thread_id to be automatically provided."
        )
    return storage_id, thread_id


def _generate_google_search_referrer(target_url: str) -> str:
    """Generates a fake Google search referrer URL for a given target URL.

    This helps simulate traffic coming from a Google search result link,
    which can sometimes affect how websites serve content.

    Args:
        target_url: The URL that the fake referrer should point to.

    Returns:
        A string representing the generated Google referrer URL.
    """
    encoded_target_url = urllib.parse.quote(target_url, safe="")

    fake_ved = "2ahUKEwj_0sL5yPaFAxW_FRAIHeYxBpUQwgF6BAgGEAA"
    fake_opi = "89974493"

    referrer = (
        f"https://www.google.com/url?sa=t&rct=j&q={encoded_target_url}&esrc=s&source=web&"
        f"cd=1&cad=rja&uact=8&ved={fake_ved}&url={encoded_target_url}&opi={fake_opi}"
    )

    return referrer


async def _get_url(url: str) -> Response:
    """Fetch content from a given URL.

    It uses configured proxy, retries, and timeout settings from gpt_settings,
    and includes various headers including a generated Google referrer and a
    random User-Agent to mimic a real browser request.

    Args:
        url: The URL to fetch content from.

    Returns:
        An httpx.Response object containing the response from the URL.

    Raises:
        Httpx exceptions if the request fails (e.g., network errors).
    """
    transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
    headers: dict[str, str] = {
        "User-Agent": ua_generator.random,
        "Referer": _generate_google_search_referrer(target_url=url),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,"
            "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
    }
    async with httpx.AsyncClient(
        transport=transport, timeout=gpt_settings.timeout, proxy=gpt_settings.proxy, follow_redirects=True
    ) as client:
        return await client.get(url=url, headers=headers)


@inject_database
async def get_sub_agent_response(
    db: Database,
    user_id: int,
    prompt: str,
    model_name: str,
    provider_name: str,
    caller_storage_id: int | None = None,
    caller_thread_id: int | None = None,
) -> ChatResponseSchema:
    """Run a one-shot sub-agent request sharing the parent turn's session context.

    The parent's ``(storage_id, thread_id)`` travels both into the sub-agent's
    own LLM request (so its system prompt advertises the same effective working
    directory as the parent thread) and into every tool invocation the sub-agent
    issues (via ``caller_storage_id``/``caller_thread_id``), so tools like
    ``set_working_dir`` mutate the same thread-scoped slot. The parent's
    interface object is deliberately NOT propagated to avoid reentrancy.

    Args:
        db: The database instance.
        user_id: The storage ID of the user.
        prompt: The task prompt for the sub-agent.
        model_name: The LLM model the sub-agent should use.
        provider_name: The LLM provider the sub-agent should use.
        caller_storage_id: Parent session storage ID, or None when unknown.
        caller_thread_id: Parent session thread ID, or None when unknown.

    Returns:
        The sub-agent chat response.
    """
    user = await db.get_or_create_user(user_id=user_id)
    provider: Provider | None = user.providers.get(provider_name=provider_name)

    if not provider:
        raise ValueError(f"No provider with name '{provider_name}' found.")

    user_prompt = {
        "user_type": "llm",
        "current_working_dir": user.get_effective_working_dir(caller_thread_id),
        "prompt": prompt,
        "datetime_now": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    }

    user_message = Message(role="user", content=json.dumps(user_prompt))
    conversation_messages = [
        user_message,
    ]

    chat_response, _ = await provider.get_chat_response(
        messages=conversation_messages,
        user=user,
        model=model_name,
        system_prompt=get_sub_executor_prompt(gpt_settings.filesystem_access),
        caller_storage_id=caller_storage_id,
        caller_thread_id=caller_thread_id,
    )
    return chat_response


async def download(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=90.0)  # TODO: move timeout to settings or use one of existent
            response.raise_for_status()
            data = response.content
            logger.log("TOOL", f"Downloaded data from URL {url}: {len(data)} bytes")
            return data
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
    return None


class CallTracker(metaclass=SingletonMeta):
    def __init__(self, ttl: int = 60) -> None:
        self._calls: TTLCache = TTLCache(maxsize=3000, ttl=ttl)

    @staticmethod
    def _make_key(model_name: str, tool_name: str, serializable_kwargs: dict[str, Any]) -> str:
        kwargs_str = json.dumps(serializable_kwargs, sort_keys=True, default=str)
        return f"{tool_name}_{model_name}:{sha256(kwargs_str.encode()).hexdigest()[:16]}"

    def track(self, model_name: str, tool_name: str, serializable_kwargs: dict[str, Any]) -> int:
        key = self._make_key(model_name=model_name, tool_name=tool_name, serializable_kwargs=serializable_kwargs)

        if key not in self._calls:
            self._calls[key] = 1
            return 1

        self._calls[key] += 1
        return self._calls[key]


async def get_models_available_to_user(user_id: int, image_generation: bool = False) -> list[ModelChangeSchema]:
    from chibi.services.user import get_models_available

    data: list[ModelChangeSchema] = await get_models_available(user_id=user_id, image_generation=image_generation)
    return data
