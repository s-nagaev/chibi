import datetime
import json
import urllib.parse
from typing import TYPE_CHECKING, ParamSpec, TypedDict, TypeVar

import httpx
from fake_useragent import UserAgent
from httpx import Response
from loguru import logger
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.constants import SUB_EXECUTOR_PROMPT
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider

P = ParamSpec("P")
R = TypeVar("R")

ua_generator = UserAgent()


class AdditionalOptions(TypedDict, total=False):
    user_id: int | None
    model: str | None
    telegram_context: ContextTypes.DEFAULT_TYPE | None
    telegram_update: Update | None


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
    async with httpx.AsyncClient(transport=transport, timeout=gpt_settings.timeout, proxy=gpt_settings.proxy) as client:
        return await client.get(url=url, headers=headers)


@inject_database
async def get_sub_agent_response(
    db: Database,
    user_id: int,
    prompt: str,
    model_name: str | None = None,
    provider_name: str | None = None,
) -> ChatResponseSchema:
    user = await db.get_or_create_user(user_id=user_id)
    provider: Provider | None
    if not model_name or not provider_name:
        provider = user.active_gpt_provider
        model = user.selected_gpt_model_name
    else:
        provider = user.providers.get(provider_name=provider_name)
        model = model_name

    if not provider:
        raise ValueError(f"No provider found. Provided provider name: {provider_name}")

    user_prompt = {
        "user_type": "llm",
        "current_working_dir": user.working_dir,
        "prompt": prompt,
        "datetime_now": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
    }

    user_message = Message(role="user", content=json.dumps(user_prompt))
    conversation_messages = [
        user_message,
    ]

    chat_response, _ = await provider.get_chat_response(
        messages=conversation_messages, user=user, model=model, system_prompt=SUB_EXECUTOR_PROMPT
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
