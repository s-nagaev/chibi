import urllib.parse
from typing import ParamSpec, TypedDict, TypeVar

import httpx
from fake_useragent import UserAgent
from httpx import Response
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings

P = ParamSpec("P")
R = TypeVar("R")

ua_generator = UserAgent()


class AdditionalOptions(TypedDict, total=False):
    user_id: int | None
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
