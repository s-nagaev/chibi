import json
from typing import Any, Unpack
from urllib.parse import urlparse

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions

JINA_READER_BASE_URL = "https://r.jina.ai"
JINA_SEARCH_BASE_URL = "https://s.jina.ai"
JINA_SEARCH_RESULT_TRUNCATION_LIMIT = 2000
JINA_ALLOWED_RETURN_FORMATS = ("text", "markdown", "html", "screenshot", "pageshot")


def _truncate_result_content(text: str, limit: int = JINA_SEARCH_RESULT_TRUNCATION_LIMIT) -> str:
    """Truncate a search-result body to the configured limit.

    Args:
        text: The raw result content.
        limit: Maximum number of characters to keep (default 2000).

    Returns:
        The original text when it fits, otherwise a truncated copy with an
        ellipsis marker appended.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated)"


def _parse_search_results(raw: str) -> list[dict[str, Any]]:
    """Parse a Jina Search API response body into compact result dicts.

    The API returns a JSON array (or an object with a ``data`` array) of
    results with title, url and full content. Every result keeps its title
    and URL untouched, while the content body is truncated to
    ``JINA_SEARCH_RESULT_TRUNCATION_LIMIT`` characters. A non-JSON body is
    degraded into a single truncated result instead of failing.

    Args:
        raw: The raw response body text.

    Returns:
        A list of dicts with ``title``, ``url`` and truncated ``content``.
    """
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        if raw.strip():
            return [{"title": "", "url": "", "content": _truncate_result_content(text=raw)}]
        return []

    if isinstance(data, dict):
        items = data.get("data", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []

    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("description") or ""
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": _truncate_result_content(text=str(content)),
            }
        )
    return results


def _map_request_error(context: str, error: Exception) -> ToolException:
    """Convert an httpx request failure into a ToolException with guidance.

    Args:
        context: Short human-readable description of the failed Jina call.
        error: The original exception raised by the HTTP stack.

    Returns:
        A ToolException whose message is actionable for the calling model.
    """
    if isinstance(error, httpx.TimeoutException):
        return ToolException(
            f"{context} timed out. The Jina API did not respond in time; retry later or lower the timeout."
        )
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 401:
        return ToolException(
            f"{context} failed with 401 Unauthorized. This Jina API endpoint requires an API key: "
            "set JINA_API_KEY in the environment and restart. Meanwhile, use ddgs_web_search as a fallback."
        )
    if status_code == 429:
        return ToolException(
            f"{context} failed with 429 Too Many Requests. Jina API rate limits: r.jina.ai 20 RPM without "
            "a key / 500 RPM with a key; s.jina.ai 100 RPM with a key. Wait a bit and retry."
        )
    return ToolException(f"{context} failed: {error}")


class JinaReadTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="jina_read",
            description=(
                "Read the content of a web page via the Jina Reader API. Handles JS-heavy and "
                "anti-bot pages that plain extraction may fail on, with optional CSS targeting, "
                "link summaries and screenshot capture."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Web page URL to fetch. Must be an absolute http(s) URL.",
                    },
                    "return_format": {
                        "type": "string",
                        "enum": list(JINA_ALLOWED_RETURN_FORMATS),
                        "description": (
                            "Desired output format: text, markdown, html, screenshot or pageshot (default: text)."
                        ),
                    },
                    "target_selector": {
                        "type": "string",
                        "description": "Optional CSS selector to extract only a part of the page.",
                    },
                    "with_links_summary": {
                        "type": "boolean",
                        "description": (
                            "Append a summary of all page links at the end of the response (default: false)."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 60).",
                    },
                },
                "required": ["url"],
            },
        ),
    )
    name = "jina_read"

    @classmethod
    async def function(
        cls,
        url: str,
        return_format: str = "text",
        target_selector: str | None = None,
        with_links_summary: bool = False,
        timeout: int = 60,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        """Fetch a page through the Jina Reader API.

        Args:
            url: Absolute http(s) URL of the page to read.
            return_format: One of text, markdown, html, screenshot, pageshot.
            target_selector: Optional CSS selector narrowing the extracted region.
            with_links_summary: Whether to append a link summary to the response.
            timeout: Request timeout in seconds.
            kwargs: Additional tool-invocation options injected by the provider.

        Returns:
            A dict with the reader output under the ``content`` key.

        Raises:
            ToolException: If the URL scheme is not http(s), the return format
                is unsupported, the request fails, or the response is empty.
        """
        caller_model = kwargs.get("caller_model", "unknown model")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ToolException(f"Invalid URL '{url}': jina_read requires an absolute http(s) URL.")

        if return_format not in JINA_ALLOWED_RETURN_FORMATS:
            allowed = ", ".join(JINA_ALLOWED_RETURN_FORMATS)
            raise ToolException(f"Unsupported return_format '{return_format}'. Allowed values: {allowed}.")

        logger.log("TOOL", f"[{caller_model}] Reading URL via Jina Reader: {url} (format={return_format})")

        headers: dict[str, str] = {"X-Return-Format": return_format}
        if target_selector:
            headers["X-Target-Selector"] = target_selector
        if with_links_summary:
            headers["X-With-Links-Summary"] = "true"
        if gpt_settings.jina_api_key:
            headers["Authorization"] = f"Bearer {gpt_settings.jina_api_key}"

        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                proxy=gpt_settings.proxy,
            ) as client:
                response = await client.get(url=f"{JINA_READER_BASE_URL}/{url}", headers=headers)
                response.raise_for_status()
        except Exception as e:
            raise _map_request_error(context=f"Reading URL '{url}' via Jina Reader", error=e)

        content = response.text
        if not content:
            raise ToolException(f"Jina Reader returned an empty response for URL: {url}.")

        return {
            "content": content,
        }


class JinaSearchTool(ChibiTool):
    register = bool(gpt_settings.jina_api_key)
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="jina_search",
            description=(
                "Search the web via the Jina Search API and return top results with titles, URLs "
                "and truncated content. Requires JINA_API_KEY."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_phrase": {
                        "type": "string",
                        "description": "The text of the search query.",
                    },
                },
                "required": ["search_phrase"],
            },
        ),
    )
    name = "jina_search"

    @classmethod
    async def function(cls, search_phrase: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """Search the web through the Jina Search API.

        Args:
            search_phrase: The keywords or phrase to search for.
            kwargs: Additional tool-invocation options injected by the provider.

        Returns:
            A dict with a list of truncated results (title, url, content) under
            the ``search_results`` key, or an explanatory message when nothing
            was found.

        Raises:
            ToolException: If no API key is configured, or the request fails.
        """
        caller_model = kwargs.get("caller_model", "unknown model")
        if not gpt_settings.jina_api_key:
            raise ToolException(
                "jina_search requires JINA_API_KEY. Set it in the environment and restart, "
                "or use ddgs_web_search as a keyless fallback."
            )

        logger.log("TOOL", f"[{caller_model}] Using Jina web-search for '{search_phrase}'")

        headers = {
            "Authorization": f"Bearer {gpt_settings.jina_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=gpt_settings.timeout,
                proxy=gpt_settings.proxy,
            ) as client:
                response = await client.post(url=JINA_SEARCH_BASE_URL, headers=headers, json={"q": search_phrase})
                response.raise_for_status()
        except Exception as e:
            raise _map_request_error(context=f"Searching '{search_phrase}' via Jina Search", error=e)

        search_results = _parse_search_results(raw=response.text)
        if not search_results:
            logger.warning(f"{cls.name} tool returned an empty list of results. Search phrase: {search_phrase}.")
            return {"search_results": "Ooops, the search returned an empty list of results."}

        return {
            "search_results": search_results,
        }
