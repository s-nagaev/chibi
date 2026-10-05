import time
from typing import Any, Unpack

import httpx
from ddgs import DDGS
from httpx import Response
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from trafilatura import extract

from chibi.config import gpt_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.jina import JINA_READER_BASE_URL, _map_request_error
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, _get_url

MIN_EXTRACT_CHARS = 200
BREAKER_FAILURE_THRESHOLD = 3
BREAKER_COOLDOWN_SECONDS = 300
JINA_PREAMBLE_MARKER = "Markdown Content:"
JS_GARBAGE_MARKERS: tuple[str, ...] = (
    "function(",
    "=>",
    "document.",
    "window.",
    "addeventlistener",
    "getelementbyid",
    "innerhtml",
    "console.log",
    "</script>",
    "json.parse",
)
JS_GARBAGE_MAX_MARKERS_PER_1000_CHARS = 5.0
MIN_ALPHANUMERIC_RATIO = 0.3

_clock = time.monotonic


class _CircuitBreaker:
    """In-memory circuit breaker guarding one extraction path.

    ``BREAKER_FAILURE_THRESHOLD`` consecutive failures open the circuit for
    ``BREAKER_COOLDOWN_SECONDS``; after the cooldown a single half-open trial
    request is allowed. A successful trial closes the circuit again, a failed
    one reopens it. State is process-local by design (no persistence) and is
    safe under the single asyncio loop: the state check and mutation happen
    without any intervening awaits.
    """

    def __init__(
        self,
        failure_threshold: int = BREAKER_FAILURE_THRESHOLD,
        cooldown_seconds: int = BREAKER_COOLDOWN_SECONDS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self.reset()

    def reset(self) -> None:
        """Return the breaker to the fully closed state."""
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_trial = False

    def allow_request(self) -> bool:
        """Decide whether a request may pass through the path right now.

        Returns:
            True when the circuit is closed or a free half-open trial slot is
            available; the trial slot is consumed atomically on True.
        """
        if self._opened_at is None:
            return True
        if _clock() - self._opened_at < self._cooldown_seconds:
            return False
        if self._half_open_trial:
            return False
        self._half_open_trial = True
        return True

    def record_success(self) -> None:
        """Close the circuit after a successful request."""
        self.reset()

    def record_failure(self) -> None:
        """Register a failure and open the circuit when the threshold is hit."""
        if self._half_open_trial:
            self._opened_at = _clock()
            self._half_open_trial = False
            return
        if self._opened_at is not None:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = _clock()


_trafilatura_breaker = _CircuitBreaker()
_jina_breaker = _CircuitBreaker()


def _strip_jina_preamble(text: str) -> str:
    """Strip the 'Title: / URL Source: / Markdown Content:' Jina preamble.

    Args:
        text: The raw Jina Reader response body.

    Returns:
        The body content without the preamble, or the original text when no
        preamble marker is found.
    """
    index = text.find(JINA_PREAMBLE_MARKER)
    if index == -1:
        return text
    return text[index + len(JINA_PREAMBLE_MARKER) :].lstrip()


def _looks_like_html(text: str) -> bool:
    """Detect that a text is actually an HTML document instead of extracted prose.

    Args:
        text: The text to inspect.

    Returns:
        True when the text starts with a well-known HTML document or block tag.
    """
    stripped = text.lstrip()[:200].lower()
    return stripped.startswith(("<!doctype html", "<html", "<head", "<body", "<div", "<script"))


def _looks_like_js_garbage(text: str) -> bool:
    """Heuristically detect JavaScript-heavy garbage instead of readable content.

    Two signals are used: the density of well-known JS tokens per 1000
    characters, and the ratio of alphanumeric/whitespace characters (minified
    code and encoded payloads are mostly punctuation).

    Args:
        text: The text to inspect.

    Returns:
        True when the text looks like script output rather than prose.
    """
    lowered = text.lower()
    marker_count = sum(lowered.count(marker) for marker in JS_GARBAGE_MARKERS)
    if marker_count * 1000 / len(text) > JS_GARBAGE_MAX_MARKERS_PER_1000_CHARS:
        return True
    readable = sum(ch.isalnum() or ch.isspace() for ch in text) / len(text)
    return readable < MIN_ALPHANUMERIC_RATIO


def _passes_quality_gate(text: str | None, raw_html: str | None = None) -> bool:
    """Decide whether fetched text is good enough to hand to the model.

    Args:
        text: The candidate content.
        raw_html: The raw page HTML the text was extracted from, when known.

    Returns:
        True when the text passes all quality checks: non-empty, at least
        ``MIN_EXTRACT_CHARS`` long, not a raw HTML fallback, not identical to
        the raw page and not JavaScript garbage.
    """
    if not text:
        return False
    if len(text) < MIN_EXTRACT_CHARS:
        return False
    if _looks_like_html(text=text):
        return False
    if raw_html is not None and text.strip() == raw_html.strip():
        return False
    if _looks_like_js_garbage(text=text):
        return False
    return True


async def _fetch_via_trafilatura(url: str) -> tuple[str, str]:
    """Fetch a page and extract its main content with trafilatura.

    Args:
        url: The URL to fetch.

    Returns:
        A ``(text, raw_html)`` tuple. When extraction fails, the raw HTML is
        returned as the text so the caller's quality gate can judge it.

    Raises:
        ToolException: If the request fails or the status code is not 200.
    """
    response: Response = await _get_url(url)
    if response.status_code != 200:
        raise ToolException(f"Failed to get URL: {url}. Status code: {response.status_code}")
    raw_html = response.text
    if not raw_html:
        raise ToolException(f"Failed to extract data from URL: {url}. Empty response received.")
    content = extract(filecontent=raw_html, include_links=True)
    return (content if content else raw_html), raw_html


async def _fetch_via_jina(url: str) -> tuple[str, str | None]:
    """Fetch a page as plain text through the Jina Reader API.

    Works without an API key on the free 20 RPM tier; with ``JINA_API_KEY``
    the request is authenticated for the 500 RPM tier.

    Args:
        url: The URL to fetch.

    Returns:
        A ``(text, raw_html)`` tuple with the preamble-stripped page text and
        a ``None`` raw HTML (Jina does not return the source document).

    Raises:
        ToolException: If the request fails or the response is empty.
    """
    headers: dict[str, str] = {"X-Return-Format": "text"}
    if gpt_settings.jina_api_key:
        headers["Authorization"] = f"Bearer {gpt_settings.jina_api_key}"
    transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=gpt_settings.timeout,
            proxy=gpt_settings.proxy,
        ) as client:
            response = await client.get(url=f"{JINA_READER_BASE_URL}/{url}", headers=headers)
            response.raise_for_status()
    except Exception as e:
        raise _map_request_error(context=f"Reading URL '{url}' via Jina Reader", error=e)
    text = _strip_jina_preamble(text=response.text)
    if not text:
        raise ToolException(f"Jina Reader returned an empty response for URL: {url}.")
    return text, None


class SearchNewsTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="search_news",
            description="Searches for current news articles based on the given search query at duckduckgo.com",
            parameters={
                "type": "object",
                "properties": {
                    "search_phrase": {
                        "type": "string",
                        "description": "The text of the search query for news searching.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of news articles to return (default is 10).",
                    },
                },
                "required": ["search_phrase"],
            },
        ),
    )
    name = "search_news"

    @classmethod
    async def function(
        cls, search_phrase: str, max_results: int = 10, **kwargs: Unpack[AdditionalOptions]
    ) -> dict[str, Any]:
        """Search for news articles using DuckDuckGo News.

        Args:
            search_phrase: The keywords or phrase to search for in news.
            max_results: The maximum number of news results to return (default is 10).

        Returns:
            A JSON formatted string containing the list of news articles found,
            or an error message string if the search fails.
        """
        caller_model = kwargs.get("caller_model", "unknown model")
        logger.log(
            "TOOL",
            f"[{caller_model}] Searching news for '{search_phrase}', max_results={max_results}",
        )
        try:
            result = DDGS(proxy=gpt_settings.proxy).news(query=search_phrase, max_results=max_results, region="wt-wt")
        except Exception as e:
            raise ToolException(f"Couldn't find news for '{search_phrase}', max_results={max_results}. Error: {e}")
        return {
            "news": result,
        }


class DDGSWebSearchTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="ddgs_web_search",
            description=(
                "Search for information on the internet using the DDGS python library. "
                "Use this function if other web search functions are unavailable or not working."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_phrase": {
                        "type": "string",
                        "description": "The text of the search query for web searching.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of web search results to return (default is 10).",
                    },
                },
                "required": ["search_phrase"],
            },
        ),
    )
    name = "ddgs_web_search"

    @classmethod
    async def function(
        cls, search_phrase: str, max_results: int = 10, **kwargs: Unpack[AdditionalOptions]
    ) -> dict[str, Any]:
        """Perform a general web search using DDGS python library.

        Args:
            search_phrase: The keywords or phrase to search for on the web.
            max_results: The maximum number of search results to return (default is 10).

        Returns:
            A JSON formatted string containing the list of search results found,
            or an error message string if the search fails.
        """
        logger.log(
            "TOOL",
            (
                f"[{kwargs.get('caller_model', 'unknown model')}] Using web-search for '{search_phrase}', "
                f"max_results={max_results}"
            ),
        )
        try:
            result = DDGS(proxy=gpt_settings.proxy).text(query=search_phrase, max_results=max_results, region="wt-wt")
        except Exception as e:
            raise ToolException(
                f"Couldn't get search result for '{search_phrase}', max_results={max_results}. Error: {e}"
            )

        return {
            "search_results": result,
        }


class GoogleSearchTool(ChibiTool):
    register = gpt_settings.google_search_client_set
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="google_web_search",
            description=("Search for information on the internet via Google Web Search API."),
            parameters={
                "type": "object",
                "properties": {
                    "search_phrase": {
                        "type": "string",
                        "description": "The text of the search query for web searching.",
                    },
                },
                "required": ["search_phrase"],
            },
        ),
    )
    name = "google_web_search"

    @classmethod
    async def function(cls, search_phrase: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """Perform a general web search using Google Web Search.

        TODO: upgrade to using `max_results` arg.

        Args:
            search_phrase: The keywords or phrase to search for on the web.

        Returns:
            A JSON formatted string containing the list of search results found,
            or an error message string if the search fails.
        """
        logger.log(
            "TOOL", f"[{kwargs.get('caller_model', 'unknown model')}] Using Google web-search for '{search_phrase}'"
        )
        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)
        params = {
            "key": gpt_settings.google_search_api_key,
            "cx": gpt_settings.google_search_cx,
            "q": search_phrase,
        }
        url = "https://www.googleapis.com/customsearch/v1"
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=gpt_settings.timeout,
                proxy=gpt_settings.proxy,
            ) as client:
                response = await client.get(
                    url=url,
                    params=params,
                )
                response.raise_for_status()
        except Exception as e:
            raise ToolException(f"An error occurred while calling the Google Search API: {e}")

        data = response.json()
        items = data.get("items")
        if not items:
            logger.warning(f"{cls.name} tool returned an empty list of results. Search phrase: {search_phrase}.")
            return {"search_results": "Ooops, the search returned an empty list of results."}

        target_keys = ["title", "link", "snippet"]
        search_results = [{key: item.get(key) for key in target_keys} for item in items]
        return {
            "search_results": search_results,
        }


class ReadWebPageTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="read_web_page",
            description=(
                "Read the content of the web page. Uses a hybrid pipeline: Jina Reader and "
                "trafilatura with automatic fallback and quality checks, so JS-heavy or anti-bot "
                "pages are still retrievable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Web page URL to fetch."},
                },
                "required": ["url"],
            },
        ),
    )
    name = "read_web_page"

    @classmethod
    async def function(cls, url: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """Fetch and extract the main content from a given web page URL.

        Routing depends on ``JINA_API_KEY``: with a key the Jina Reader path
        (500 RPM tier) is tried first, without a key trafilatura goes first and
        the free r.jina.ai tier (20 RPM) acts as the fallback. Each path is
        guarded by a circuit breaker and its result passes a quality gate; when
        both paths fail the gate, the better raw result is returned anyway.

        Args:
            url: The URL of the web page to read.
            kwargs: Additional tool-invocation options injected by the provider.

        Returns:
            A dict with the extracted text under the ``content`` key, or with
            the best raw result under ``data`` plus a ``warning`` when no path
            produced clean text.

        Raises:
            ToolException: If every available extraction path failed.
        """
        caller_model = kwargs.get("caller_model", "unknown model")
        logger.log("TOOL", f"[{caller_model}] Reading URL: {url}")

        primary, secondary = ("jina", "trafilatura") if gpt_settings.jina_api_key else ("trafilatura", "jina")
        fetchers: dict[str, Any] = {"jina": _fetch_via_jina, "trafilatura": _fetch_via_trafilatura}
        breakers: dict[str, _CircuitBreaker] = {
            "jina": _jina_breaker,
            "trafilatura": _trafilatura_breaker,
        }

        candidates: list[tuple[str, str]] = []
        for path_name in (primary, secondary):
            breaker = breakers[path_name]
            if not breaker.allow_request():
                logger.debug(f"[{caller_model}] read_web_page: '{path_name}' circuit is open, skipping for {url}")
                continue
            try:
                text, raw_html = await fetchers[path_name](url)
            except Exception as e:
                breaker.record_failure()
                logger.warning(f"[{caller_model}] read_web_page: '{path_name}' path failed for {url}: {e}")
                continue

            if _passes_quality_gate(text=text, raw_html=raw_html):
                breaker.record_success()
                logger.log(
                    "TOOL",
                    f"[{caller_model}] The data from the URL {url} extracted via '{path_name}' passed the quality gate",
                )
                return {
                    "content": text,
                }

            breaker.record_failure()
            candidates.append((path_name, text))
            logger.warning(f"[{caller_model}] read_web_page: '{path_name}' result for {url} failed the quality gate")

        if candidates:
            _, best_text = max(candidates, key=lambda item: len(item[1]))
            msg = f"Failed to extract clean content from URL: {url}. Sending the best raw result to the model"
            logger.warning(f"[{caller_model}] {msg}")
            return {
                "data": best_text,
                "warning": msg,
            }

        raise ToolException(f"Couldn't read URL: {url}. All available extraction paths failed.")
