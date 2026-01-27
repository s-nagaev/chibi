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
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions, _get_url


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
        logger.log(
            "TOOL",
            f"[{kwargs.get('model', 'Unknown model')}] Searching news for '{search_phrase}', max_results={max_results}",
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
                f"[{kwargs.get('model', 'Unknown model')}] Using web-search for '{search_phrase}', "
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
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Using Google web-search for '{search_phrase}'")
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
                "Read the content of the web page. Be prepared that trafilatura may not cope and "
                "will not be able to retrieve information either due to captcha or because of js."
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

        Args:
            url: The URL of the web page to read.

        Returns:
            A string containing the extracted main text of the page, the raw HTML
            content if extraction fails, or an error message string if fetching
            fails or status code is not 200.
        """
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Reading URL: {url}")
        try:
            response: Response = await _get_url(url)
        except Exception as e:
            raise ToolException(f"Couldn't read URL: {url}. Error: {e}")

        if response.status_code != 200:
            raise ToolException(f"Failed to get URL: {url}. Status code: {response.status_code}")

        data = response.text
        if not data:
            raise ToolException(f"Failed to extract data from URL: {url}. Empty response received.")

        content = extract(filecontent=data, include_links=True)
        if not content:
            msg = f"Failed to extract URL: {url}. Empty extracted data. Trying to send raw HTML to model"
            logger.warning(f"[{kwargs.get('model', 'Unknown model')}] {msg}")
            return {
                "data": data,
                "warning": msg,
            }

        logger.log(
            "TOOL",
            f"[{kwargs.get('model', 'Unknown model')}] The data from the URL {url} seems to be successfully extracted",
        )

        return {
            "content": content,
        }
