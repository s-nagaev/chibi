from duckduckgo_search import DDGS
from httpx import Response
from loguru import logger
from trafilatura import extract

from chibi.config import gpt_settings
from chibi.services.providers.tools.schemas import ToolResponse
from chibi.services.providers.tools.utils import _get_url

duckduckgo = DDGS(proxy=gpt_settings.proxy)


async def search_news(search_phrase: str, max_results: int = 10) -> str:
    """Search for news articles using DuckDuckGo News.

    Args:
        search_phrase: The keywords or phrase to search for in news.
        max_results: The maximum number of news results to return (default is 10).

    Returns:
        A JSON formatted string containing the list of news articles found,
        or an error message string if the search fails.
    """
    logger.log("TOOL", f"Searching news for '{search_phrase}', max_results={max_results}")
    try:
        result = duckduckgo.news(keywords=search_phrase, max_results=max_results)
    except Exception as e:
        msg = f"Couldn't find news for '{search_phrase}', max_results={max_results}. Error: {e}"
        logger.warning(msg)
        response = ToolResponse(
            tool_name="search_news",
            status="error",
            result=msg,
        )
        return response.model_dump_json()
    response = ToolResponse(
        tool_name="search_news",
        status="ok",
        result=result,
    )
    return response.model_dump_json()


async def web_search(search_phrase: str, max_results: int = 10) -> str:
    """Perform a general web search using DuckDuckGo Text Search.

    Args:
        search_phrase: The keywords or phrase to search for on the web.
        max_results: The maximum number of search results to return (default is 10).

    Returns:
        A JSON formatted string containing the list of search results found,
        or an error message string if the search fails.
    """
    logger.log("TOOL", f"Using web-search for '{search_phrase}', max_results={max_results}")
    try:
        result = duckduckgo.text(keywords=search_phrase, max_results=max_results, backend="lite")
    except Exception as e:
        msg = f"Couldn't get search result for '{search_phrase}', max_results={max_results}.Error: {e}"
        logger.warning(msg)
        response = ToolResponse(
            tool_name="web_search",
            status="error",
            result=msg,
        )
        return response.model_dump_json()

    response = ToolResponse(
        tool_name="web_search",
        status="ok",
        result=result,
    )
    return response.model_dump_json()


async def read_web_page(url: str) -> str:
    """Fetch and extract the main content from a given web page URL.

    Args:
        url: The URL of the web page to read.

    Returns:
        A string containing the extracted main text of the page, the raw HTML
        content if extraction fails, or an error message string if fetching
        fails or status code is not 200.
    """
    logger.log("TOOL", f"Reading URL: {url}")
    try:
        response: Response = await _get_url(url)
    except Exception as e:
        msg = f"Couldn't read URL: {url}. Error: {e}"
        logger.warning(msg)
        return ToolResponse(
            tool_name="read_web_page",
            status="error",
            result=msg,
        ).model_dump_json()

    if response.status_code != 200:
        msg = f"Failed to get URL: {url}. Status code: {response.status_code}"
        logger.warning(msg)
        return ToolResponse(
            tool_name="read_web_page",
            status="error",
            result=msg,
        ).model_dump_json()

    data = response.text
    if not data:
        msg = f"Failed to extract data from URL: {url}. Empty response received."
        logger.warning(msg)
        return ToolResponse(
            tool_name="read_web_page",
            status="error",
            result=msg,
        ).model_dump_json()

    content = extract(filecontent=data, include_links=True)
    if not content:
        msg = f"Failed to extract URL: {url}. Empty extracted data. Trying to send raw HTML to model"
        logger.warning(msg)
        return ToolResponse(
            tool_name="read_web_page",
            status="warning",
            result=data,
            additional_details=msg,
        ).model_dump_json()

    logger.log("TOOL", f"The data from the URL {url} seems to be successfully extracted")

    return ToolResponse(
        tool_name="read_web_page",
        status="ok",
        result=content,
    ).model_dump_json()
