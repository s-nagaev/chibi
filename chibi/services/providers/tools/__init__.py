from typing import Any, Awaitable, Callable

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.services.providers.tools.current_date import get_current_datetime
from chibi.services.providers.tools.web_search import (
    read_web_page,
    search_news,
    web_search,
)

logger.level("TOOL", no=20, color="<blue>")


tools = [
    ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_current_datetime",
            description="Get the current date and time.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    ),
    ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="web_search",
            description=(
                "Search for information on the internet. Use this function to look up information online via "
                "duckduckgo.com"
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
    ),
    ChatCompletionToolParam(
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
    ),
    ChatCompletionToolParam(
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
    ),
]

registered_functions: dict[str, Callable[..., Awaitable[Any]]] = {
    "get_current_datetime": get_current_datetime,
    "web_search": web_search,
    "search_news": search_news,
    "read_web_page": read_web_page,
}
