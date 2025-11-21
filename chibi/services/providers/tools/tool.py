from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from chibi.services.providers.tools.schemas import ToolResponse

P = ParamSpec("P")
R = TypeVar("R")

RegisteredFunctionsMap = dict[str, Callable[P, Coroutine[Any, Any, ToolResponse]]]


class RegisteredChibiTools:
    tools: set[type["ChibiTool"]] = set()

    @classmethod
    def get_tool_definitions(cls) -> list[ChatCompletionToolParam]:
        return [tool.definition for tool in cls.tools]

    @classmethod
    def get_registered_functions(cls) -> RegisteredFunctionsMap:
        registered_functions = {tool.name: tool.tool for tool in cls.tools}
        registered_functions["stub_function"] = cls._stub_function
        return registered_functions

    @classmethod
    async def _stub_function(cls, *args: Any, **kwargs: Any) -> ToolResponse:
        """A stub function that is executed when the LLM calls a non-existent function.

        Returns:
            A ToolResponse object describing the error.
        """
        logger.log("TOOL", f"Running stub function. Args: {args}, kwargs: {kwargs}")
        return ToolResponse(tool_name="stub", status="error", result="A non-existent function called")


class ChibiTool:
    register: bool
    definition: ChatCompletionToolParam
    name: str

    @classmethod
    async def tool(cls, *args: Any, **kwargs: Any) -> ToolResponse:
        try:
            result = await cls.function(*args, **kwargs)
            return ToolResponse(tool_name=cls.name, status="ok", result=result)
        except Exception as e:
            logger.warning(f"Tool {cls.name} raised an exception: {e}")
            return ToolResponse(tool_name=cls.name, status="error", result=str(e))

    @classmethod
    async def function(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls.register:
            RegisteredChibiTools.tools.add(cls)
