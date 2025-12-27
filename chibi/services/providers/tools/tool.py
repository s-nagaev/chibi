from __future__ import annotations

import json
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from chibi.services.providers.tools.schemas import ToolResponse
from chibi.services.providers.utils import escape_and_truncate

P = ParamSpec("P")
R = TypeVar("R")

ToolFunction = Callable[P, Coroutine[Any, Any, ToolResponse]]
RegisteredFunctionsMap = dict[str, ToolFunction]
ToolsDefinitionMap = dict[str, ChatCompletionToolParam]


class ChibiTool:
    register: bool
    definition: ChatCompletionToolParam
    name: str

    @classmethod
    async def tool(cls, *args: Any, **kwargs: Any) -> ToolResponse:
        # non_printable_kwargs = list(AdditionalOptions.__annotations__.keys())
        printable_kwargs = {k: v for k, v in kwargs.items() if k not in ("telegram_context", "telegram_update")}
        logger.log(
            "CALL", f"Calling a function '{cls.name}'. Args: {escape_and_truncate(json.dumps(printable_kwargs))}"
        )
        try:
            result = await cls.function(*args, **kwargs)
            logger.log(
                "CALL", f"Function '{cls.name}' called, result retrieved: {escape_and_truncate(json.dumps(result))}"
            )
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
            RegisteredChibiTools.register(cls)


class RegisteredChibiTools:
    tools_map: dict[str, type[ChibiTool]] = {}

    @classmethod
    def get_tool_definitions(cls) -> list[ChatCompletionToolParam]:
        return [tool.definition for tool in cls.tools_map.values()]

    @classmethod
    def get_registered_functions(cls) -> RegisteredFunctionsMap:
        registered_functions = {name: tool.tool for name, tool in cls.tools_map.items()}
        registered_functions["stub_function"] = cls._stub_function
        return registered_functions

    @classmethod
    def register(cls, tool: type[ChibiTool]) -> None:
        cls.tools_map[tool.name] = tool

    @classmethod
    def deregister_tools(cls, tool_names: list[str]) -> None:
        for tool_name in tool_names:
            if tool_name not in cls.tools_map:
                continue
            cls.tools_map.pop(tool_name)
            logger.info(f"The tool {tool_name} had been deregistered.")

    @classmethod
    def get(cls, tool_name: str) -> ToolFunction:
        if chibi_tool_class := cls.tools_map.get(tool_name):
            return chibi_tool_class.tool
        logger.error(f"Function {tool_name} called but it's not registered.")
        return cls._stub_function

    @classmethod
    async def call(cls, tool_name: str, tools_args: dict[str, Any]) -> ToolResponse:
        tool = cls.get(tool_name)
        return await tool(**tools_args)

    @classmethod
    async def _stub_function(cls, *args: Any, **kwargs: Any) -> ToolResponse:
        """A stub function that is executed when the LLM calls a non-existent function.

        Returns:
            A ToolResponse object describing the error.
        """
        logger.log("TOOL", f"Running stub function. Args: {args}, kwargs: {kwargs}")
        return ToolResponse(tool_name="stub", status="error", result="A non-existent function called")
