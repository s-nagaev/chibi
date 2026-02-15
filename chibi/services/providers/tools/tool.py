from __future__ import annotations

from typing import Any, Callable, Coroutine, ParamSpec, TypeVar, cast

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from telegram import Update
from telegram.ext import ContextTypes

from chibi.config import gpt_settings
from chibi.services.providers.tools.exceptions import LoopDetectedException
from chibi.services.providers.tools.schemas import ToolResponse
from chibi.services.providers.tools.utils import CallTracker
from chibi.services.providers.utils import escape_and_truncate
from chibi.services.task_manager import task_manager

P = ParamSpec("P")
R = TypeVar("R")

ToolFunction = Callable[P, Coroutine[Any, Any, ToolResponse]]
RegisteredFunctionsMap = dict[str, ToolFunction]
ToolsDefinitionMap = dict[str, ChatCompletionToolParam]


class ChibiTool:
    register: bool
    definition: ChatCompletionToolParam
    name: str
    run_in_background_by_default: bool = False
    allow_model_to_change_background_mode: bool = True
    loop_warning: int = 5
    loop_break: int = 7

    @classmethod
    def add_global_params(cls) -> dict[str, Any]:
        return {
            "run_in_background": {
                "type": "boolean",
                "description": "Execute task in background. You'll receive a result when it done.",
                "default": cls.run_in_background_by_default,
            }
        }

    @classmethod
    async def _get_tool_call_result(cls, *args, **kwargs) -> ToolResponse:
        try:
            result = await cls.function(**kwargs)
            logger.log(
                "CALL",
                (
                    f"[{kwargs.get('model', 'Unknown model')}] Function '{cls.name}' called, "
                    f"result retrieved: {escape_and_truncate(result)}"
                ),
            )
            return ToolResponse(tool_name=cls.name, status="ok", result=result)
        except Exception as e:
            logger.warning(f"[{kwargs.get('model', 'Unknown model')}] Tool {cls.name} raised an exception: {e}")
            return ToolResponse(tool_name=cls.name, status="error", result=str(e))

    @classmethod
    async def _get_and_send_tool_call_result(
        cls, *args, update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs
    ) -> None:
        from chibi.services.bot import handle_tool_response

        tool_call_result = await cls._get_tool_call_result(*args, **kwargs)
        await handle_tool_response(tool_response=tool_call_result, update=update, context=context)

    @classmethod
    async def tool(cls, *args, run_in_background: bool | None = None, **kwargs: Any) -> ToolResponse:
        telegram_context = kwargs.get("telegram_context")
        telegram_update = kwargs.get("telegram_update")
        if run_in_background is None:
            run_in_background = cls.run_in_background_by_default
        background_run_ready = run_in_background and telegram_update and telegram_context

        printable_kwargs = {k: v for k, v in kwargs.items() if k not in ("telegram_context", "telegram_update")}
        model_name = kwargs.get("model", "Unknown model")
        logger.log(
            "CALL",
            (
                f"[{model_name}] Calling a function '{cls.name}' "
                f"in {'background' if background_run_ready else 'foreground'} mode. "
                f"Args: {escape_and_truncate(printable_kwargs)}"
            ),
        )
        hit_count = CallTracker().track(model_name=model_name, tool_name=cls.name, serializable_kwargs=printable_kwargs)
        if hit_count > cls.loop_break:
            raise LoopDetectedException(
                f"A cyclical call of the {cls.name} function by model {model_name} has been interrupted."
            )

        if hit_count > cls.loop_warning:
            logger.warning(f"[{model_name}] Model seems stuck in a loop calling tool '{cls.name}'. Call #{hit_count}.")
            return ToolResponse(
                tool_name=cls.name,
                status="WARNING",
                result="ABORTED",
                additional_details=(
                    f"You have already called the same tool with the same parameters {hit_count} times in a row. "
                    "Most likely, you are stuck in a loop. STOP carrying out the task and clarify the "
                    "assignment with the user IMMEDIATELY!"
                ),
            )

        if not background_run_ready:
            return await cls._get_tool_call_result(*args, **kwargs)

        assert telegram_update
        assert telegram_context

        coro = cls._get_and_send_tool_call_result(*args, update=telegram_update, context=telegram_context, **kwargs)
        task_manager.run_task(coro)
        return ToolResponse(
            tool_name=cls.name,
            status="tool is running in background",
            result="in progress",
            additional_details="you'll receive the result when it ready",
        )

    @classmethod
    async def function(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not cls.register:
            return None

        if gpt_settings.tools_whitelist and cls.name not in gpt_settings.tools_whitelist:
            return None

        if cls.allow_model_to_change_background_mode:
            cast(dict, cls.definition["function"]["parameters"]["properties"]).update(cls.add_global_params())

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
