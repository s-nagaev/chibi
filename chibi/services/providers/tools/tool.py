from __future__ import annotations

import json
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar, cast

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from chibi.config import gpt_settings
from chibi.interfaces.interface import UserInterface
from chibi.services.events.tool_events import tool_event_tracker
from chibi.services.providers.tools.exceptions import LoopDetectedException, NoUserInterfaceProvidedException
from chibi.services.providers.tools.schemas import ToolResponseSchema
from chibi.services.providers.tools.utils import AdditionalOptions, CallTracker
from chibi.services.providers.utils import escape_and_truncate
from chibi.services.task_manager import task_manager

P = ParamSpec("P")
R = TypeVar("R")

ToolFunction = Callable[P, Coroutine[Any, Any, ToolResponseSchema]]
RegisteredFunctionsMap = dict[str, ToolFunction]
ToolsDefinitionMap = dict[str, ChatCompletionToolParam]

_TOOL_EVENT_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "interface",
        "user_id",
        "caller_model",
        "caller_provider",
        "caller_storage_id",
        "caller_thread_id",
        "telegram_context",
        "telegram_update",
    }
)


def _tool_event_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the model-supplied tool arguments for a ``tool_event`` frame.

    The session-context keys injected by the provider layer are dropped so the
    frame carries only what the model actually asked the tool to do. Values are
    passed through unchanged; the transport applies the size cap.

    Args:
        kwargs: The full tool-invocation payload.

    Returns:
        The tool arguments without the injected session-context keys.
    """
    return {key: value for key, value in kwargs.items() if key not in _TOOL_EVENT_CONTEXT_KEYS}


def _tool_event_preview(response: ToolResponseSchema) -> str:
    """Return a textual preview of a tool result for a ``tool_event`` frame.

    Args:
        response: The tool response whose result is previewed.

    Returns:
        The result as-is when it is a string, otherwise its JSON rendering.
    """
    if isinstance(response.result, str):
        return response.result
    return json.dumps(response.result, ensure_ascii=False, default=str)


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
    def get_interface(cls, kwargs: AdditionalOptions | dict[str, Any]) -> UserInterface:
        if interface := kwargs.get("interface"):
            return interface
        raise NoUserInterfaceProvidedException

    @classmethod
    async def _get_tool_call_result(cls, *args, **kwargs) -> ToolResponseSchema:
        try:
            result = await cls.function(**kwargs)
            logger.log(
                "CALL",
                (
                    f"[{kwargs.get('caller_model', 'unknown model')}] Function '{cls.name}' called, "
                    f"result retrieved: {escape_and_truncate(result)}"
                ),
            )
            return ToolResponseSchema(tool_name=cls.name, status="ok", result=result)
        except Exception as e:
            logger.warning(f"[{kwargs.get('caller_model', 'unknown model')}] Tool {cls.name} raised an exception: {e}")
            return ToolResponseSchema(tool_name=cls.name, status="error", result=str(e))

    @classmethod
    async def _get_and_send_tool_call_result(cls, *args, **kwargs) -> None:
        from chibi.services.bot import handle_tool_response

        interface = cls.get_interface(kwargs)
        tool_call_result = await cls._get_tool_call_result(*args, **kwargs)
        await handle_tool_response(tool_response=tool_call_result, interface=interface)

    @classmethod
    async def _run_foreground(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> ToolResponseSchema:
        """Run a foreground tool call, emitting ``tool_event`` frames when opted in.

        Frames are emitted only for a live, in-session foreground call: the
        interface must be present and still open, and a sink must be bound
        (the client declared the ``tool_events`` capability at handshake).
        Background runs and interface-less (sub-agent-internal) calls never
        reach this method, so they emit nothing. Emission is best effort: the
        tracker swallows transport failures, so a broken stdout never affects
        the tool result.

        Args:
            args: Positional arguments forwarded to the tool function.
            kwargs: Keyword arguments forwarded to the tool function.

        Returns:
            The tool response, exactly as the plain foreground path returns it.
        """
        interface = kwargs.get("interface")
        if interface is None or getattr(interface, "is_closed", False) or not tool_event_tracker.enabled:
            return await cls._get_tool_call_result(*args, **kwargs)
        thread_id = interface.thread_id
        await tool_event_tracker.tool_started(thread_id, cls.name, _tool_event_args(kwargs))
        try:
            response = await cls._get_tool_call_result(*args, **kwargs)
        except Exception as exc:
            await tool_event_tracker.tool_finished(thread_id, cls.name, False, str(exc))
            raise
        await tool_event_tracker.tool_finished(
            thread_id, cls.name, response.status == "ok", _tool_event_preview(response)
        )
        return response

    @classmethod
    async def tool(cls, *args, run_in_background: bool | None = None, **kwargs: Any) -> ToolResponseSchema:
        interface = kwargs.get("interface")
        if run_in_background is None:
            run_in_background = cls.run_in_background_by_default
        background_run_ready = run_in_background and interface

        printable_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("telegram_context", "telegram_update", "interface")
        }
        caller_model = kwargs.get("caller_model", "unknown model")
        logger.log(
            "CALL",
            (
                f"[{caller_model}] Calling a function '{cls.name}' "
                f"in {'background' if background_run_ready else 'foreground'} mode. "
                f"Args: {escape_and_truncate(printable_kwargs)}"
            ),
        )
        hit_count = CallTracker().track(
            model_name=caller_model, tool_name=cls.name, serializable_kwargs=printable_kwargs
        )
        if hit_count > cls.loop_break:
            raise LoopDetectedException(
                f"A cyclical call of the {cls.name} function by model {caller_model} has been interrupted."
            )

        if hit_count > cls.loop_warning:
            logger.warning(
                f"[{caller_model}] Model seems stuck in a loop calling tool '{cls.name}'. Call #{hit_count}."
            )
            return ToolResponseSchema(
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
            return await cls._run_foreground(args, kwargs)

        coro = cls._get_and_send_tool_call_result(*args, **kwargs)
        user_id = kwargs.get("user_id", -1)
        task_manager.run_task(coro=coro, user_id=user_id, thread_id=interface.thread_id if interface else 0)
        return ToolResponseSchema(
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
    async def call(cls, tool_name: str, tools_args: dict[str, Any]) -> ToolResponseSchema:
        tool = cls.get(tool_name)
        return await tool(**tools_args)

    @classmethod
    async def _stub_function(cls, *args: Any, **kwargs: Any) -> ToolResponseSchema:
        """A stub function that is executed when the LLM calls a non-existent function.

        Returns:
            A ToolResponse object describing the error.
        """
        logger.log("TOOL", f"Running stub function. Args: {args}, kwargs: {kwargs}")
        return ToolResponseSchema(tool_name="stub", status="error", result="A non-existent function called")
