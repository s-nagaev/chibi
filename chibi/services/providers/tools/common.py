import asyncio
import datetime
from typing import Any, Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import (
    AdditionalOptions,
    get_models_available_to_user,
    get_sub_agent_response,
    resolve_session_context,
)
from chibi.services.subagent_events import subagent_tracker
from chibi.utils.app import convert_list_of_models_to_str


class GetAvailableLLMModelsTool(ChibiTool):
    register = False
    loop_warning: int = 10
    loop_break: int = 12
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_available_llm_models",
            description="Get LLM models and providers available for user.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "get_available_llm_models"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        logger.log("TOOL", f"Getting available LLM models for user {user_id}...")

        data: list[ModelChangeSchema] = await get_models_available_to_user(user_id=user_id, image_generation=False)

        return {
            "available_models": convert_list_of_models_to_str(models=data),
        }


class DelegateTool(ChibiTool):
    register = gpt_settings.allow_delegation
    run_in_background_by_default = True
    allow_model_to_change_background_mode = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="delegate_task",
            description=(
                "Delegate exactly one task to a sub-agent - an LLM identical to you. The prompt should be "
                "exhaustive and expect a concrete result, or an explanation for its absence. The task should be "
                "as atomic as possible. Delegate preferably tasks that involve processing large volumes of "
                "information, to avoid saturating your context. Try to assign simpler tasks to cheaper and faster "
                "models."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt"},
                    "provider_name": {"type": "string", "description": "Provider name, i.e. 'OpenAI'"},
                    "model_name": {"type": "string", "description": "Model name, i.e. 'gpt-5.2'"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 0},
                },
                "required": ["prompt", "provider_name", "model_name"],
            },
        ),
    )
    name = "delegate_task"

    @classmethod
    async def function(
        cls,
        prompt: str,
        provider_name: str | None = None,
        model_name: str | None = None,
        timeout: float | None = gpt_settings.delegate_task_timeout,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        if model_name and not provider_name:
            raise ToolException("If you specify a model_name, you must also specify a provider_name.")

        if provider_name and not model_name:
            raise ToolException("If you specify a provider_name, you must also specify a model_name.")

        caller_model = kwargs.get("caller_model", "unknown model")
        caller_provider = kwargs["caller_provider"]

        if model_name:
            assert isinstance(provider_name, str)
            available_models: list[ModelChangeSchema] = await get_models_available_to_user(user_id=user_id)

            model_setups = [
                model for model in available_models if model.name == model_name and model.provider == provider_name
            ]
            if not model_setups:
                model_provider_map = (f"{model.name} ({model.provider})" for model in available_models)
                raise ToolException(
                    f"Model name <-> Provider name mismatch. Please check your model name and provider name."
                    f"Available models: {', '.join(model_provider_map)}"
                )
        else:
            model_name = caller_model
            provider_name = caller_provider

        logger.log("DELEGATE", f"[{caller_model}] Delegating a task to {model_name}: {prompt}")

        caller_storage_id, caller_thread_id = resolve_session_context(**kwargs)

        coro = get_sub_agent_response(
            user_id=user_id,
            prompt=prompt,
            provider_name=provider_name,
            model_name=model_name,
            caller_storage_id=caller_storage_id,
            caller_thread_id=caller_thread_id,
        )
        # Every validation failure above raised before this line, so a
        # failed delegation attempt (bad tool params and the like) is never
        # counted and never emits a started event. The matching finally
        # below guarantees the finished event on every exit path, including
        # the timeout and cancellation branches.
        subagent_tracker.subagent_started(caller_thread_id, name=model_name)
        try:
            response: ChatResponseSchema = await asyncio.wait_for(fut=coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolException("Timed out waiting for delegated task to complete!")
        finally:
            subagent_tracker.subagent_finished(caller_thread_id, name=model_name)

        logger.log("SUBAGENT", f"[{model_name or caller_model}] Delegated task is done: {response.answer}")

        return {"response": response.answer}


class GetCurrentDatetimeTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_current_datetime",
            description="Get the current date and time.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    )
    name = "get_current_datetime"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        logger.log("TOOL", f"[{kwargs.get('caller_model', 'unknown model')}] Getting current date & time")
        now = datetime.datetime.now()
        return {
            "datetime_now": now.strftime("%Y-%m-%d %H:%M:%S"),
        }


class SayTool(ChibiTool):
    """Terminal delivery tool: the only channel for user-facing communication.

    Never dispatched as an ordinary tool: the agent loop must intercept a
    ``say()`` call, persist ``text`` as the visible assistant message and end
    the turn without producing a tool result (sub-task 4b). Until that branch
    lands, the base-class dispatch path would treat this call like any other
    tool, which is the acknowledged intermediate state of sub-task 4a.
    """

    register = True
    run_in_background_by_default = False
    allow_model_to_change_background_mode = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="say",
            description=(
                "Deliver a message to the user. This is the ONLY channel for user-facing "
                "communication: text written outside this tool is never shown to the user. "
                "Call say() whenever the user is owed an answer or anything worth reporting. "
                "Calling say() ENDS your turn — it must be the last call of the turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The message the user will see.",
                    },
                },
                "required": ["text"],
            },
        ),
    )
    name = "say"


class AckSilentlyTool(ChibiTool):
    """Terminal service acknowledgement: ends the turn with nothing delivered.

    The user never sees this response; it is not a quiet version of ``say()``.
    Never dispatched as an ordinary tool: the agent loop must intercept an
    ``ack_silently()`` call, persist a hidden assistant message and end the
    turn without producing a tool result (sub-task 4b). Until that branch
    lands, the base-class dispatch path would treat this call like any other
    tool, which is the acknowledged intermediate state of sub-task 4a.
    """

    register = True
    run_in_background_by_default = False
    allow_model_to_change_background_mode = False
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="ack_silently",
            description=(
                "Silently absorb a background event (a tool or scheduler result) and end the "
                "turn with NOTHING delivered to the user. This is a service response: the user "
                "NEVER sees it, and it is not a quiet version of say(). Use it only when you "
                "have nothing to tell the user and work continues; if the user is owed an "
                "answer, call say() instead. Calling ack_silently() ENDS your turn — it must "
                "be the last call of the turn."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "ack_silently"
