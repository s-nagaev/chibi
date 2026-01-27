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
from chibi.services.providers.tools.utils import AdditionalOptions, get_sub_agent_response


class GetAvailableLLMModelsTool(ChibiTool):
    register = True
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

        from chibi.services.user import get_models_available

        data: list[ModelChangeSchema] = await get_models_available(user_id=user_id, image_generation=False)

        return {
            "available_models": [info.model_dump(include={"provider", "name", "display_name"}) for info in data],
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
                "models. You can find out the list of available models by executing tool get_available_llm_models. "
                "If no model/provider specified, your model will be used (be sure you know your model)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Prompt"},
                    "provider_name": {"type": "string", "description": "Provider name, i.e. 'OpenAI'"},
                    "model_name": {"type": "string", "description": "Model name, i.e. 'gpt-5.2'"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 600},
                },
                "required": ["prompt"],
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
        timeout: int = 600,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        model = kwargs.get("model")
        if not model:
            raise ToolException("This function requires model to be automatically provided.")
        logger.log("DELEGATE", f"[{model}] Delegating a task to {model_name or model}: {prompt}")

        coro = get_sub_agent_response(
            user_id=user_id, prompt=prompt, provider_name=provider_name, model_name=model_name
        )
        try:
            response: ChatResponseSchema = await asyncio.wait_for(fut=coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolException("Timed out waiting for delegated task to complete!")

        logger.log("SUBAGENT", f"[{model_name or model}] Delegated task is done: {response.answer}")

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
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Getting current date & time")
        now = datetime.datetime.now()
        return {
            "datetime_now": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
