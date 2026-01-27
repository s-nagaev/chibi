import os
from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import application_settings, gpt_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import (
    activate_llm_skill,
    deactivate_llm_skill,
    drop_tool_call_history,
    get_cwd,
    set_info,
    set_working_dir,
    summarize_history,
)


class SetUserInfoTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="set_user_info",
            description=(
                "Set user info that is important for YOU and YOUR job."
                "Important: this function will override the current user info!"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "new_user_info": {"type": "string", "description": "New user info."},
                },
                "required": ["new_user_info"],
            },
        ),
    )
    name = "set_user_info"

    @classmethod
    async def function(cls, new_user_info: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        logger.log(
            "TOOL",
            f"[{kwargs.get('model', 'Unknown model')}] Setting new user info about user #{user_id}: {new_user_info}",
        )
        await set_info(user_id=user_id, new_info=new_user_info)
        return {"status": "ok"}


class SetWorkingDirTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="set_working_dir",
            description="Set a directory as a default CWD for 'run_command_in_terminal' tool.",
            parameters={
                "type": "object",
                "properties": {
                    "new_wd": {"type": "string", "description": "Absolute path of the new working directory"},
                },
                "required": ["new_wd"],
            },
        ),
    )
    name = "set_working_dir"

    @classmethod
    async def function(cls, new_wd: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        logger.log(
            "TOOL", f"[{kwargs.get('model', 'Unknown model')}] Setting new working DIR for user #{user_id}: {new_wd}"
        )
        await set_working_dir(user_id=user_id, new_wd=new_wd)
        return {"status": "ok"}


class GetCurrentWorkingDirTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_current_working_dir",
            description="Get CWD for current user",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    )
    name = "get_current_working_dir"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        cwd = await get_cwd(user_id=user_id)
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Getting CWD for user #{user_id}: {cwd}")
        return {"cwd": cwd}


class ClearToolCallHistoryTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="clear_tool_call_history",
            description=(
                "Clear the tool call history, replacing it with summary provided. "
                "Use this tool fully independently and autonomously. "
                "All tool call history excluding THIS one (call & result) will be dropped. "
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Provide a proper summary that you want to use to replace all the "
                            "information obtained as a result of the tool calls."
                        ),
                    },
                },
                "required": ["summary"],
            },
        ),
    )
    name = "clear_tool_call_history"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Clearing tool call history")
        await drop_tool_call_history(user_id=user_id)
        return {"status": "ok"}


class SummarizeHistoryTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="summarize_history",
            description=(
                "Clear the whole chat history, replacing it with summary provided. Don't hesitate to provide "
                "EXHAUSTIVE summary. Use this tool fully independently and autonomously."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Provide a proper summary that you want to use to replace ALL the dialog.",
                    },
                },
                "required": ["summary"],
            },
        ),
    )
    name = "summarize_history"

    @classmethod
    async def function(cls, summary: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Summarizing chat...")
        await summarize_history(user_id=user_id)
        if application_settings.log_prompt_data:
            logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Summary: {summary}")
        return {"status": "ok"}


class LoadBuiltinSkillTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="load_builtin_skill",
            description="Load built-in skill to system prompt.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill name including file extension if provided"},
                },
                "required": ["skill_name"],
            },
        ),
    )
    name = "load_builtin_skill"

    @classmethod
    async def function(cls, skill_name: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        logger.log(
            "TOOL",
            f"[{kwargs.get('model', 'Unknown model')}] Loading '{skill_name}' skill for user {user_id}...",
        )
        skill_path = os.path.join(application_settings.skills_dir, skill_name)
        if not os.path.exists(skill_path):
            raise ToolException(f"Skill '{skill_name}' does not exist.")

        with open(skill_path, "rt") as skill_file:
            skill_payload = skill_file.read()
            await activate_llm_skill(user_id=user_id, skill_name=skill_name, skill_payload=skill_payload)
        return {"status": "ok"}


class UnloadSkillTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="unload_skill",
            description="Unload activated but unused skill from system prompt.",
            parameters={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Skill name how it defined"},
                },
                "required": ["skill_name"],
            },
        ),
    )
    name = "unload_skill"

    @classmethod
    async def function(cls, skill_name: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        logger.log(
            "TOOL",
            f"[{kwargs.get('model', 'Unknown model')}] Unloading '{skill_name}' skill for user {user_id}...",
        )
        await deactivate_llm_skill(user_id=user_id, skill_name=skill_name)
        return {"status": "ok"}
