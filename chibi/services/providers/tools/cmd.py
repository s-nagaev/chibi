import asyncio
import json
from pathlib import Path
from typing import Any, Unpack

from aiocache import cached
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from chibi.config import application_settings, gpt_settings
from chibi.models import Message
from chibi.services.providers.tools.constants import CMD_STDOUT_LIMIT, MODERATOR_PROMPT
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions


class ModeratorsAnswer(BaseModel):
    status: str
    verdict: str
    reason: str | None = None


@cached(ttl=120)
async def moderate_command(cmd: str) -> ModeratorsAnswer:
    from chibi.services.providers import Gemini

    if not gpt_settings.gemini_key:
        raise ToolException("Moderator is not configured")  # TODO: very temporary solution
    moderator = Gemini(token=gpt_settings.gemini_key)

    messages = [
        Message(role="user", content=cmd),
    ]
    response, _ = await moderator.get_chat_response(
        messages=messages,
        system_prompt=MODERATOR_PROMPT,
    )
    answer = response.answer
    answer = answer.strip("```")
    answer = answer.strip("json")
    answer = answer.strip()
    try:
        result_data = json.loads(answer)
    except Exception:
        logger.error(f"Error parsing moderator's response: {answer}")
        return ModeratorsAnswer(verdict="declined", reason=answer, status="error")

    verdict = result_data.get("verdict", "declined")
    if verdict == "accepted":
        return ModeratorsAnswer(verdict="accepted", status="ok")

    reason = result_data.get("reason", None)
    if reason is None:
        logger.error(f"Moderator did not provide reason properly: {answer}")

    return ModeratorsAnswer(verdict="declined", reason=reason, status="operation aborted")


class RunCommandInTerminalTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="run_command_in_terminal",
            description=(
                "Run command in the zsh shell (MacOS). Will run via python's subprocess.run() "
                "Will return json including return code, stdout and stderr."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run."},
                    "cwd": {
                        "type": "string",
                        "description": (
                            f"The working directory to run the command in. Default: {application_settings.home_dir}"
                        ),
                    },
                },
                "required": ["cmd"],
            },
        ),
    )
    name = "run_command_in_terminal"

    @classmethod
    async def function(
        cls, cmd: str, cwd: str = application_settings.home_dir, **kwargs: Unpack[AdditionalOptions]
    ) -> dict[str, Any]:
        logger.log("CHECK", f"Pre-moderating command: '{cmd}'")
        moderator_answer: ModeratorsAnswer = await moderate_command(cmd)
        if moderator_answer.verdict == "declined":
            raise ToolException(f"Moderator DECLINED command '{cmd}'. Reason: {moderator_answer.reason}")

        logger.log("CHECK", f"Moderator ACCEPTED command '{cmd}'")
        logger.log("TOOL", f"Running command in terminal: {cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
            )
            stdout, stderr = await process.communicate()
        except Exception as e:
            raise ToolException(f"Failed to run command in terminal! Command: '{cmd}'. Error: {e}")

        raw_stdout = stdout.decode()
        raw_stderr = stderr.decode()

        result: dict[str, str | int | None] = {"return_code": process.returncode}

        if len(raw_stdout) > CMD_STDOUT_LIMIT or len(raw_stderr) > CMD_STDOUT_LIMIT:
            result["WARNING"] = (
                "The volume of stdout/stderr data is excessively large "
                f"(over {CMD_STDOUT_LIMIT} characters). If this is the "
                f"result of reading from a file, try reading it in parts."
            )
        result["stdout"] = (
            raw_stdout if len(raw_stdout) < CMD_STDOUT_LIMIT else f"...truncated... {raw_stdout[CMD_STDOUT_LIMIT:]}"
        )
        result["stderr"] = (
            raw_stderr if len(raw_stderr) < CMD_STDOUT_LIMIT else f"...truncated... {raw_stderr[CMD_STDOUT_LIMIT:]}"
        )

        logger.log(
            "TOOL",
            f"Command '{cmd}' executed. Return code: {process.returncode}.",
        )
        return result


class CreateFileTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="create_file",
            description="Create a file at the given full path (including any directories).",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "File content",
                    },
                    "full_path": {
                        "type": "string",
                        "description": "Full file name, including file path.",
                    },
                    "overwrite": {
                        "type": "string",
                        "enum": ["true", "false"],
                        "description": (
                            "Set it to true to overwrite the file. If overwrite is false and "
                            "the file exists, raises FileExistsError."
                        ),
                    },
                },
                "required": ["content", "full_path", "overwrite"],
            },
        ),
    )
    name = "create_file"

    @classmethod
    async def function(
        cls,
        full_path: str,
        content: str,
        overwrite: str = "false",
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        try:
            path = Path(full_path).expanduser().resolve()
            parent = path.parent
            parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                if overwrite != "true":
                    raise FileExistsError(f"File {path} already exists")
                logger.log("TOOL", f"File {path} exists. Overwriting.")

            with path.open("w", encoding=encoding) as f:
                f.write(content)

            logger.log("TOOL", f"File {path} created successfully.")
            return {"file": str(path)}

        except Exception as e:
            raise ToolException(f"Failed to create file {full_path}. Error: {e}")
