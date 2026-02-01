import asyncio
import json
import os
import signal
from typing import Any, Unpack

from aiocache import cached
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel

from chibi.config import gpt_settings
from chibi.models import Message
from chibi.services.providers.tools.constants import CMD_STDOUT_LIMIT, MODERATOR_PROMPT
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import get_cwd


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
                            "The working directory to run the command in. The default value provided in user data."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "The timeout for command execution in seconds. Default is 30 sec. "
                            "Change it if you're expecting longer execution."
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
        cls, cmd: str, cwd: str | None = None, timeout: int = 30, **kwargs: Unpack[AdditionalOptions]
    ) -> dict[str, Any]:
        model = kwargs.get("model", "Unknown model")
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")

        if not cwd:
            cwd = await get_cwd(user_id=user_id)

        logger.log("MODERATOR", f"[{model}] Pre-moderating command: '{cmd}'. CWD: {cwd}")
        moderator_answer: ModeratorsAnswer = await moderate_command(cmd)
        if moderator_answer.verdict == "declined":
            raise ToolException(
                f"[{model}] Moderator DECLINED command '{cmd}' from model {kwargs.get('model', 'unknown')}. "
                f"Reason: {moderator_answer.reason}"
            )

        logger.log(
            "MODERATOR", f"[{model}] Moderator ACCEPTED command '{cmd}' from model {kwargs.get('model', 'unknown')}"
        )
        logger.log("TOOL", f"[{model}] Running command in terminal: {cmd}. CWD: {cwd}. Timeout: {timeout}")
        try:
            process = await asyncio.create_subprocess_shell(
                cmd=cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=float(timeout))
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

            raise ToolException(f"Command execution timed out after {timeout} seconds. Process group killed.")
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
            f"[{kwargs.get('model', 'Unknown model')}] Command '{cmd}' executed. Return code: {process.returncode}.",
        )
        return result
