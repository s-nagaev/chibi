"""Tests for the standardized moderator-unavailable handling."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

import chibi.services.jobs.agent_task as agent_task_module
from chibi.exceptions import ServiceRateLimitError
from chibi.schemas.app import ModeratorsAnswer
from chibi.services.jobs.agent_task import run_agent_job
from chibi.services.providers.tools._moderation import (
    MODERATOR_UNAVAILABLE_MESSAGE_TEMPLATE,
    ModeratorUnavailableError,
    moderate_command_for_tool,
    moderate_command_standardized,
)
from chibi.services.providers.tools.cmd import RunCommandInTerminalTool
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.scheduler import ScheduleTaskTool

pytestmark = pytest.mark.asyncio

VALID_PAYLOAD = {
    "user_id": 1,
    "storage_id": 1,
    "chat_id": 555,
    "thread_id": 42,
    "title": "Health heartbeat",
}


def _api_status_error(status_code: int) -> APIStatusError:
    """Build a real openai.APIStatusError with the given HTTP status code."""
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx2.Response(status_code=status_code, request=request)
    return APIStatusError("Moderation backend error", response=response, body=None)


def _connection_error() -> APIConnectionError:
    """Build a real openai.APIConnectionError."""
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
    return APIConnectionError(request=request)


def _moderation_provider(answer: Any = None, error: Exception | None = None) -> MagicMock:
    """Build a moderation provider mock whose `moderate_command` answers or raises."""
    provider = MagicMock()
    provider.name = "mock-moderator"
    provider.default_moderation_model = "moderator-model"
    provider.default_model = "default-model"
    if error is not None:
        provider.moderate_command = AsyncMock(side_effect=error)
    else:
        provider.moderate_command = AsyncMock(return_value=answer)
    return provider


def _expected_message(code: str, error_type: str) -> str:
    """The standardized message with the given code and error type substituted."""
    return MODERATOR_UNAVAILABLE_MESSAGE_TEMPLATE.format(code=code, error_type=error_type)


class TestModerateCommandStandardized:
    """Failure mapping of the shared moderation helper."""

    async def test_api_status_error_500_maps_code_and_type(self):
        """An APIStatusError 500 surfaces with its HTTP code and class name."""
        provider = _moderation_provider(error=_api_status_error(500))
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "500"
        assert exc_info.value.error_type == "APIStatusError"
        assert exc_info.value.message == _expected_message("500", "APIStatusError")

    async def test_api_status_error_429_maps_rate_limit_code(self):
        """An APIStatusError 429 surfaces with the rate-limit code."""
        provider = _moderation_provider(error=_api_status_error(429))
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "429"

    async def test_openai_rate_limit_error_maps_429(self):
        """An openai.RateLimitError surfaces as HTTP 429."""
        request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")
        response = httpx2.Response(status_code=429, request=request)
        error = RateLimitError("Too many requests", response=response, body=None)
        provider = _moderation_provider(error=error)
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "429"
        assert exc_info.value.error_type == "RateLimitError"

    async def test_service_rate_limit_error_maps_429(self):
        """A ServiceRateLimitError surfaces with the mapped 429 code."""
        provider = _moderation_provider(error=ServiceRateLimitError(provider="mock-moderator"))
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "429"
        assert exc_info.value.error_type == "ServiceRateLimitError"

    async def test_connection_error_maps_na(self):
        """A connection error surfaces with the n/a code fallback."""
        provider = _moderation_provider(error=_connection_error())
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "n/a"
        assert exc_info.value.error_type == "APIConnectionError"

    async def test_generic_provider_exception_maps_na(self):
        """Any other provider exception surfaces with the n/a code fallback."""
        provider = _moderation_provider(error=RuntimeError("provider exploded"))
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "n/a"
        assert exc_info.value.error_type == "RuntimeError"

    async def test_error_status_answer_raises(self):
        """An answer with status='error' surfaces as an invalid moderator response."""
        answer = ModeratorsAnswer(verdict="declined", status="error", reason="<unparsable garbage>")
        provider = _moderation_provider(answer=answer)
        with pytest.raises(ModeratorUnavailableError) as exc_info:
            await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert exc_info.value.code == "n/a"
        assert exc_info.value.error_type == "InvalidModeratorResponse"

    @pytest.mark.parametrize(
        "answer",
        [
            ModeratorsAnswer(verdict="declined", status="operation aborted", reason="rm -rf is not allowed"),
            ModeratorsAnswer(verdict="declined", status=None, reason="no tool call received"),
            ModeratorsAnswer(verdict="declined", status="ok", reason="too dangerous"),
        ],
    )
    async def test_genuine_declines_pass_through_unchanged(self, answer):
        """Genuine declines keep their original status and are not remapped."""
        provider = _moderation_provider(answer=answer)
        result = await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert result is answer
        provider.moderate_command.assert_awaited_once()

    async def test_accepted_answer_passes_through(self):
        """An accepted answer passes through without conversion."""
        answer = ModeratorsAnswer(verdict="accepted", status="ok")
        provider = _moderation_provider(answer=answer)
        result = await moderate_command_standardized(provider, cmd="ls", model="moderator-model")

        assert result is answer

    async def test_failure_logged_at_moderator_level_with_provider_and_model(self):
        """The failure log carries the code, error type, provider name and moderator model."""
        provider = _moderation_provider(error=_api_status_error(500))
        with patch("chibi.services.providers.tools._moderation.logger") as logger_mock:
            with pytest.raises(ModeratorUnavailableError):
                await moderate_command_standardized(
                    provider, cmd="ls", model="moderator-model", caller_model="caller-model"
                )

        logger_mock.log.assert_called_once()
        args = logger_mock.log.call_args.args
        assert args[0] == "MODERATOR"
        entry = args[1]
        assert "mock-moderator" in entry
        assert "moderator-model" in entry
        assert "HTTP 500" in entry
        assert "APIStatusError" in entry

    async def test_moderation_model_falls_back_to_provider_defaults_in_log(self):
        """The logged moderator model falls back to the provider defaults when no override is given."""
        provider = _moderation_provider(error=_api_status_error(500))
        provider.default_moderation_model = None
        provider.default_model = "fallback-model"
        with patch("chibi.services.providers.tools._moderation.logger") as logger_mock:
            with pytest.raises(ModeratorUnavailableError):
                await moderate_command_standardized(provider, cmd="ls", model=None)

        assert "fallback-model" in logger_mock.log.call_args.args[1]


class TestModerateCommandForTool:
    """Tool-facing wrapper raising ToolException with the standardized message."""

    async def test_unavailable_moderator_raises_tool_exception_with_standardized_message(self):
        """The tool wrapper raises ToolException with the standardized message."""
        provider = _moderation_provider(error=_api_status_error(500))
        with pytest.raises(ToolException, match="модератор недоступен") as exc_info:
            await moderate_command_for_tool(provider, cmd="ls", model="moderator-model")

        assert "HTTP 500" in str(exc_info.value)
        assert "APIStatusError" in str(exc_info.value)

    async def test_genuine_decline_is_not_converted(self):
        """A genuine decline passes through the tool wrapper unconverted."""
        answer = ModeratorsAnswer(verdict="declined", status="operation aborted", reason="not allowed")
        provider = _moderation_provider(answer=answer)
        result = await moderate_command_for_tool(provider, cmd="ls", model="moderator-model")

        assert result is answer


class TestToolCallSites:
    """Moderator failures surface from the tools as an error tool response."""

    async def test_cmd_tool_surfaces_standardized_error(self):
        """The terminal command tool surfaces a moderator failure as a standardized error."""
        provider = _moderation_provider(error=_api_status_error(500))
        with patch("chibi.services.providers.tools.cmd.get_moderation_provider", new=AsyncMock(return_value=provider)):
            with pytest.raises(ToolException, match="модератор недоступен") as exc_info:
                await RunCommandInTerminalTool.function(cmd="echo test", user_id=1, cwd="/tmp")

        assert "HTTP 500" in str(exc_info.value)
        provider.moderate_command.assert_awaited_once()

    async def test_cmd_tool_genuine_decline_keeps_declined_path(self):
        """The terminal command tool keeps the existing declined path for genuine declines."""
        answer = ModeratorsAnswer(verdict="declined", status="operation aborted", reason="no")
        provider = _moderation_provider(answer=answer)
        with patch("chibi.services.providers.tools.cmd.get_moderation_provider", new=AsyncMock(return_value=provider)):
            with pytest.raises(ToolException, match="DECLINED"):
                await RunCommandInTerminalTool.function(cmd="echo test", user_id=1, cwd="/tmp")

    async def test_cmd_tool_accepted_command_still_runs(self):
        """An accepted command still runs and returns its result."""
        answer = ModeratorsAnswer(verdict="accepted", status="ok")
        provider = _moderation_provider(answer=answer)
        spawn_mock = AsyncMock(return_value=MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"ok", b""))))
        with (
            patch("chibi.services.providers.tools.cmd.get_moderation_provider", new=AsyncMock(return_value=provider)),
            patch(
                "chibi.services.providers.tools.cmd.get_new_process_group_kwargs",
                return_value={"start_new_session": True},
            ),
            patch("asyncio.create_subprocess_shell", new=spawn_mock),
        ):
            result = await RunCommandInTerminalTool.function(cmd="echo test", user_id=1, cwd="/tmp")

        assert result["return_code"] == 0
        assert result["stdout"] == "ok"
        provider.moderate_command.assert_awaited_once()

    async def test_scheduler_tool_surfaces_standardized_error(self, tmp_path):
        """The scheduler tool surfaces a moderator failure as a standardized error."""
        provider = _moderation_provider(error=_api_status_error(500))
        settings = SimpleNamespace(
            scheduler_tool_enabled=True,
            scheduler_notify_enabled=True,
            scheduler_agent_commands_enabled=True,
        )
        interface = MagicMock()
        interface.user_id = 123
        interface.storage_id = 111
        interface.chat_id = 555
        interface.thread_id = 42
        with (
            patch(
                "chibi.services.providers.tools.scheduler.get_moderation_provider",
                new=AsyncMock(return_value=provider),
            ),
            patch("chibi.services.providers.tools.scheduler.application_settings", settings),
            patch("chibi.services.providers.tools.scheduler.ChibiScheduler"),
        ):
            with pytest.raises(ToolException, match="модератор недоступен") as exc_info:
                await ScheduleTaskTool.function(
                    title="Broken job",
                    schedule={"kind": "once", "at": "2099-01-01T00:00:00+00:00"},
                    action={"type": "command", "command": "echo hi"},
                    user_id=123,
                    caller_model="test-model",
                    interface=interface,
                )

        assert "HTTP 500" in str(exc_info.value)


class TestAgentJobWakePath:
    """An unavailable moderator wakes the agent instead of silently failing the job."""

    async def test_moderator_error_wakes_agent_with_standardized_message(self, tmp_path):
        """A moderator failure wakes the agent with the standardized message in the trigger text."""
        provider = _moderation_provider(error=_api_status_error(500))
        with (
            patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=provider)),
            patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
            patch.object(agent_task_module.SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        ):
            await run_agent_job(
                "agent:1:git-check",
                **VALID_PAYLOAD,
                action={"type": "command", "command": "echo hello-scheduler", "cwd": str(tmp_path)},
            )

        trigger_mock.assert_awaited_once()
        assert trigger_mock.await_args is not None
        trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
        assert _expected_message("500", "APIStatusError") in trigger_text
        assert "echo hello-scheduler" in trigger_text
        send_mock.assert_not_awaited()

    async def test_moderator_error_status_answer_wakes_agent(self, tmp_path):
        """A status='error' answer wakes the agent with the standardized message."""
        answer = ModeratorsAnswer(verdict="declined", status="error", reason="<garbage>")
        provider = _moderation_provider(answer=answer)
        with (
            patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=provider)),
            patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
            patch.object(agent_task_module.SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        ):
            await run_agent_job(
                "agent:1:git-check",
                **VALID_PAYLOAD,
                action={"type": "command", "command": "echo hello-scheduler", "cwd": str(tmp_path)},
            )

        trigger_mock.assert_awaited_once()
        assert trigger_mock.await_args is not None
        trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
        assert _expected_message("n/a", "InvalidModeratorResponse") in trigger_text
        send_mock.assert_not_awaited()

    async def test_moderator_error_keeps_job_alive(self, tmp_path):
        """The moderator failure must not escape as a job failure notification."""
        provider = _moderation_provider(error=_api_status_error(500))
        with (
            patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=provider)),
            patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()),
            patch("chibi.services.jobs.agent_task._notify_failure", new=AsyncMock()) as failure_mock,
        ):
            await run_agent_job(
                "agent:1:git-check",
                **VALID_PAYLOAD,
                action={"type": "command", "command": "echo hello-scheduler", "cwd": str(tmp_path)},
            )

        failure_mock.assert_not_awaited()

    async def test_genuine_decline_still_skips_run_without_wake(self, tmp_path):
        """A genuine decline skips the run without waking the agent."""
        answer = ModeratorsAnswer(verdict="declined", status="operation aborted", reason="no")
        provider = _moderation_provider(answer=answer)
        with (
            patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=provider)),
            patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        ):
            await run_agent_job(
                "agent:1:git-check",
                **VALID_PAYLOAD,
                action={"type": "command", "command": "echo hello-scheduler", "cwd": str(tmp_path)},
            )

        trigger_mock.assert_not_awaited()
