"""Tests for the agent-facing scheduler tools (design §5, §6)."""

import importlib
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chibi.services.providers.tools.scheduler as scheduler_tools_module
from chibi.config import gpt_settings
from chibi.constants import IDE_STORAGE_ID
from chibi.schemas.app import ModeratorsAnswer
from chibi.services.jobs.agent_task import run_agent_job
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.scheduler import (
    DeleteScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)
from chibi.services.scheduler import ChibiScheduler
from chibi.utils.app import SingletonMeta

TOOL_MODULE = "chibi.services.providers.tools.scheduler"
SCHEDULER_MODULE = "chibi.services.scheduler"
TOOL_NAMES = ["schedule_task", "list_scheduled_tasks", "delete_scheduled_task"]


def _foreign_job_kwargs(job_id: str) -> dict:
    """Build a valid payload kwargs set for a job of a foreign user."""
    return {
        "job_id": job_id,
        "user_id": 999,
        "storage_id": 999,
        "thread_id": 0,
        "chat_id": 999,
        "title": "Foreign job",
        "action": {"type": "self", "trigger_text": "foreign"},
    }


@pytest.fixture()
def scheduler(tmp_path):
    """Provide a fresh ChibiScheduler backed by a temporary SQLite job store."""
    settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)
    SingletonMeta._instances.pop(ChibiScheduler, None)
    with patch(f"{SCHEDULER_MODULE}.application_settings", settings):
        yield ChibiScheduler()
    SingletonMeta._instances.pop(ChibiScheduler, None)


@pytest.fixture()
def tool_settings():
    """Tool-gate settings with every scheduler gate enabled."""
    settings = SimpleNamespace(
        scheduler_tool_enabled=True,
        scheduler_notify_enabled=True,
        scheduler_agent_commands_enabled=True,
    )
    with patch(f"{TOOL_MODULE}.application_settings", settings):
        yield settings


@pytest.fixture()
def moderation():
    """Mock the moderation provider used for command pre-moderation."""
    provider = MagicMock()
    provider.name = "mock-moderator"
    provider.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    with patch(f"{TOOL_MODULE}.get_moderation_provider", AsyncMock(return_value=provider)):
        yield provider


@pytest.fixture()
def interface():
    """A user interface stub carrying a full Telegram session context."""
    mock = MagicMock()
    mock.user_id = 123
    mock.storage_id = 111
    mock.chat_id = 555
    mock.thread_id = 42
    return mock


def _user_kwargs(interface) -> dict:
    """Build the additional-options payload of a tool invocation."""
    return {"user_id": 123, "caller_model": "test-model", "interface": interface}


class TestScheduleTaskTool:
    """Tests for ScheduleTaskTool.function."""

    @pytest.mark.asyncio
    async def test_interval_job_created_with_fixed_context(self, scheduler, tool_settings, interface):
        result = await ScheduleTaskTool.function(
            title="Health heartbeat",
            schedule={"kind": "interval", "every_seconds": 1800},
            action={"type": "self", "trigger_text": "Check the state and report"},
            **_user_kwargs(interface),
        )

        assert result["status"] == "ok"
        assert result["job_id"] == "agent:123:health-heartbeat"
        assert result["action_type"] == "self"
        job = scheduler.get_jobs()[0]
        assert job.func is run_agent_job
        payload_kwargs = job.kwargs
        assert payload_kwargs["user_id"] == 123
        assert payload_kwargs["storage_id"] == 111
        assert payload_kwargs["thread_id"] == 42
        assert payload_kwargs["chat_id"] == 555
        assert payload_kwargs["title"] == "Health heartbeat"
        assert payload_kwargs["action"] == {"type": "self", "trigger_text": "Check the state and report"}

    @pytest.mark.asyncio
    async def test_ide_session_creates_job_with_negative_identity(self, scheduler, tool_settings):
        """A tool call in an IDE session pins the negative IDE identity and the session thread."""
        ide_interface = MagicMock()
        ide_interface.user_id = IDE_STORAGE_ID
        ide_interface.storage_id = IDE_STORAGE_ID
        ide_interface.chat_id = IDE_STORAGE_ID
        ide_interface.thread_id = 7

        run_at = datetime.now().astimezone() + timedelta(hours=1)
        result = await ScheduleTaskTool.function(
            title="IDE reminder",
            schedule={"kind": "once", "at": run_at.isoformat()},
            action={"type": "self", "trigger_text": "Summarize the open PR"},
            user_id=IDE_STORAGE_ID,
            caller_model="test-model",
            interface=ide_interface,
        )

        assert result["status"] == "ok"
        assert result["job_id"].startswith(f"agent:{IDE_STORAGE_ID}:")
        job = scheduler.get_jobs()[0]
        payload_kwargs = job.kwargs
        assert payload_kwargs["user_id"] == IDE_STORAGE_ID
        assert payload_kwargs["storage_id"] == IDE_STORAGE_ID
        assert payload_kwargs["chat_id"] == IDE_STORAGE_ID
        assert payload_kwargs["thread_id"] == 7

    @pytest.mark.asyncio
    async def test_cron_fields_build_five_field_expression(self, scheduler, tool_settings, interface):
        await ScheduleTaskTool.function(
            title="Morning report",
            schedule={"kind": "cron", "minute": "0", "hour": "9", "day_of_week": "1-5"},
            action={"type": "self", "trigger_text": "Report time"},
            **_user_kwargs(interface),
        )

        trigger = str(scheduler.get_jobs()[0].trigger)
        assert "minute='0'" in trigger
        assert "hour='9'" in trigger
        assert "day_of_week='1-5'" in trigger
        assert "month='*'" in trigger
        assert "day='*'" in trigger

    @pytest.mark.asyncio
    async def test_invalid_cron_field_rejected(self, scheduler, tool_settings, interface):
        with pytest.raises(ToolException):
            await ScheduleTaskTool.function(
                title="Broken cron",
                schedule={"kind": "cron", "minute": "99"},
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )

        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_once_job_created_from_iso8601(self, scheduler, tool_settings, interface):
        run_at = datetime.now().astimezone() + timedelta(hours=1)
        await ScheduleTaskTool.function(
            title="One-time reminder",
            schedule={"kind": "once", "at": run_at.isoformat()},
            action={"type": "self", "trigger_text": "Reminder fired"},
            **_user_kwargs(interface),
        )

        job = scheduler.get_jobs()[0]
        assert job.trigger.run_date == run_at

    @pytest.mark.asyncio
    async def test_once_in_past_rejected(self, scheduler, tool_settings, interface):
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Late reminder",
                schedule={"kind": "once", "at": (datetime.now() - timedelta(hours=1)).isoformat()},
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )

        assert "future" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("every_seconds", [None, 0, -5, True])
    async def test_interval_requires_positive_every_seconds(self, scheduler, tool_settings, interface, every_seconds):
        schedule = {"kind": "interval"}
        if every_seconds is not None:
            schedule["every_seconds"] = every_seconds
        with pytest.raises(ToolException):
            await ScheduleTaskTool.function(
                title="Broken interval",
                schedule=schedule,
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )

        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_slug_job_id_generated_from_title(self, scheduler, tool_settings, interface):
        result = await ScheduleTaskTool.function(
            title="Daily Report!",
            schedule={"kind": "interval", "every_seconds": 60},
            action={"type": "self", "trigger_text": "x"},
            **_user_kwargs(interface),
        )

        assert result["job_id"] == "agent:123:daily-report"

    @pytest.mark.asyncio
    async def test_explicit_job_id_suffix_is_used(self, scheduler, tool_settings, interface):
        result = await ScheduleTaskTool.function(
            title="Some fancy title",
            job_id="my_task-1",
            schedule={"kind": "interval", "every_seconds": 60},
            action={"type": "self", "trigger_text": "x"},
            **_user_kwargs(interface),
        )

        assert result["job_id"] == "agent:123:my_task-1"

    @pytest.mark.asyncio
    async def test_full_job_id_with_own_prefix_accepted(self, scheduler, tool_settings, interface):
        result = await ScheduleTaskTool.function(
            title="Custom id task",
            job_id="agent:123:custom",
            schedule={"kind": "interval", "every_seconds": 60},
            action={"type": "self", "trigger_text": "x"},
            **_user_kwargs(interface),
        )

        assert result["job_id"] == "agent:123:custom"
        assert scheduler.list_jobs(prefix="agent:123:custom")

    @pytest.mark.asyncio
    async def test_conflict_lists_existing_jobs(self, scheduler, tool_settings, interface):
        await ScheduleTaskTool.function(
            title="Water reminder",
            schedule={"kind": "interval", "every_seconds": 60},
            action={"type": "self", "trigger_text": "x"},
            **_user_kwargs(interface),
        )
        existing_ids = [info.job_id for info in scheduler.list_jobs(prefix="agent:123:")]

        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Water reminder",
                schedule={"kind": "interval", "every_seconds": 120},
                action={"type": "self", "trigger_text": "y"},
                **_user_kwargs(interface),
            )

        message = str(exc_info.value)
        assert "already taken" in message
        for job_id in existing_ids:
            assert job_id in message
        assert "replace=true" in message
        assert len(scheduler.list_jobs(prefix="agent:123:")) == 1

    @pytest.mark.asyncio
    async def test_replace_overwrites_existing_job(self, scheduler, tool_settings, interface):
        scheduler.start()
        try:
            await ScheduleTaskTool.function(
                title="Water reminder",
                job_id="water-reminder",
                schedule={"kind": "interval", "every_seconds": 60},
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )
            result = await ScheduleTaskTool.function(
                title="Water reminder v2",
                job_id="water-reminder",
                schedule={"kind": "interval", "every_seconds": 120},
                action={"type": "self", "trigger_text": "y"},
                replace=True,
                **_user_kwargs(interface),
            )
        finally:
            scheduler.shutdown(wait=False)

        jobs = scheduler.list_jobs(prefix="agent:123:")
        assert len(jobs) == 1
        assert jobs[0].job_id == result["job_id"] == "agent:123:water-reminder"
        assert scheduler.get_jobs()[0].kwargs["title"] == "Water reminder v2"

    @pytest.mark.asyncio
    async def test_unknown_schedule_kind_rejected(self, scheduler, tool_settings, interface):
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Broken kind",
                schedule={"kind": "yearly"},
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )

        assert "kind" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_invalid_action_type_rejected(self, scheduler, tool_settings, interface):
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Broken action",
                schedule={"kind": "interval", "every_seconds": 60},
                action={"type": "bogus"},
                **_user_kwargs(interface),
            )

        assert "Invalid task parameters" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_notify_gate_disabled(self, scheduler, tool_settings, interface):
        tool_settings.scheduler_notify_enabled = False
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Water reminder",
                schedule={"kind": "interval", "every_seconds": 7200},
                action={"type": "notify", "message": "Time to drink water"},
                **_user_kwargs(interface),
            )

        assert "notify" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_command_gate_disabled(self, scheduler, tool_settings, interface, moderation):
        tool_settings.scheduler_agent_commands_enabled = False
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Git check",
                schedule={"kind": "interval", "every_seconds": 1800},
                action={"type": "command", "command": "git status --porcelain"},
                **_user_kwargs(interface),
            )

        assert "scheduler_agent_commands_enabled" in str(exc_info.value)
        moderation.moderate_command.assert_not_awaited()
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_command_premoderation_declined_blocks_creation(
        self, scheduler, tool_settings, interface, moderation
    ):
        moderation.moderate_command = AsyncMock(
            return_value=ModeratorsAnswer(verdict="declined", reason="dangerous", status="ok")
        )
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Risky task",
                schedule={"kind": "interval", "every_seconds": 60},
                action={"type": "command", "command": "rm -rf /"},
                **_user_kwargs(interface),
            )

        assert "declined" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_command_premoderation_accepted_creates_job(self, scheduler, tool_settings, interface, moderation):
        result = await ScheduleTaskTool.function(
            title="Git check",
            schedule={"kind": "interval", "every_seconds": 1800},
            action={"type": "command", "command": "git status --porcelain", "cwd": "/tmp"},
            **_user_kwargs(interface),
        )

        assert result["status"] == "ok"
        assert result["action_type"] == "command"
        moderation.moderate_command.assert_awaited_once()
        assert moderation.moderate_command.await_args.kwargs["cmd"] == "git status --porcelain"
        assert moderation.moderate_command.await_args.kwargs["model"] == gpt_settings.moderation_model
        job = scheduler.get_jobs()[0]
        assert job.func is run_agent_job
        assert job.kwargs["action"]["command"] == "git status --porcelain"
        assert job.kwargs["action"]["timeout_seconds"] == 60

    @pytest.mark.asyncio
    @pytest.mark.parametrize("foreign_job_id", ["agent:999:steal", "system:hack"])
    async def test_isolation_foreign_job_id_rejected(self, scheduler, tool_settings, interface, foreign_job_id):
        with pytest.raises(ToolException) as exc_info:
            await ScheduleTaskTool.function(
                title="Hostile task",
                job_id=foreign_job_id,
                schedule={"kind": "interval", "every_seconds": 60},
                action={"type": "self", "trigger_text": "x"},
                **_user_kwargs(interface),
            )

        assert "Access denied" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:123:") == []
        assert scheduler.list_jobs(prefix="agent:999:") == []


class TestListScheduledTasksTool:
    """Tests for ListScheduledTasksTool.function."""

    @pytest.mark.asyncio
    async def test_lists_only_own_jobs_with_details(self, scheduler, tool_settings, interface):
        await ScheduleTaskTool.function(
            title="Health heartbeat",
            schedule={"kind": "interval", "every_seconds": 1800},
            action={"type": "self", "trigger_text": "Check the state"},
            **_user_kwargs(interface),
        )
        scheduler.schedule_interval_job(
            job_id="agent:999:foreign",
            func=run_agent_job,
            interval_seconds=60,
            kwargs=_foreign_job_kwargs("agent:999:foreign"),
        )

        result = await ListScheduledTasksTool.function(**_user_kwargs(interface))

        assert result["status"] == "ok"
        assert result["count"] == 1
        job_entry = result["jobs"][0]
        assert job_entry["job_id"] == "agent:123:health-heartbeat"
        assert job_entry["title"] == "Health heartbeat"
        assert job_entry["action_type"] == "self"
        assert "interval" in job_entry["schedule"]
        assert "next_run_time" in job_entry

    @pytest.mark.asyncio
    async def test_empty_list_has_message(self, scheduler, tool_settings, interface):
        result = await ListScheduledTasksTool.function(**_user_kwargs(interface))

        assert result["status"] == "ok"
        assert result["count"] == 0
        assert result["jobs"] == []
        assert "No scheduled tasks" in result["message"]


class TestDeleteScheduledTaskTool:
    """Tests for DeleteScheduledTaskTool.function."""

    @pytest.mark.asyncio
    async def test_deletes_own_job(self, scheduler, tool_settings, interface):
        result = await ScheduleTaskTool.function(
            title="Water reminder",
            schedule={"kind": "interval", "every_seconds": 60},
            action={"type": "self", "trigger_text": "x"},
            **_user_kwargs(interface),
        )
        delete_result = await DeleteScheduledTaskTool.function(job_id=result["job_id"], **_user_kwargs(interface))

        assert delete_result["status"] == "ok"
        assert scheduler.list_jobs(prefix="agent:123:") == []

    @pytest.mark.asyncio
    async def test_unknown_job_id_rejected(self, scheduler, tool_settings, interface):
        with pytest.raises(ToolException) as exc_info:
            await DeleteScheduledTaskTool.function(job_id="agent:123:missing", **_user_kwargs(interface))

        assert "No scheduled task" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("foreign_job_id", ["agent:999:foreign", "system:retention_cleanup"])
    async def test_isolation_foreign_prefix_rejected(self, scheduler, tool_settings, interface, foreign_job_id):
        scheduler.schedule_interval_job(
            job_id="agent:999:foreign",
            func=run_agent_job,
            interval_seconds=60,
            kwargs=_foreign_job_kwargs("agent:999:foreign"),
        )

        with pytest.raises(ToolException) as exc_info:
            await DeleteScheduledTaskTool.function(job_id=foreign_job_id, **_user_kwargs(interface))

        assert "Access denied" in str(exc_info.value)
        assert scheduler.list_jobs(prefix="agent:999:")


class TestRegisterGate:
    """Tests for the runner-aware tool registration gate (design §1.5, §6.4)."""

    @pytest.mark.parametrize("client", ["telegram", "tui", "vscode", "pycharm", "neovim"])
    def test_gate_registers_for_supported_runners(self, client):
        """Scheduler tools register for the telegram bot and every stdio client."""
        with patch("chibi.config.application_settings", SimpleNamespace(scheduler_tool_enabled=True, client=client)):
            reloaded = importlib.reload(scheduler_tools_module)
            assert reloaded.ScheduleTaskTool.register is True
            assert reloaded.ListScheduledTasksTool.register is True
            assert reloaded.DeleteScheduledTaskTool.register is True
            assert RegisteredChibiTools.tools_map["schedule_task"] is reloaded.ScheduleTaskTool
            assert RegisteredChibiTools.tools_map["list_scheduled_tasks"] is reloaded.ListScheduledTasksTool
            assert RegisteredChibiTools.tools_map["delete_scheduled_task"] is reloaded.DeleteScheduledTaskTool

        RegisteredChibiTools.deregister_tools(TOOL_NAMES)
        for tool_name in TOOL_NAMES:
            assert tool_name not in RegisteredChibiTools.tools_map

    def test_gate_excludes_terminal_client(self):
        """The terminal REPL (client='terminal') never registers the scheduler tools."""
        with patch(
            "chibi.config.application_settings", SimpleNamespace(scheduler_tool_enabled=True, client="terminal")
        ):
            reloaded = importlib.reload(scheduler_tools_module)
            assert reloaded.ScheduleTaskTool.register is False
            assert reloaded.ListScheduledTasksTool.register is False
            assert reloaded.DeleteScheduledTaskTool.register is False
            for tool_name in TOOL_NAMES:
                assert tool_name not in RegisteredChibiTools.tools_map

    def test_gate_requires_setting(self):
        """Scheduler tools stay unregistered when scheduler_tool_enabled is False."""
        with patch(
            "chibi.config.application_settings", SimpleNamespace(scheduler_tool_enabled=False, client="telegram")
        ):
            reloaded = importlib.reload(scheduler_tools_module)
            assert reloaded.ScheduleTaskTool.register is False
            assert reloaded.ListScheduledTasksTool.register is False
            assert reloaded.DeleteScheduledTaskTool.register is False
            for tool_name in TOOL_NAMES:
                assert tool_name not in RegisteredChibiTools.tools_map

    def test_predicate_rejects_unknown_client(self):
        """The predicate itself rejects any client value outside the supported set."""
        with patch(
            "chibi.services.providers.tools.scheduler.application_settings",
            SimpleNamespace(scheduler_tool_enabled=True, client="fsck"),
        ):
            assert scheduler_tools_module._scheduler_tools_register() is False
        with patch(
            "chibi.services.providers.tools.scheduler.application_settings",
            SimpleNamespace(scheduler_tool_enabled=True, client="vscode"),
        ):
            assert scheduler_tools_module._scheduler_tools_register() is True
