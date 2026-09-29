"""Tests for the agent job dispatcher (design §3.3, §3.4, §5.2, §6.4–6.6)."""

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chibi.services.jobs.agent_task as agent_task_module
from chibi.schemas.app import ModeratorsAnswer
from chibi.services.jobs.agent_task import recover_agent_jobs, run_agent_job
from chibi.services.scheduler import ChibiScheduler
from chibi.services.scheduler_interface import SchedulerInterface
from chibi.storage.local import LocalStorage
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

VALID_PAYLOAD = {
    "user_id": 1,
    "storage_id": 1,
    "chat_id": 555,
    "thread_id": 42,
    "title": "Health heartbeat",
}


@pytest.fixture
def dispatcher_settings():
    """Dispatcher-related application settings with all gates enabled."""
    settings = SimpleNamespace(
        client="telegram",
        scheduler_notify_enabled=True,
        scheduler_agent_commands_enabled=True,
        scheduler_command_timeout_max=900,
        scheduler_failure_notify=True,
    )
    with patch("chibi.services.jobs.agent_task.application_settings", settings):
        yield settings


@pytest.fixture(autouse=True)
def _reset_anti_flood_state():
    """Reset the in-memory anti-flood state around every test."""
    agent_task_module._failure_notify_timestamps.clear()
    yield
    agent_task_module._failure_notify_timestamps.clear()


@pytest.fixture
def local_db(tmp_path: Path) -> LocalStorage:
    """Local storage backend pointed at a temp directory."""
    return LocalStorage(storage_path=str(tmp_path))


@pytest.fixture
def database_settings(tmp_path: Path, local_db: LocalStorage):
    """Point the DatabaseCache factory at the test's own local backend.

    The module-level ``_db_provider`` singleton is replaced with a stub serving
    the test's ``local_db`` instance, so injected-database calls (orphan
    cleanup) never resolve to a stale global cache that a previous test may
    have bound to a different storage backend or event loop.
    """
    settings = SimpleNamespace(storage_backend="local", local_data_path=str(tmp_path))
    with (
        patch("chibi.storage.database.application_settings", settings),
        patch("chibi.storage.database._db_provider", local_db_provider(local_db)),
    ):
        yield settings


def local_db_provider(db: LocalStorage) -> SimpleNamespace:
    """Build a fake database provider serving one LocalStorage instance.

    Args:
        db: The storage instance the injected database calls resolve to.

    Returns:
        Provider namespace compatible with ``chibi.storage.database._db_provider``.
    """
    return SimpleNamespace(get_database=AsyncMock(return_value=db))


async def test_self_action_routes_to_handle_scheduler_trigger(dispatcher_settings) -> None:
    """A `self` action wakes the agent via handle_scheduler_trigger with the job context."""
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job("agent:1:heartbeat", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "Check up"})

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    kwargs = trigger_mock.await_args.kwargs
    assert kwargs["trigger_text"] == "Check up"
    assert kwargs["job_id"] == "agent:1:heartbeat"
    interface = kwargs["interface"]
    assert isinstance(interface, SchedulerInterface)
    assert interface.user_id == 1
    assert interface.storage_id == 1
    assert interface.chat_id == 555
    assert interface.thread_id == 42
    send_mock.assert_not_awaited()


async def test_notify_action_delivers_message(dispatcher_settings) -> None:
    """A `notify` action delivers the static message through the job interface."""
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:reminder", **VALID_PAYLOAD, action={"type": "notify", "message": "Time to drink water"}
        )

    send_mock.assert_awaited_once_with(message="Time to drink water")
    trigger_mock.assert_not_awaited()


async def test_notify_action_gate_disabled_skips_delivery(dispatcher_settings) -> None:
    """A `notify` action is skipped (job kept) when scheduler_notify_enabled is off."""
    dispatcher_settings.scheduler_notify_enabled = False
    with (
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.jobs.agent_task._notify_failure", new=AsyncMock()) as failure_mock,
    ):
        await run_agent_job("agent:1:reminder", **VALID_PAYLOAD, action={"type": "notify", "message": "Hi"})

    send_mock.assert_not_awaited()
    failure_mock.assert_not_awaited()


async def test_invalid_payload_rejected_without_execution(dispatcher_settings) -> None:
    """A payload missing mandatory fields is rejected; nothing executes, no exception escapes."""
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job("agent:1:broken", user_id=1, action={"type": "self", "trigger_text": "x"})

    trigger_mock.assert_not_awaited()
    send_mock.assert_not_awaited()


async def test_command_action_nonempty_output_wakes_agent(dispatcher_settings, tmp_path: Path) -> None:
    """A `command` action with non-empty stdout wakes the agent with the full result context."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:git-check",
            **{**VALID_PAYLOAD, "user_id": 1},
            action={"type": "command", "command": "echo hello-scheduler", "cwd": str(tmp_path)},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "hello-scheduler" in trigger_text
    assert "exit code: 0" in trigger_text.lower()
    assert "echo hello-scheduler" in trigger_text
    assert isinstance(trigger_mock.await_args.kwargs["interface"], SchedulerInterface)
    send_mock.assert_not_awaited()


async def test_command_action_empty_output_still_wakes_agent(dispatcher_settings) -> None:
    """An empty stdout still wakes the agent, with an explicit '(empty)' output context."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:silent",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "true"},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "(empty)" in trigger_text
    assert "exit code: 0" in trigger_text.lower()
    send_mock.assert_not_awaited()


async def test_command_action_wake_carries_stdout_and_stderr(dispatcher_settings) -> None:
    """The wake trigger text contains stdout and stderr of the finished command."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    process = MagicMock()
    process.returncode = 0
    process.pid = 4242
    process.communicate = AsyncMock(return_value=(b"hello-scheduler\n", b"warning line\n"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:report",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "echo hello-scheduler"},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "hello-scheduler" in trigger_text
    assert "warning line" in trigger_text
    send_mock.assert_not_awaited()


async def test_command_action_nonzero_exit_wakes_agent_without_user_message(dispatcher_settings) -> None:
    """A non-zero exit code is a command failure, not a wake failure: the agent is woken, the user is not."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    process = MagicMock()
    process.returncode = 3
    process.pid = 4242
    process.communicate = AsyncMock(return_value=(b"", b"git: fatal: not a repository\n"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.jobs.agent_task._notify_failure", new=AsyncMock()) as failure_mock,
    ):
        await run_agent_job(
            "agent:1:git-fail",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "git status"},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "exit code: 3" in trigger_text.lower()
    assert "not a repository" in trigger_text
    failure_mock.assert_not_awaited()
    send_mock.assert_not_awaited()


async def test_command_action_spawn_failure_wakes_agent_with_error_context(dispatcher_settings) -> None:
    """A spawn failure wakes the agent with the error context instead of notifying the user directly."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch(
            "chibi.services.jobs.agent_task.asyncio.create_subprocess_shell",
            new=AsyncMock(side_effect=OSError("spawn failed")),
        ),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.jobs.agent_task._notify_failure", new=AsyncMock()) as failure_mock,
    ):
        await run_agent_job(
            "agent:1:broken",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "echo hi"},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "could not be started" in trigger_text
    assert "OSError" in trigger_text
    failure_mock.assert_not_awaited()
    send_mock.assert_not_awaited()


async def test_command_wake_failure_falls_back_to_user_notification(dispatcher_settings) -> None:
    """A failed wake (LLM error) falls back to a single anti-flooded direct user notification."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    process = MagicMock()
    process.returncode = 0
    process.pid = 4242
    process.communicate = AsyncMock(return_value=(b"hello-scheduler\n", b""))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        trigger_mock.side_effect = RuntimeError("provider exploded")
        await run_agent_job(
            "agent:1:wake-fail",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "echo hi"},
        )
        await run_agent_job(
            "agent:1:wake-fail",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "echo hi"},
        )

    assert send_mock.await_count == 1
    assert send_mock.await_args is not None
    message = send_mock.await_args.kwargs["message"]
    assert "agent:1:wake-fail" in message
    assert "wake failed" in message


async def test_command_action_timeout_kills_process_group(dispatcher_settings) -> None:
    """A command exceeding its per-job timeout is killed via its process group and reported to the agent."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    started = time.monotonic()
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:slow",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "trap '' TERM; sleep 30", "timeout_seconds": 1},
        )
    elapsed = time.monotonic() - started

    assert elapsed < 20
    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    trigger_text = trigger_mock.await_args.kwargs["trigger_text"]
    assert "timed out" in trigger_text or "killed" in trigger_text
    send_mock.assert_not_awaited()


async def test_command_action_timeout_clamped_to_configured_maximum(dispatcher_settings) -> None:
    """A per-job timeout above scheduler_command_timeout_max is clamped down."""
    dispatcher_settings.scheduler_command_timeout_max = 2
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="accepted", status="ok"))
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        await run_agent_job(
            "agent:1:clamp",
            **VALID_PAYLOAD,
            action={"type": "command", "command": "sleep 30", "timeout_seconds": 5000},
        )

    trigger_mock.assert_awaited_once()
    assert trigger_mock.await_args is not None
    assert "after 2s" in trigger_mock.await_args.kwargs["trigger_text"]
    send_mock.assert_not_awaited()


async def test_command_action_moderation_declined_skips_run(dispatcher_settings) -> None:
    """A declined re-moderation skips the run; the job is not removed, run nor woken."""
    moderation = MagicMock()
    moderation.moderate_command = AsyncMock(
        return_value=ModeratorsAnswer(verdict="declined", reason="dangerous", status="ok")
    )
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock(return_value=moderation)),
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock()) as subprocess_mock,
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch("chibi.services.jobs.agent_task._notify_failure", new=AsyncMock()) as failure_mock,
    ):
        await run_agent_job("agent:1:moderated", **VALID_PAYLOAD, action={"type": "command", "command": "rm -rf /"})

    subprocess_mock.assert_not_awaited()
    send_mock.assert_not_awaited()
    trigger_mock.assert_not_awaited()
    failure_mock.assert_not_awaited()


async def test_command_action_gate_disabled_skips_execution(dispatcher_settings) -> None:
    """A `command` action is skipped (job kept) when scheduler_agent_commands_enabled is off."""
    dispatcher_settings.scheduler_agent_commands_enabled = False
    with (
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock()) as moderation_mock,
        patch("chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock()) as subprocess_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
    ):
        await run_agent_job("agent:1:cmd", **VALID_PAYLOAD, action={"type": "command", "command": "echo hi"})

    moderation_mock.assert_not_awaited()
    subprocess_mock.assert_not_awaited()
    send_mock.assert_not_awaited()
    trigger_mock.assert_not_awaited()


@pytest.mark.usefixtures("database_settings")
async def test_command_action_passes_gate_with_production_defaults() -> None:
    """With the shipped configuration (D5: commands enabled by default) a `command` job executes."""
    from chibi.config.app import ApplicationSettings

    process = MagicMock()
    process.returncode = 0
    process.pid = 4242
    process.communicate = AsyncMock(return_value=(b"hello-scheduler\n", b""))
    with (
        patch("chibi.services.jobs.agent_task.application_settings", ApplicationSettings()),
        patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock()),
        patch(
            "chibi.services.jobs.agent_task.asyncio.create_subprocess_shell", new=AsyncMock(return_value=process)
        ) as subprocess_mock,
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
    ):
        await run_agent_job("agent:1:cmd", **VALID_PAYLOAD, action={"type": "command", "command": "echo hi"})

    subprocess_mock.assert_awaited_once()
    trigger_mock.assert_awaited_once()


async def test_failure_notify_sent_once_within_cooldown(dispatcher_settings) -> None:
    """A failing job triggers a single anti-flooded user notification per hour."""
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        trigger_mock.side_effect = RuntimeError("provider exploded")
        await run_agent_job("agent:1:failing", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "tick"})
        await run_agent_job("agent:1:failing", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "tick"})

    assert send_mock.await_count == 1
    assert send_mock.await_args is not None
    message = send_mock.await_args.kwargs["message"]
    assert "agent:1:failing" in message
    assert "failed" in message


async def test_failure_notify_repeats_after_cooldown_expiry(dispatcher_settings) -> None:
    """After the one-hour cooldown a new failure notifies the user again."""
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        trigger_mock.side_effect = RuntimeError("boom")
        await run_agent_job("agent:1:failing", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "tick"})
        agent_task_module._failure_notify_timestamps["agent:1:failing"] = time.monotonic() - 3601
        await run_agent_job("agent:1:failing", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "tick"})

    assert send_mock.await_count == 2


async def test_failure_notify_disabled_sends_nothing(dispatcher_settings) -> None:
    """With scheduler_failure_notify=False failures are logged but not delivered."""
    dispatcher_settings.scheduler_failure_notify = False
    with (
        patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
    ):
        trigger_mock.side_effect = RuntimeError("boom")
        await run_agent_job("agent:1:failing", **VALID_PAYLOAD, action={"type": "self", "trigger_text": "tick"})

    send_mock.assert_not_awaited()


@pytest.fixture()
def scheduler(tmp_path: Path):
    """Provide a fresh ChibiScheduler backed by a temporary SQLite job store."""
    settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)
    SingletonMeta._instances.pop(ChibiScheduler, None)
    with patch("chibi.services.scheduler.application_settings", settings):
        yield ChibiScheduler()
    SingletonMeta._instances.pop(ChibiScheduler, None)


async def _schedule_agent_job(scheduler: ChibiScheduler, job_id: str, user_id: int, kwargs: dict) -> None:
    """Register an agent job pointing at the stable dispatcher function."""
    scheduler.add_job(
        run_agent_job,
        trigger="interval",
        seconds=3600,
        kwargs={"job_id": job_id, "user_id": user_id, **kwargs},
        id=job_id,
        replace_existing=True,
    )


@pytest.mark.usefixtures("database_settings")
async def test_recovery_removes_broken_and_unknown_jobs_keeps_valid(scheduler, local_db: LocalStorage) -> None:
    """Startup recovery drops invalid-payload and foreign-function agent jobs, keeps valid ones."""
    await local_db.get_or_create_user(user_id=1)
    await _schedule_agent_job(
        scheduler, "agent:1:valid", 1, dict(VALID_PAYLOAD, action={"type": "notify", "message": "ok"})
    )
    await _schedule_agent_job(scheduler, "agent:1:broken", 1, {"user_id": 1, "title": "incomplete"})
    await _schedule_agent_job(
        scheduler, "agent:1:foreign", 1, dict(VALID_PAYLOAD, action={"type": "notify", "message": "ok"})
    )
    foreign_job = next(job for job in scheduler.get_jobs() if str(job.id) == "agent:1:foreign")
    foreign_job.func = async_foreign_function

    await recover_agent_jobs(scheduler)

    remaining = {job.job_id for job in scheduler.list_jobs(prefix="agent:")}
    assert remaining == {"agent:1:valid"}


@pytest.mark.usefixtures("database_settings")
async def test_recovery_removes_jobs_of_deleted_users(scheduler, local_db: LocalStorage) -> None:
    """Orphan cleanup removes agent jobs of users missing from the storage."""
    await local_db.get_or_create_user(user_id=1)
    await _schedule_agent_job(
        scheduler, "agent:1:alive", 1, dict(VALID_PAYLOAD, action={"type": "notify", "message": "ok"})
    )
    await _schedule_agent_job(
        scheduler, "agent:999:orphan", 999, dict(VALID_PAYLOAD, user_id=999, action={"type": "notify", "message": "x"})
    )

    await recover_agent_jobs(scheduler)

    remaining = {job.job_id for job in scheduler.list_jobs(prefix="agent:")}
    assert remaining == {"agent:1:alive"}


@pytest.mark.usefixtures("database_settings")
async def test_recovery_tolerates_empty_store(scheduler) -> None:
    """Recovery on a job store without agent jobs is a no-op."""
    scheduler.add_job(async_foreign_function, trigger="interval", seconds=3600, id="system:legacy")

    await recover_agent_jobs(scheduler)

    assert {job.job_id for job in scheduler.list_jobs()} == {"system:legacy"}


async def test_recovery_never_raises_on_unreadable_store(scheduler) -> None:
    """A job store read failure is logged, not raised, so startup continues."""
    with patch.object(scheduler, "get_jobs", side_effect=RuntimeError("jobstore corrupted")):
        await recover_agent_jobs(scheduler)


async def async_foreign_function() -> None:
    """Async placeholder standing for a job function from a foreign module."""
