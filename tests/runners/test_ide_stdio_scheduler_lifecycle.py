"""Tests for the stdio runner scheduler lifecycle (start / shutdown / gate)."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.base import STATE_STOPPED

import chibi.config  # noqa: F401
from chibi.config import application_settings
from chibi.constants import IDE_STORAGE_ID
from chibi.runners.ide_transport import IDEStdioRunner
from chibi.services.scheduler import (
    RETENTION_CLEANUP_JOB_ID,
    STDIO_SCHEDULER_DB_FILENAME,
    StdioScheduler,
)
from chibi.services.task_manager import task_manager
from chibi.utils.app import SingletonMeta


@pytest.fixture(autouse=True)
async def reap_leaked_background_tasks():
    """Cancel and await any task leaked into the global task manager.

    A fire-and-forget request task surviving the test would run outside the
    test's patches and could touch real services (storage, providers).
    """
    yield
    leaked = [task for tasks in task_manager._tasks.values() for task in tasks if not task.done()]
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)


@pytest.fixture()
def sqlite_scheduler_settings(tmp_path):
    """Return scheduler settings patched onto the scheduler module."""
    return SimpleNamespace(
        redis=None,
        local_data_path=str(tmp_path),
        scheduler_misfire_grace_time=3600,
        chroma_history_retention_days=7,
    )


@pytest.fixture()
def fresh_stdio_scheduler_singleton():
    """Ensure a clean StdioScheduler singleton around the test."""
    SingletonMeta._instances.pop(StdioScheduler, None)
    yield
    SingletonMeta._instances.pop(StdioScheduler, None)


@pytest.fixture()
def scheduler_gate_enabled():
    """Force ``scheduler_tool_enabled`` on for the duration of the test."""
    original = application_settings.scheduler_tool_enabled
    application_settings.scheduler_tool_enabled = True
    yield
    application_settings.scheduler_tool_enabled = original


@pytest.fixture()
def scheduler_gate_disabled():
    """Force ``scheduler_tool_enabled`` off for the duration of the test."""
    original = application_settings.scheduler_tool_enabled
    application_settings.scheduler_tool_enabled = False
    yield
    application_settings.scheduler_tool_enabled = original


def make_runner() -> IDEStdioRunner:
    """Build a stdio runner whose event loop ends immediately (stdin EOF)."""
    runner = IDEStdioRunner()
    # ``_read_line`` is a regular method, so mypy would flag a direct Mock
    # assignment as method-assign; bypass the typed attribute access.
    object.__setattr__(runner, "_read_line", AsyncMock(return_value=""))
    return runner


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_enabled")
@pytest.mark.asyncio
async def test_run_starts_stdio_scheduler_with_retention_and_recovery(sqlite_scheduler_settings, tmp_path) -> None:
    """With the gate on, run() starts the stdio scheduler, registers retention cleanup and runs recovery."""
    runner = make_runner()
    captured: dict = {}

    async def fake_recover(scheduler) -> None:
        captured["state"] = scheduler.state
        captured["job_ids"] = [job.id for job in scheduler.get_jobs()]

    with (
        patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
        patch("chibi.services.jobs.archive.perform_retention_cleanup", new_callable=MagicMock),
        patch("chibi.runners.ide_transport.recover_agent_jobs", side_effect=fake_recover) as recover_mock,
        patch("chibi.runners.ide_transport.task_manager") as task_manager_mock,
    ):
        task_manager_mock.shutdown = AsyncMock()
        await runner.run()

        recover_mock.assert_awaited_once()
        assert captured["state"] != STATE_STOPPED, "Scheduler must be running during recovery"
        assert RETENTION_CLEANUP_JOB_ID in captured["job_ids"]

        scheduler = StdioScheduler()
        assert Path(sqlite_scheduler_settings.local_data_path, STDIO_SCHEDULER_DB_FILENAME).exists()
        assert runner._scheduler is scheduler
        assert scheduler.state == STATE_STOPPED, "run() must stop the scheduler on exit"

    task_manager_mock.shutdown.assert_awaited_once()


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_enabled")
@pytest.mark.asyncio
async def test_shutdown_stops_scheduler_before_task_manager(sqlite_scheduler_settings) -> None:
    """The scheduler must be fully stopped by the time the task manager shuts down."""
    runner = make_runner()
    state_at_task_manager_shutdown: list[int] = []

    async def record_state() -> None:
        scheduler = StdioScheduler()
        state_at_task_manager_shutdown.append(scheduler.state)

    with (
        patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
        patch("chibi.services.jobs.archive.perform_retention_cleanup", new_callable=MagicMock),
        patch("chibi.runners.ide_transport.recover_agent_jobs", new_callable=AsyncMock),
        patch("chibi.runners.ide_transport.task_manager") as task_manager_mock,
    ):
        task_manager_mock.shutdown = AsyncMock(side_effect=record_state)
        await runner.run()

    assert state_at_task_manager_shutdown == [STATE_STOPPED]


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_enabled")
@pytest.mark.asyncio
async def test_shutdown_scheduler_is_idempotent(sqlite_scheduler_settings) -> None:
    """Calling the shutdown helper repeatedly must not raise."""
    runner = make_runner()
    with patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings):
        await runner._start_scheduler()
        scheduler = StdioScheduler()
        try:
            await runner._shutdown_scheduler()
            await runner._shutdown_scheduler()
            assert scheduler.state == STATE_STOPPED
        finally:
            await runner._shutdown_scheduler()


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_disabled")
@pytest.mark.asyncio
async def test_run_without_gate_does_not_start_scheduler(tmp_path) -> None:
    """With the gate off, no scheduler is created and shutdown stays safe."""
    runner = make_runner()

    with (
        patch("chibi.runners.ide_transport.recover_agent_jobs", new_callable=AsyncMock) as recover_mock,
        patch("chibi.runners.ide_transport.task_manager") as task_manager_mock,
    ):
        task_manager_mock.shutdown = AsyncMock()
        await runner.run()

        recover_mock.assert_not_awaited()
        task_manager_mock.shutdown.assert_awaited_once()

    assert StdioScheduler not in SingletonMeta._instances
    assert not Path(tmp_path, STDIO_SCHEDULER_DB_FILENAME).exists()
    assert runner._scheduler is None


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_disabled")
@pytest.mark.asyncio
async def test_shutdown_scheduler_safe_when_never_started() -> None:
    """Shutting down a runner whose scheduler never started must be a no-op."""
    runner = make_runner()
    await runner._shutdown_scheduler()
    assert runner._scheduler is None


def test_stdio_identity_matches_reserved_storage_id() -> None:
    """Sanity: stdio runner sessions key jobs by the reserved negative IDE identity."""
    assert IDE_STORAGE_ID < 0
