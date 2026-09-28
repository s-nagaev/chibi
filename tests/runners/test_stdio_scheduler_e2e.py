"""End-to-end stdio scheduler scenario: schedule → fire → unsolicited message frame → clean shutdown.

Covers the full stdio path in a single scenario: a fake IDE client handshakes
with the ``background_messages`` capability, sends one request on its thread
(registering it as deliverable), the agent creates a one-time ``self`` job
through the real ``schedule_task`` tool, the stdio scheduler fires it, the
woken agent's answer reaches the client as an unsolicited ``message`` frame,
and the runner shuts down cleanly (scheduler stopped, task manager drained).
"""

import asyncio
import json
import queue
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.base import STATE_STOPPED

import chibi.config  # noqa: F401
from chibi.constants import IDE_STORAGE_ID
from chibi.runners.ide_transport import PROTOCOL_VERSION, IDEInterface, IDEStdioRunner
from chibi.schemas.app import ChatResponseSchema
from chibi.services import task_manager as task_manager_module
from chibi.services.jobs.agent_task import run_agent_job
from chibi.services.providers.tools.scheduler import ScheduleTaskTool
from chibi.services.scheduler import STDIO_SCHEDULER_DB_FILENAME, StdioScheduler
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

TOOL_MODULE = "chibi.services.providers.tools.scheduler"
SCHEDULER_MODULE = "chibi.services.scheduler"
AGENT_TASK_MODULE = "chibi.services.jobs.agent_task"
FIRE_TIMEOUT_SECONDS = 30.0
THREAD_ID = 42
JOB_ANSWER = "All systems nominal"
STDIN_READ_TIMEOUT_SECONDS = 90.0


class ScriptedStdin:
    """Blocking fake stdin consumed by the runner's real ``_read_line`` path.

    ``IDEStdioRunner._read_line`` calls ``sys.stdin.readline`` inside
    ``asyncio.to_thread``, so ``readline`` must block the worker thread until
    the next JSONL line (or EOF) is supplied from the test coroutine.
    """

    def __init__(self) -> None:
        """Initialize an empty line queue."""
        self._lines: queue.Queue[str] = queue.Queue()

    def put_line(self, message: dict[str, Any]) -> None:
        """Enqueue one protocol message as a JSONL line.

        Args:
            message: JSON-compatible protocol message to feed to the runner.
        """
        self._lines.put(json.dumps(message, ensure_ascii=False))

    def close(self) -> None:
        """Signal stdin EOF so the runner's event loop exits into its ``finally`` block."""
        self._lines.put("")

    def readline(self) -> str:
        """Block until the next line is available.

        Returns:
            The next JSONL line, or an empty string at EOF.

        Raises:
            queue.Empty: If no line arrives before the read timeout, so a
                broken test fails instead of hanging forever.
        """
        return self._lines.get(timeout=STDIN_READ_TIMEOUT_SECONDS)


async def _wait_until(predicate: Any, timeout: float = FIRE_TIMEOUT_SECONDS, poll_interval: float = 0.05) -> None:
    """Poll a predicate until it becomes True or the timeout expires.

    Args:
        predicate: Zero-argument callable checked on every poll iteration.
        timeout: Maximum time to wait in seconds.
        poll_interval: Sleep between polls in seconds.

    Raises:
        AssertionError: If the predicate is still False when the timeout expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll_interval)
    raise AssertionError("Condition was not met within the timeout")


class OutputRecorder:
    """Capture protocol frames the runner writes to its stdout."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: Emitted JSON-compatible protocol frame.
        """
        self.frames.append(message)


def make_runner() -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Build a stdio runner with captured protocol output.

    Returns:
        The runner under test and its mutable captured-frame list.
    """
    runner = IDEStdioRunner()
    recorder = OutputRecorder()
    runner.__dict__["_write"] = recorder
    return runner, recorder.frames


@pytest.fixture()
def stdio_scheduler_settings(tmp_path):
    """Scheduler settings pointed at a temp data path (SQLite stdio job store)."""
    return SimpleNamespace(
        redis=None,
        local_data_path=str(tmp_path),
        scheduler_misfire_grace_time=3600,
        chroma_history_retention_days=7,
    )


@pytest.fixture()
def tool_settings():
    """Tool-gate settings as a stdio client sees them (every scheduler gate enabled)."""
    return SimpleNamespace(
        client="vscode",
        scheduler_tool_enabled=True,
        scheduler_notify_enabled=True,
        scheduler_agent_commands_enabled=True,
        scheduler_command_timeout_max=900,
        scheduler_failure_notify=True,
    )


@pytest.fixture()
def agent_task_settings():
    """Dispatcher settings: the fired job must deliver through the stdio interface."""
    return SimpleNamespace(client="vscode", scheduler_failure_notify=True)


@pytest.fixture()
def fresh_stdio_scheduler_singleton():
    """Ensure clean scheduler singletons around the test."""
    SingletonMeta._instances.pop(StdioScheduler, None)
    SingletonMeta._instances.pop(StdioScheduler.__mro__[1], None)
    yield
    SingletonMeta._instances.pop(StdioScheduler, None)
    SingletonMeta._instances.pop(StdioScheduler.__mro__[1], None)


@pytest.fixture()
def scheduler_gate_enabled():
    """Force ``scheduler_tool_enabled`` on for the duration of the test."""
    from chibi.config import application_settings

    original = application_settings.scheduler_tool_enabled
    application_settings.scheduler_tool_enabled = True
    yield
    application_settings.scheduler_tool_enabled = original


@pytest.fixture(autouse=True)
def _clean_emitter():
    """Ensure the module-level emitter registry is clean around every test."""
    from chibi.services.scheduler_interface import clear_stdio_delivery_emitter

    clear_stdio_delivery_emitter()
    yield
    clear_stdio_delivery_emitter()


@pytest.fixture(autouse=True)
def _reset_anti_flood_state():
    """Reset the dispatcher's in-memory anti-flood state around every test."""
    import chibi.services.jobs.agent_task as agent_task_module

    agent_task_module._failure_notify_timestamps.clear()
    yield
    agent_task_module._failure_notify_timestamps.clear()


@pytest.mark.usefixtures("fresh_stdio_scheduler_singleton", "scheduler_gate_enabled")
async def test_once_self_job_end_to_end_over_stdio(
    stdio_scheduler_settings, tool_settings, agent_task_settings, tmp_path
) -> None:
    """A tool-created one-time self job fires and the answer lands as a message frame.

    Drives a genuine ``IDEStdioRunner.run()`` session: the fake client's JSONL
    lines flow through the real stdin reader path, and the runner's own
    ``finally`` block performs the scheduler and task-manager shutdown.
    """
    llm_calls: list[int] = []

    async def fake_llm_answer(**kwargs: Any) -> ChatResponseSchema:
        """Return the scripted agent answer for the woken scheduler turn.

        Args:
            kwargs: Ignored LLM chain arguments.

        Returns:
            The scripted chat response.
        """
        llm_calls.append(1)
        return ChatResponseSchema(answer=JOB_ANSWER, provider="OpenAI", model="m", usage=None)

    async def fake_prompt(interface: IDEInterface) -> None:
        """Serve the foreground request without a real LLM.

        Args:
            interface: Request-local interface of the running request.
        """
        await interface.send_message("Foreground answer")

    runner, output = make_runner()
    fake_stdin = ScriptedStdin()
    with (
        patch(f"{SCHEDULER_MODULE}.application_settings", stdio_scheduler_settings),
        patch("chibi.services.jobs.archive.perform_retention_cleanup", new_callable=MagicMock),
        patch("chibi.runners.ide_transport.recover_agent_jobs", new_callable=AsyncMock),
        patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(side_effect=fake_llm_answer)),
        patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        patch(f"{TOOL_MODULE}.application_settings", tool_settings),
        patch(f"{AGENT_TASK_MODULE}.application_settings", agent_task_settings),
        patch("chibi.config.logging.use_stderr_logging"),
        patch("sys.stdin", fake_stdin),
        patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
    ):
        run_task = asyncio.create_task(runner.run())

        fake_stdin.put_line(
            {"type": "initialize", "protocol_version": PROTOCOL_VERSION, "capabilities": {"background_messages": True}}
        )
        await _wait_until(lambda: runner._initialized)

        fake_stdin.put_line(
            {
                "type": "request",
                "request_id": "r1",
                "thread_id": THREAD_ID,
                "prompt": "hi",
                "workspace_root": "/tmp",
                "active_file": None,
                "selection": None,
                "cursor_position": None,
                "language_id": None,
            }
        )
        await _wait_until(lambda: any(frame.get("content") == "Foreground answer" for frame in output))
        request_task = runner._tasks.get("r1")
        if request_task is not None:
            await request_task

        interface = IDEInterface(thread_id=THREAD_ID, prompt="schedule", context={}, emit=lambda text: None)
        run_at = datetime.now() + timedelta(seconds=1.5)
        result = await ScheduleTaskTool.function(
            title="stdio check",
            schedule={"kind": "once", "at": run_at.isoformat()},
            action={"type": "self", "trigger_text": "Run the stdio check"},
            user_id=IDE_STORAGE_ID,
            caller_model="test-model",
            interface=interface,
        )

        assert result["status"] == "ok"
        job_id = result["job_id"]
        assert job_id.startswith(f"agent:{IDE_STORAGE_ID}:")

        scheduler = StdioScheduler()
        jobs = [job for job in scheduler.get_jobs() if job.id == job_id]
        assert len(jobs) == 1, "The job must live in the stdio job store"
        job = jobs[0]
        assert job.func is run_agent_job
        assert job.kwargs["user_id"] == IDE_STORAGE_ID
        assert job.kwargs["storage_id"] == IDE_STORAGE_ID
        assert job.kwargs["thread_id"] == THREAD_ID
        assert job.kwargs["action"]["type"] == "self"
        assert job.kwargs["action"]["trigger_text"] == "Run the stdio check"

        await _wait_until(lambda: llm_calls, timeout=FIRE_TIMEOUT_SECONDS)
        await _wait_until(
            lambda: any(frame.get("type") == "message" and frame.get("thread_id") == THREAD_ID for frame in output),
            timeout=FIRE_TIMEOUT_SECONDS,
        )
        frame = next(
            frame for frame in output if frame.get("type") == "message" and frame.get("thread_id") == THREAD_ID
        )
        assert frame == {"type": "message", "thread_id": THREAD_ID, "content": JOB_ANSWER}

        fake_stdin.close()
        exit_code = await run_task

    assert exit_code == 0, "The runner session must end cleanly on stdin EOF"
    assert scheduler.state == STATE_STOPPED, "run() must stop the stdio scheduler in its finally block"
    assert not task_manager_module.task_manager._task_to_user_id, "No background tasks may survive the shutdown"

    assert Path(tmp_path, STDIO_SCHEDULER_DB_FILENAME).exists()
    assert not Path(tmp_path, "scheduler.db").exists(), "The tool must never touch the Telegram job store"
