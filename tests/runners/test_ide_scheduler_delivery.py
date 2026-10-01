"""Session-level scheduler delivery for the IDE stdio transport."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chibi.config  # noqa: F401
import chibi.services.jobs.agent_task as agent_task_module
from chibi.constants import IDE_STORAGE_ID
from chibi.runners.ide_transport import COMMANDS, PROTOCOL_VERSION, IDEStdioRunner
from chibi.schemas.scheduler import AgentJobPayload
from chibi.services.jobs.agent_task import _notify_failure, run_agent_job
from chibi.services.scheduler_interface import (
    StdioSchedulerInterface,
    clear_stdio_delivery_emitter,
    get_stdio_delivery_emitter,
)

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS},
}


class OutputRecorder:
    """Capture protocol frames through an async callable."""

    def __init__(self, delay: float = 0.0) -> None:
        """Initialize empty captured output.

        Args:
            delay: Artificial per-write delay used to interleave writers.
        """
        self.frames: list[dict[str, Any]] = []
        self._delay = delay

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: Emitted JSON-compatible protocol frame.
        """
        if self._delay:
            await asyncio.sleep(self._delay)
        self.frames.append(message)


def runner(recorder_delay: float = 0.0) -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    Args:
        recorder_delay: Artificial per-write delay for the recorder.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder(delay=recorder_delay)
    instance.__dict__["_write"] = recorder
    return instance, recorder.frames


def initialize(capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an initialize message.

    Args:
        capabilities: Optional client capabilities payload.

    Returns:
        A protocol initialize message.
    """
    message: dict[str, Any] = {"type": "initialize", "protocol_version": PROTOCOL_VERSION}
    if capabilities is not None:
        message["capabilities"] = capabilities
    return message


async def run_request(instance: IDEStdioRunner, request_id: str, thread_id: int) -> None:
    """Run one protocol request through the runner.

    Args:
        instance: Runner under test.
        request_id: Request identifier.
        thread_id: Thread identifier.
    """
    await instance._handle_message(
        {
            "type": "request",
            "request_id": request_id,
            "thread_id": thread_id,
            "prompt": "hi",
            "workspace_root": "/tmp",
            "active_file": None,
            "selection": None,
            "cursor_position": None,
            "language_id": None,
        }
    )


def stdio_interface(thread_id: int) -> StdioSchedulerInterface:
    """Build a scheduler interface bound to an IDE session thread.

    Args:
        thread_id: Client-minted session thread id.

    Returns:
        The stdio scheduler interface under test.
    """
    return StdioSchedulerInterface(
        user_id=IDE_STORAGE_ID, storage_id=IDE_STORAGE_ID, chat_id=IDE_STORAGE_ID, thread_id=thread_id
    )


def stdio_payload(job_id: str, thread_id: int = 42) -> AgentJobPayload:
    """Build a valid agent job payload bound to an IDE session thread.

    Args:
        job_id: Fully qualified job identifier.
        thread_id: Client-minted session thread id.

    Returns:
        The validated payload.
    """
    return AgentJobPayload.model_validate(
        {
            "job_id": job_id,
            "user_id": IDE_STORAGE_ID,
            "storage_id": IDE_STORAGE_ID,
            "chat_id": IDE_STORAGE_ID,
            "thread_id": thread_id,
            "title": "Stdio job",
            "action": {"type": "notify", "message": "Stdio job"},
        }
    )


@pytest.fixture(autouse=True)
def _clean_emitter() -> Any:
    """Ensure the module-level emitter registry is clean around every test."""
    clear_stdio_delivery_emitter()
    yield
    clear_stdio_delivery_emitter()


@pytest.fixture(autouse=True)
def _reset_anti_flood_state() -> Any:
    """Reset the dispatcher's in-memory anti-flood state around every test."""
    agent_task_module._failure_notify_timestamps.clear()
    yield
    agent_task_module._failure_notify_timestamps.clear()


class TestHandshakeRegistration:
    """The session-level emitter is installed at handshake."""

    @pytest.mark.asyncio
    async def test_handshake_registers_session_emitter(self) -> None:
        """After initialize the runner's scheduler emitter is registered."""
        instance, output = runner()

        await instance._handle_message(initialize({"background_messages": True}))

        assert get_stdio_delivery_emitter() == instance._emit_scheduler_message
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_handshake_registers_emitter_without_capability(self) -> None:
        """The emitter is registered even without background_messages: it self-gates."""
        instance, _output = runner()

        await instance._handle_message(initialize())

        assert get_stdio_delivery_emitter() == instance._emit_scheduler_message
        assert instance._background_messages_enabled is False

    @pytest.mark.asyncio
    async def test_new_handshake_replaces_previous_registration(self) -> None:
        """A fresh session's handshake overwrites the previous registration."""
        first, _ = runner()
        second, _ = runner()

        await first._handle_message(initialize({"background_messages": True}))
        await second._handle_message(initialize({"background_messages": True}))

        assert get_stdio_delivery_emitter() == second._emit_scheduler_message


class TestSchedulerMessageFrames:
    """Delivery of scheduled job output as unsolicited message frames."""

    @pytest.mark.asyncio
    async def test_delivery_emits_message_frame(self) -> None:
        """A scheduler message for a known session thread becomes a message frame."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        emitter = get_stdio_delivery_emitter()
        assert emitter is not None
        instance._session_threads.add(42)

        await emitter(42, "Job answer")

        assert {"type": "message", "thread_id": 42, "content": "Job answer"} in output

    @pytest.mark.asyncio
    async def test_interface_send_message_delivers_frame(self) -> None:
        """StdioSchedulerInterface delivers through the session emitter."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        instance._session_threads.add(7)

        await stdio_interface(7).send_message(message="Notify text")

        assert {"type": "message", "thread_id": 7, "content": "Notify text"} in output

    @pytest.mark.asyncio
    async def test_frame_dropped_when_capability_undeclared(self) -> None:
        """A client without background_messages never sees scheduler frames."""
        instance, output = runner()
        await instance._handle_message(initialize())
        emitter = get_stdio_delivery_emitter()
        assert emitter is not None
        instance._session_threads.add(42)

        await emitter(42, "Job answer")

        assert not any(f.get("type") == "message" for f in output)

    @pytest.mark.asyncio
    async def test_stale_thread_id_is_skipped_without_crash(self) -> None:
        """A payload thread unseen in this session is logged and skipped."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        emitter = get_stdio_delivery_emitter()
        assert emitter is not None

        await emitter(999, "Answer for a thread from a previous session")

        assert not any(f.get("type") == "message" for f in output)

    @pytest.mark.asyncio
    async def test_request_registers_thread_for_delivery(self) -> None:
        """A real request marks its thread as deliverable for scheduler jobs."""
        instance, output = runner()

        async def fake_prompt(interface: Any) -> None:
            await interface.send_message("Foreground answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"background_messages": True}))
            await run_request(instance, "r1", 42)
            # The request runs as a background task: wait for it to finish so
            # the thread registration in _run_request has happened.
            request_task = instance._tasks.get("r1")
            if request_task is not None:
                await request_task

        await stdio_interface(42).send_message(message="Job answer")

        assert {"type": "message", "thread_id": 42, "content": "Job answer"} in output

    @pytest.mark.asyncio
    async def test_scheduler_delivery_serializes_with_request_frames(self) -> None:
        """Concurrent scheduler delivery and request writes stay frame-atomic.

        The scheduler job and the JSONL loop share one asyncio loop; both
        writers contend for the same stdout lock, so a slow request write
        must never interleave with or drop a scheduler frame.
        """
        instance, output = runner(recorder_delay=0.01)
        await instance._handle_message(initialize({"background_messages": True}))
        instance._session_threads.add(42)
        emitter = get_stdio_delivery_emitter()
        assert emitter is not None

        await asyncio.gather(
            emitter(42, "Scheduler answer"),
            instance._write({"type": "result", "request_id": "r1", "content": "Request answer"}),
        )

        assert {"type": "message", "thread_id": 42, "content": "Scheduler answer"} in output
        assert {"type": "result", "request_id": "r1", "content": "Request answer"} in output


class TestDispatcherStdioDelivery:
    """The agent job dispatcher reaches the stdio client end to end."""

    @pytest.mark.asyncio
    async def test_notify_action_delivers_message_frame(self) -> None:
        """A fired notify job is delivered as a message frame via _build_interface."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        instance._session_threads.add(42)
        settings = SimpleNamespace(
            client="vscode",
            scheduler_notify_enabled=True,
            scheduler_agent_commands_enabled=True,
            scheduler_command_timeout_max=900,
            scheduler_failure_notify=True,
        )
        with patch("chibi.services.jobs.agent_task.application_settings", settings):
            await run_agent_job(
                "agent:-10000000000000000:reminder",
                user_id=IDE_STORAGE_ID,
                storage_id=IDE_STORAGE_ID,
                chat_id=IDE_STORAGE_ID,
                thread_id=42,
                title="Drink water",
                action={"type": "notify", "message": "Time to drink water"},
            )

        assert {"type": "message", "thread_id": 42, "content": "Time to drink water"} in output

    @pytest.mark.asyncio
    async def test_command_action_wakes_agent_through_stdio_interface(self) -> None:
        """A fired command job wakes the agent via the stdio interface; no direct message frame."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        instance._session_threads.add(42)
        settings = SimpleNamespace(
            client="vscode",
            scheduler_agent_commands_enabled=True,
            scheduler_command_timeout_max=900,
            scheduler_failure_notify=True,
        )
        process = MagicMock()
        process.returncode = 0
        process.pid = 4242
        process.communicate = AsyncMock(return_value=(b"git output\n", b""))
        with (
            patch("chibi.services.jobs.agent_task.application_settings", settings),
            patch("chibi.services.jobs.agent_task.get_moderation_provider", new=AsyncMock()),
            patch(
                "chibi.services.jobs.agent_task.asyncio.create_subprocess_shell",
                new=AsyncMock(return_value=process),
            ),
            patch("chibi.services.jobs.agent_task.handle_scheduler_trigger", new=AsyncMock()) as trigger_mock,
        ):
            await run_agent_job(
                "agent:-10000000000000000:git-check",
                user_id=IDE_STORAGE_ID,
                storage_id=IDE_STORAGE_ID,
                chat_id=IDE_STORAGE_ID,
                thread_id=42,
                title="Git check",
                action={"type": "command", "command": "git status --short"},
            )

        trigger_mock.assert_awaited_once()
        assert trigger_mock.await_args is not None
        assert isinstance(trigger_mock.await_args.kwargs["interface"], StdioSchedulerInterface)
        assert "git status --short" in trigger_mock.await_args.kwargs["trigger_text"]
        assert not [frame for frame in output if frame.get("type") == "message"]

    @pytest.mark.asyncio
    async def test_failure_notification_delivered_as_message_frame(self) -> None:
        """_notify_failure works in stdio: delivered as a frame, no ConfigurationError."""
        instance, output = runner()
        await instance._handle_message(initialize({"background_messages": True}))
        instance._session_threads.add(42)
        settings = SimpleNamespace(client="vscode", scheduler_failure_notify=True)

        with patch("chibi.services.jobs.agent_task.application_settings", settings):
            await _notify_failure(stdio_payload("agent:-10000000000000000:broken"), reason="RuntimeError: boom")

        frame = next(f for f in output if f.get("type") == "message")
        assert frame["thread_id"] == 42
        assert "agent:-10000000000000000:broken" in frame["content"]
        assert "RuntimeError: boom" in frame["content"]
