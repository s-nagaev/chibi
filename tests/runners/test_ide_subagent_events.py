"""Subagent lifecycle event frame coverage for the IDE stdio transport."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import COMMANDS, PROTOCOL_VERSION, IDEInterface, IDEStdioRunner
from chibi.services.bot import handle_stop
from chibi.services.subagent_events import subagent_tracker

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS},
}

RUNNERS: list[IDEStdioRunner] = []
SPAWNED_TASKS: list[asyncio.Task[None]] = []


class OutputRecorder:
    """Capture protocol frames written through the locked line writer."""

    def __init__(self, fail_on_types: set[str] | None = None) -> None:
        """Initialize empty captured output.

        Args:
            fail_on_types: Frame types whose writes must raise, simulating a broken stdout.
        """
        self.frames: list[dict[str, Any]] = []
        self._fail_on_types = fail_on_types or set()

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: JSON-compatible protocol frame.
        """
        if message.get("type") in self._fail_on_types:
            raise RuntimeError("stdout pipe is broken")
        self.frames.append(message)


def runner(fail_on_types: set[str] | None = None) -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    The recorder replaces ``_write_line`` so both regular writes and
    agent_event emissions (which bypass ``_write`` on purpose) are captured.

    Args:
        fail_on_types: Frame types whose writes must raise.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder(fail_on_types=fail_on_types)
    instance.__dict__["_write_line"] = recorder
    RUNNERS.append(instance)
    return instance, recorder.frames


def initialize(capabilities: Any = None) -> dict[str, Any]:
    """Build an initialize message.

    Args:
        capabilities: Optional client capabilities payload. Non-object values
            (for example a bare string) are allowed for malformed-input tests.

    Returns:
        A protocol initialize message.
    """
    message: dict[str, Any] = {"type": "initialize", "protocol_version": PROTOCOL_VERSION}
    if capabilities is not None:
        message["capabilities"] = capabilities
    return message


async def run_request(instance: IDEStdioRunner, request_id: str, thread_id: int, prompt: str = "hi") -> None:
    """Send one protocol request through the runner.

    Args:
        instance: Runner under test.
        request_id: Request identifier.
        thread_id: Thread identifier.
        prompt: Request prompt.
    """
    await instance._handle_message(
        {
            "type": "request",
            "request_id": request_id,
            "thread_id": thread_id,
            "prompt": prompt,
            "workspace_root": "/tmp",
            "active_file": None,
            "selection": None,
            "cursor_position": None,
            "language_id": None,
        }
    )


async def wait_for_frame(output: list[dict[str, Any]], frame_type: str) -> dict[str, Any]:
    """Wait until a frame of the given type is emitted and return it.

    Args:
        output: Captured frame list.
        frame_type: Frame type to wait for.

    Returns:
        The first matching frame.
    """
    for _ in range(300):
        for frame in output:
            if frame.get("type") == frame_type:
                return frame
        await asyncio.sleep(0.01)
    raise AssertionError(f"Missing {frame_type} frame: {output}")


async def wait_until_closed(interface: IDEInterface) -> None:
    """Wait until the request owning the interface is finished."""
    for _ in range(300):
        if interface._closed:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Interface was never closed")


def agent_events(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the agent_event frames from the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The agent_event frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == "agent_event"]


@pytest.fixture(autouse=True)
async def subagent_event_cleanup() -> Any:
    """Release the tracker sink and settle spawned tasks after each test."""
    yield
    subagent_tracker.set_sink(None)
    tasks, SPAWNED_TASKS[:] = SPAWNED_TASKS[:], []
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for instance in RUNNERS:
        await instance.drain_subagent_events()
    RUNNERS.clear()


class TestHandshake:
    """Capability parsing for the subagents opt-in flag."""

    @pytest.mark.asyncio
    async def test_subagents_capability_enables_emission(self) -> None:
        """Declaring subagents enables the session flag and binds the sink."""
        instance, output = runner()

        await instance._handle_message(initialize({"subagents": True}))

        assert instance._subagent_events_enabled is True
        assert subagent_tracker._sink is instance
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_unknown_capability_keys_are_ignored(self) -> None:
        """Capability keys the backend does not know are tolerated."""
        instance, output = runner()

        await instance._handle_message(initialize({"subagents": True, "future_flag": {"nested": 1}}))

        assert instance._subagent_events_enabled is True
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_v1_client_without_capability_never_binds_sink(self) -> None:
        """Clients that never send capabilities get no subagent sink."""
        instance, output = runner()

        await instance._handle_message(initialize())

        assert instance._subagent_events_enabled is False
        assert subagent_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_malformed_capabilities_do_not_enable_emission(self) -> None:
        """Non-object capabilities payloads keep emission off."""
        instance, output = runner()

        await instance._handle_message(initialize("not-an-object"))

        assert instance._subagent_events_enabled is False
        assert subagent_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_later_plain_session_releases_the_sink(self) -> None:
        """A session without the opt-in replaces a previously bound sink."""
        first, _ = runner()
        second, _ = runner()

        await first._handle_message(initialize({"subagents": True}))
        await second._handle_message(initialize())

        assert first._subagent_events_enabled is True
        assert second._subagent_events_enabled is False
        assert subagent_tracker._sink is None


class TestAgentEventFrames:
    """Emission of started and finished frames with per-request counters."""

    @pytest.mark.asyncio
    async def test_started_and_finished_frames_with_counters(self) -> None:
        """A delegated subagent produces exactly one started and one finished frame."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(42, "gpt-x")
            await subagent_tracker.drain()
            await interface.send_message("Delegated answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        events = agent_events(output)
        assert events == [
            {
                "type": "agent_event",
                "request_id": "r1",
                "event": "started",
                "active": 1,
                "total": 1,
                "name": "gpt-x",
            },
            {
                "type": "agent_event",
                "request_id": "r1",
                "event": "finished",
                "active": 0,
                "total": 1,
                "name": "gpt-x",
            },
        ]

    @pytest.mark.asyncio
    async def test_last_agent_event_precedes_result_frame(self) -> None:
        """No agent_event frame may arrive after the result frame."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(42, "gpt-x")
            await subagent_tracker.drain()
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        result_index = output.index(next(f for f in output if f.get("type") == "result"))
        event_indexes = [output.index(frame) for frame in agent_events(output)]
        assert event_indexes
        assert max(event_indexes) < result_index

    @pytest.mark.asyncio
    async def test_concurrent_subagents_drive_the_counters(self) -> None:
        """Two overlapping subagents report active 1, 2, 1, 0 with total growth."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            subagent_tracker.subagent_started(42, "gpt-y")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(42, "gpt-x")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(42, "gpt-y")
            await subagent_tracker.drain()
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        events = agent_events(output)
        assert [(event["event"], event["active"], event["total"]) for event in events] == [
            ("started", 1, 1),
            ("started", 2, 2),
            ("finished", 1, 2),
            ("finished", 0, 2),
        ]

    @pytest.mark.asyncio
    async def test_zero_active_on_last_natural_finish(self) -> None:
        """The final natural finish carries active 0 and no synthetic event follows."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(42, "gpt-x")
            await subagent_tracker.drain()
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await instance.drain_subagent_events()

        events = agent_events(output)
        assert events[-1]["active"] == 0
        assert events[-1]["event"] == "finished"

    @pytest.mark.asyncio
    async def test_fresh_request_restarts_counters_from_zero(self) -> None:
        """A second request on the same thread counts from zero again."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(interface.thread_id, "gpt-x")
            await subagent_tracker.drain()
            subagent_tracker.subagent_finished(interface.thread_id, "gpt-x")
            await subagent_tracker.drain()
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await run_request(instance, "r2", 42)
            for _ in range(300):
                if sum(1 for frame in output if frame.get("type") == "result") >= 2:
                    break
                await asyncio.sleep(0.01)
            await instance.drain_subagent_events()

        events = agent_events(output)
        second_request_events = [event for event in events if event["request_id"] == "r2"]
        assert [(event["event"], event["active"], event["total"]) for event in second_request_events] == [
            ("started", 1, 1),
            ("finished", 0, 1),
        ]

    @pytest.mark.asyncio
    async def test_late_finish_after_result_is_dropped(self) -> None:
        """A subagent finishing after its request completed emits nothing."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            await subagent_tracker.drain()

            async def late_finish() -> None:
                await wait_until_closed(interface)
                subagent_tracker.subagent_finished(42, "gpt-x")

            task = asyncio.create_task(late_finish())
            SPAWNED_TASKS.append(task)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            for task in list(SPAWNED_TASKS):
                await asyncio.gather(task)
            await instance.drain_subagent_events()

        events = agent_events(output)
        assert [event["event"] for event in events] == ["started"]
        result_index = output.index(next(f for f in output if f.get("type") == "result"))
        assert all(output.index(event) < result_index for event in events)


class TestV1ClientCompatibility:
    """Clients without the opt-in never see agent_event frames."""

    @pytest.mark.asyncio
    async def test_no_agent_events_without_optin(self) -> None:
        """Lifecycle calls with no bound sink produce zero frames."""
        instance, output = runner()

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            subagent_tracker.subagent_finished(42, "gpt-x")
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await instance.drain_subagent_events()

        assert agent_events(output) == []
        assert any(frame.get("type") == "result" for frame in output)


class TestEmissionFailureResilience:
    """A broken stdout during agent_event emission must not break the request."""

    @pytest.mark.asyncio
    async def test_write_failure_is_dropped_without_raising(self) -> None:
        """A failing agent_event write raises nothing at the emission site."""
        instance, output = runner(fail_on_types={"agent_event"})

        async def fake_prompt(interface: IDEInterface) -> None:
            subagent_tracker.subagent_started(42, "gpt-x")
            subagent_tracker.subagent_finished(42, "gpt-x")
            await subagent_tracker.drain()
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await instance.drain_subagent_events()

        assert agent_events(output) == []
        assert any(frame.get("type") == "result" for frame in output)


class TestKillFlush:
    """Single kill-flush event on successful /stop and /reset."""

    @pytest.mark.asyncio
    async def test_reset_with_inflight_counters_emits_exactly_one_flush(self) -> None:
        """/reset on a thread with in-flight subagents emits one authoritative flush."""
        from unittest.mock import AsyncMock

        instance, output = runner()
        gate = asyncio.Event()

        async def hanging_subagent() -> None:
            try:
                await gate.wait()
            finally:
                # An individually killed subagent must NOT emit its own finished
                # event; the flush is the authoritative reset for clients.
                subagent_tracker.subagent_finished(7, "gpt-x")

        async def fake_prompt(interface: IDEInterface) -> None:
            task = asyncio.get_running_loop().create_task(hanging_subagent())
            SPAWNED_TASKS.append(task)
            subagent_tracker.subagent_started(7, "gpt-x")
            await subagent_tracker.drain()
            await gate.wait()

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patch("chibi.services.bot.reset_chat_history", new=AsyncMock(return_value=None)),
        ):
            await instance._handle_message(initialize({"subagents": True}))
            await run_request(instance, "r1", 7)
            await wait_for_frame(output, "agent_event")
            await run_request(instance, "r2", 7, prompt="/reset")
            await wait_for_frame(output, "result")
            gate.set()
            for task in list(SPAWNED_TASKS):
                await asyncio.gather(task, return_exceptions=True)
            await instance.drain_subagent_events()

        flush_events = [
            event
            for event in agent_events(output)
            if event["event"] == "finished" and event["active"] == 0 and "name" not in event
        ]
        assert len(flush_events) == 1
        assert flush_events[0]["request_id"] == "r1"
        assert flush_events[0]["total"] == 1
        # The killed subagent's cleanup-path finish produced no extra event.
        natural_finishes = [
            event
            for event in agent_events(output)
            if event["event"] == "finished" and event["request_id"] == "r1" and "name" in event
        ]
        assert natural_finishes == []

    @pytest.mark.asyncio
    async def test_stop_with_inflight_counters_emits_flush(self) -> None:
        """A successful /stop flushes in-flight counters as one zero-active event."""
        instance, output = runner()
        responses: list[str] = []

        await instance._handle_message(initialize({"subagents": True}))
        instance.begin_request(7, "r1")
        subagent_tracker.subagent_started(7, "gpt-x")
        await subagent_tracker.drain()

        interface = IDEInterface(7, "/stop", {}, responses.append)
        await handle_stop(interface=interface)
        await instance.drain_subagent_events()

        flush_events = [
            event
            for event in agent_events(output)
            if event["event"] == "finished" and event["active"] == 0 and "name" not in event
        ]
        assert flush_events == [
            {"type": "agent_event", "request_id": "r1", "event": "finished", "active": 0, "total": 1}
        ]
        assert responses == ["Everything stopped."]

    @pytest.mark.asyncio
    async def test_stop_without_counters_emits_no_flush(self) -> None:
        """No flush when the counters were already zero."""
        instance, output = runner()
        responses: list[str] = []

        await instance._handle_message(initialize({"subagents": True}))
        instance.begin_request(7, "r1")

        interface = IDEInterface(7, "/stop", {}, responses.append)
        await handle_stop(interface=interface)
        await instance.drain_subagent_events()

        assert agent_events(output) == []
