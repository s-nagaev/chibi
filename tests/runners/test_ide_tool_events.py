"""Tool-call lifecycle event frame coverage for the IDE stdio transport."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

import chibi.config  # noqa: F401
from chibi.interfaces.stdio_runner import (
    COMMANDS,
    PROTOCOL_VERSION,
    TOOL_EVENT_ARGS_MAX_CHARS,
    TOOL_EVENT_RESULT_PREVIEW_MAX_CHARS,
    StdioRunner,
)
from chibi.services.events.subagent_events import subagent_tracker
from chibi.services.events.tool_events import tool_event_tracker
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import CallTracker

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS, "models": True},
}

RUNNERS: list[StdioRunner] = []


class EchoTool(ChibiTool):
    """Test-only foreground tool used to exercise the emission path."""

    register = False
    run_in_background_by_default = False
    name = "echo_test"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="echo_test",
            description="Echo a value back.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    )

    @classmethod
    async def function(cls, **kwargs: Any) -> dict[str, Any]:
        """Echo the ``value`` argument back.

        Args:
            **kwargs: The tool payload.

        Returns:
            The echoed value under the ``echo`` key.
        """
        return {"echo": kwargs.get("value")}


class ErrorTool(ChibiTool):
    """Test-only foreground tool whose function always raises."""

    register = False
    run_in_background_by_default = False
    name = "error_test"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="error_test",
            description="Fail.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    )

    @classmethod
    async def function(cls, **kwargs: Any) -> dict[str, Any]:
        """Raise a deterministic failure.

        Args:
            **kwargs: The tool payload.

        Raises:
            ValueError: Always raised.
        """
        raise ValueError("boom")


class OutputRecorder:
    """Capture protocol frames written through the locked line writer."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: JSON-compatible protocol frame.
        """
        self.frames.append(message)


def runner() -> tuple[StdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    The recorder replaces ``_write_line`` so every write path is captured.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = StdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write_line"] = recorder
    RUNNERS.append(instance)
    return instance, recorder.frames


def initialize(capabilities: Any = None) -> dict[str, Any]:
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


def tool_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the tool_event frames from the captured output.

    Args:
        frames: Captured frame list.

    Returns:
        The tool_event frames in wire order.
    """
    return [frame for frame in frames if frame.get("type") == "tool_event"]


def running_tool_events(thread_id: int = 7) -> SimpleNamespace:
    """Build a minimal open interface stand-in for a tool call.

    Args:
        thread_id: The thread id reported by the interface.

    Returns:
        An interface stand-in whose request is still open.
    """
    return SimpleNamespace(thread_id=thread_id, is_closed=False)


@pytest.fixture(autouse=True)
async def tool_event_cleanup() -> Any:
    """Release sinks, settle emissions and reset the call tracker after each test."""
    yield
    tool_event_tracker.set_sink(None)
    subagent_tracker.set_sink(None)
    for instance in RUNNERS:
        await instance.drain_subagent_events()
    RUNNERS.clear()
    CallTracker()._calls.clear()


class TestHandshake:
    """Capability parsing for the tool_events opt-in flag."""

    @pytest.mark.asyncio
    async def test_tool_events_capability_enables_emission(self) -> None:
        """Declaring tool_events enables the session flag and binds the sink."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))

        assert instance._tool_events_enabled is True
        assert tool_event_tracker._sink is instance
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_v1_client_without_capability_never_binds_sink(self) -> None:
        """Clients that never send capabilities get no tool-event sink."""
        instance, output = runner()

        await instance._handle_message(initialize())

        assert instance._tool_events_enabled is False
        assert tool_event_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_malformed_capabilities_do_not_enable_emission(self) -> None:
        """Non-object capabilities payloads keep emission off."""
        instance, output = runner()

        await instance._handle_message(initialize("not-an-object"))

        assert instance._tool_events_enabled is False
        assert tool_event_tracker._sink is None
        assert output[0] == READY_FRAME


class TestActiveRequestMap:
    """The dedicated thread -> request_id map, independent of the subagents gate."""

    @pytest.mark.asyncio
    async def test_begin_request_populates_the_map_without_subagents(self) -> None:
        """begin_request fills the map even when the subagents gate is closed."""
        instance, _output = runner()

        await instance._handle_message(initialize({"tool_events": True}))

        assert instance._subagent_events_enabled is False
        instance.begin_request(7, "r1")

        assert instance._active_requests == {7: "r1"}
        assert instance._subagent_requests == {}

    @pytest.mark.asyncio
    async def test_end_request_clears_the_map_entry(self) -> None:
        """end_request drops the map entry for the request that owns the thread."""
        instance, _output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")
        instance.end_request(7, "r1")

        assert instance._active_requests == {}

    @pytest.mark.asyncio
    async def test_end_request_keeps_a_newer_owners_entry(self) -> None:
        """A finishing request never evicts a newer request's map entry."""
        instance, _output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")
        instance.begin_request(7, "r2")
        instance.end_request(7, "r1")

        assert instance._active_requests == {7: "r2"}


class TestForegroundEmission:
    """Start/end frames around foreground tool calls."""

    @pytest.mark.asyncio
    async def test_start_and_end_frames_with_shape(self) -> None:
        """A foreground call emits exactly one start and one end frame."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        result = await EchoTool.tool(value="hello", interface=running_tool_events(7))

        assert result.status == "ok"
        assert tool_events(output) == [
            {
                "type": "tool_event",
                "request_id": "r1",
                "phase": "start",
                "name": "echo_test",
                "args": {"value": "hello"},
            },
            {
                "type": "tool_event",
                "request_id": "r1",
                "phase": "end",
                "name": "echo_test",
                "ok": True,
                "result_preview": '{"echo": "hello"}',
            },
        ]

    @pytest.mark.asyncio
    async def test_emits_when_subagents_disabled(self) -> None:
        """tool_events=true with subagents=false still emits via the dedicated map."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True, "subagents": False}))
        assert instance._subagent_events_enabled is False
        assert instance._tool_events_enabled is True
        instance.begin_request(7, "r1")

        await EchoTool.tool(value="only-tool-events", interface=running_tool_events(7))

        frames = tool_events(output)
        assert [frame["phase"] for frame in frames] == ["start", "end"]
        assert all(frame["request_id"] == "r1" for frame in frames)

    @pytest.mark.asyncio
    async def test_failed_tool_reports_not_ok(self) -> None:
        """A tool that fails internally emits a paired end frame with ok false."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        result = await ErrorTool.tool(interface=running_tool_events(7))

        assert result.status == "error"
        frames = tool_events(output)
        assert [frame["phase"] for frame in frames] == ["start", "end"]
        assert frames[1]["ok"] is False
        assert frames[1]["result_preview"] == "boom"

    @pytest.mark.asyncio
    async def test_args_cap_replaces_oversized_payload(self) -> None:
        """An oversized args mapping is replaced by the truncation marker."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        await EchoTool.tool(value="x" * (TOOL_EVENT_ARGS_MAX_CHARS * 4), interface=running_tool_events(7))

        start = tool_events(output)[0]
        assert start["args"] == {"_truncated": True}
        assert len(json.dumps(start["args"])) <= TOOL_EVENT_ARGS_MAX_CHARS

    @pytest.mark.asyncio
    async def test_result_preview_is_capped(self) -> None:
        """The result preview never exceeds the wire cap."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        await EchoTool.tool(value="y" * (TOOL_EVENT_RESULT_PREVIEW_MAX_CHARS * 3), interface=running_tool_events(7))

        end = tool_events(output)[1]
        assert len(end["result_preview"]) == TOOL_EVENT_RESULT_PREVIEW_MAX_CHARS


class TestNoEmission:
    """Paths that must never emit a tool_event frame."""

    @pytest.mark.asyncio
    async def test_no_capability_emits_nothing(self) -> None:
        """Without the opt-in no frames are emitted, even for a live request."""
        instance, output = runner()

        await instance._handle_message(initialize())
        instance.begin_request(7, "r1")

        await EchoTool.tool(value="no-optin", interface=running_tool_events(7))

        assert tool_events(output) == []

    @pytest.mark.asyncio
    async def test_background_run_emits_nothing(self) -> None:
        """A run_in_background tool call emits nothing, unconditionally."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        def swallow(coro: Any, **kwargs: Any) -> None:
            coro.close()
            return None

        with patch("chibi.services.providers.tools.tool.task_manager.run_task", side_effect=swallow):
            result = await EchoTool.tool(value="background", interface=running_tool_events(7), run_in_background=True)

        assert result.status == "tool is running in background"
        assert tool_events(output) == []

    @pytest.mark.asyncio
    async def test_interface_less_call_emits_nothing(self) -> None:
        """A call with no interface (sub-agent internal) emits nothing."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        await EchoTool.tool(value="no-interface", interface=None)

        assert tool_events(output) == []

    @pytest.mark.asyncio
    async def test_closed_interface_emits_nothing(self) -> None:
        """A call on an already-finished request emits nothing."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        closed = SimpleNamespace(thread_id=7, is_closed=True)
        await EchoTool.tool(value="closed", interface=closed)

        assert tool_events(output) == []

    @pytest.mark.asyncio
    async def test_no_active_request_emits_nothing(self) -> None:
        """With no live request on the thread the sink drops the frames."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))

        await EchoTool.tool(value="no-request", interface=running_tool_events(7))

        assert tool_events(output) == []

    @pytest.mark.asyncio
    async def test_after_end_request_emits_nothing(self) -> None:
        """A call after the request was retired emits nothing."""
        instance, output = runner()

        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")
        instance.end_request(7, "r1")

        await EchoTool.tool(value="after-end", interface=running_tool_events(7))

        assert tool_events(output) == []


class TestWriteFailureResilience:
    """A broken stdout during tool_event emission must not break the request."""

    @pytest.mark.asyncio
    async def test_write_failure_is_swallowed(self) -> None:
        """A failing tool_event write raises nothing at the emission site."""
        instance, output = runner()

        async def failing_writer(message: dict[str, Any]) -> None:
            if message.get("type") == "tool_event":
                raise RuntimeError("stdout pipe is broken")
            output.append(message)

        instance.__dict__["_write_line"] = failing_writer
        await instance._handle_message(initialize({"tool_events": True}))
        instance.begin_request(7, "r1")

        result = await EchoTool.tool(value="broken-pipe", interface=running_tool_events(7))

        assert result.status == "ok"
        assert tool_events(output) == []
