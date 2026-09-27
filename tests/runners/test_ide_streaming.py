"""Delta frame coverage for the IDE stdio transport streaming support."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import DELTA_FLUSH_INTERVAL, IDEInterface, IDEStdioRunner
from chibi.services.interface import UserInterface

REQUEST_FIELDS: dict[str, Any] = {
    "workspace_root": "/tmp",
    "active_file": None,
    "selection": None,
    "cursor_position": None,
    "language_id": None,
}

STREAM_CHUNKS = ["Hello ", "brave ", "new ", "streaming ", "world"]
CADENCE_CHUNKS = ["a", "b", "c", "d", "e"]

_gate: asyncio.Event | None = None
_midpoint: asyncio.Event | None = None


async def fake_streaming_prompt(interface: Any) -> None:
    """Emit several live chunks and finish immediately."""
    for chunk in STREAM_CHUNKS:
        await interface.send_delta(chunk)
    await interface.send_message("final answer")


async def fake_cadence_prompt(interface: Any) -> None:
    """Emit buffered chunks, idle past the flush interval, then finish."""
    for chunk in CADENCE_CHUNKS:
        await interface.send_delta(chunk)
    await asyncio.sleep(DELTA_FLUSH_INTERVAL + 0.1)
    await interface.send_message("done")


async def fake_gated_prompt(interface: Any) -> None:
    """Emit one pending chunk and block on the test gate."""
    await interface.send_delta("partial ")
    if _gate is not None:
        await _gate.wait()
    await interface.send_message("never")


async def fake_single_chunk_prompt(interface: Any) -> None:
    """Emit one pending chunk and finish before the flush interval."""
    await interface.send_delta("only chunk")
    await interface.send_message("answer")


class OutputRecorder:
    """Capture protocol frames with wall-clock write timestamps."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []
        self.stamps: list[float] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: Emitted JSON-compatible protocol frame.
        """
        self.frames.append(message)
        self.stamps.append(asyncio.get_running_loop().time())


class DeltaCapture:
    """Collect chunks handed to a request-local delta_emit callback."""

    def __init__(self) -> None:
        """Initialize empty captured chunks."""
        self.chunks: list[str] = []

    async def __call__(self, text: str) -> None:
        """Capture one forwarded chunk.

        Args:
            text: The partial text chunk being forwarded.
        """
        self.chunks.append(text)


class BackgroundCapture:
    """Collect text routed to a background_emit callback."""

    def __init__(self) -> None:
        """Initialize empty captured background emissions."""
        self.payloads: list[tuple[int, str]] = []

    async def __call__(
        self, thread_id: int, content: str, model: str | None, provider: str | None, thoughts: str | None
    ) -> None:
        """Capture one background emission.

        Args:
            thread_id: Thread the background text belongs to.
            content: Assistant text being delivered.
            model: Model that produced the answer, when known.
            provider: Provider that produced the answer, when known.
            thoughts: Continuation reasoning, when present.
        """
        self.payloads.append((thread_id, content))


def runner() -> tuple[IDEStdioRunner, OutputRecorder]:
    """Create a runner with captured protocol output.

    Returns:
        The runner and its captured-frame recorder.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write_line"] = recorder
    return instance, recorder


def initialize(capabilities: Any = None) -> dict[str, Any]:
    """Build an initialize message.

    Args:
        capabilities: Optional client capabilities payload.

    Returns:
        A protocol initialize message.
    """
    message: dict[str, Any] = {"type": "initialize", "protocol_version": 1}
    if capabilities is not None:
        message["capabilities"] = capabilities
    return message


def request(request_id: str, thread_id: int, prompt: str = "hello") -> dict[str, Any]:
    """Build a valid request message.

    Args:
        request_id: Unique request identifier.
        thread_id: Thread the request runs on.
        prompt: Request prompt.

    Returns:
        A protocol request message.
    """
    return {"type": "request", "request_id": request_id, "thread_id": thread_id, "prompt": prompt, **REQUEST_FIELDS}


def deltas(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the delta frames from the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The delta frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == "delta"]


def terminal_index(output: list[dict[str, Any]], request_id: str) -> int:
    """Return the index of the terminal frame of a request.

    Args:
        output: Captured frame list.
        request_id: Request whose terminal frame is located.

    Returns:
        The index of the result or error frame.

    Raises:
        AssertionError: When no terminal frame exists for the request.
    """
    for index, frame in enumerate(output):
        if frame.get("request_id") == request_id and frame.get("type") in ("result", "error"):
            return index
    raise AssertionError(f"No terminal frame for {request_id}: {output}")


async def wait_for_frame(output: list[dict[str, Any]], request_id: str, frame_type: str) -> None:
    """Wait until a correlated frame is emitted.

    Args:
        output: Captured frame list.
        request_id: Request identifier to correlate on.
        frame_type: Frame type to wait for.

    Raises:
        AssertionError: When the frame never appears.
    """
    for _ in range(300):
        for frame in output:
            if frame.get("request_id") == request_id and frame.get("type") == frame_type:
                return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Missing {frame_type} for {request_id}: {output}")


RUNNERS: list[IDEStdioRunner] = []


@pytest.fixture(autouse=True)
async def delta_state_cleanup() -> Any:
    """Cancel leftover flush timers and settle runner state after each test."""
    yield
    for instance in RUNNERS:
        for timer in list(instance._delta_timers.values()):
            timer.cancel()
    RUNNERS.clear()


class TestCapabilityGate:
    """streaming capability parsing and delta gating."""

    @pytest.mark.asyncio
    async def test_capability_enables_delta_frames(self) -> None:
        """Declaring streaming routes provider chunks into delta frames."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_streaming_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        assert instance._streaming_enabled is True
        frames = deltas(recorder.frames)
        assert frames, "streaming client must receive delta frames"
        assert all(
            frame == {"type": "delta", "request_id": "r1", "thread_id": 42, "text": frame["text"]} for frame in frames
        )
        assert "".join(frame["text"] for frame in frames) == "".join(STREAM_CHUNKS)

    @pytest.mark.asyncio
    async def test_capability_off_emits_no_delta_frames(self) -> None:
        """Clients without the opt-in never see delta frames."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_streaming_prompt):
            await instance._handle_message(initialize())
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        assert instance._streaming_enabled is False
        assert deltas(recorder.frames) == []
        results = [frame for frame in recorder.frames if frame.get("type") == "result"]
        assert results[0]["content"] == "final answer"

    @pytest.mark.asyncio
    async def test_capability_absent_flag_is_off(self) -> None:
        """A non-true capabilities value keeps the session flag off."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize({"streaming": "yes"}))

        assert instance._streaming_enabled is False
        assert recorder.frames[0]["type"] == "ready"


class TestWireOrdering:
    """Deltas strictly precede the terminal frame on the wire."""

    @pytest.mark.asyncio
    async def test_deltas_precede_result(self) -> None:
        """Every delta frame is written before the result frame."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_streaming_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        terminal = terminal_index(recorder.frames, "r1")
        assert all(recorder.frames.index(frame) < terminal for frame in deltas(recorder.frames))

    @pytest.mark.asyncio
    async def test_drain_flushes_pending_delta_before_result(self) -> None:
        """A buffered chunk is flushed by drain_deltas ahead of the result."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_single_chunk_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        frames = deltas(recorder.frames)
        assert len(frames) == 1
        assert frames[0]["text"] == "only chunk"
        assert recorder.frames.index(frames[0]) < terminal_index(recorder.frames, "r1")


class TestCoalescing:
    """Transport-level buffering and flush cadence."""

    @pytest.mark.asyncio
    async def test_rapid_chunks_coalesce_into_one_frame(self) -> None:
        """Chunks arriving within one interval share a single flush."""
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_cadence_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        frames = deltas(recorder.frames)
        assert len(frames) == 1
        assert frames[0]["text"] == "".join(CADENCE_CHUNKS)
        delta_index = recorder.frames.index(frames[0])
        running_index = next(
            index
            for index, frame in enumerate(recorder.frames)
            if frame.get("type") == "status" and frame.get("state") == "running"
        )
        assert recorder.stamps[delta_index] - recorder.stamps[running_index] >= DELTA_FLUSH_INTERVAL - 0.05

    @pytest.mark.asyncio
    async def test_large_buffer_flushes_before_interval(self) -> None:
        """A buffer reaching the character threshold flushes immediately."""
        instance, recorder = runner()
        RUNNERS.append(instance)
        big_chunk = "x" * 600

        async def fake_big_chunk_prompt(interface: Any) -> None:
            await interface.send_delta(big_chunk)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_big_chunk_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")

        frames = deltas(recorder.frames)
        assert len(frames) == 1
        assert frames[0]["text"] == big_chunk


class TestErrorAndCancellation:
    """Pending deltas never leak past terminal error frames."""

    @pytest.mark.asyncio
    async def test_cancel_drops_pending_deltas(self) -> None:
        """Cancelling mid-stream drops the buffer and emits no trailing delta."""
        global _gate
        _gate = asyncio.Event()
        instance, recorder = runner()
        RUNNERS.append(instance)

        try:
            with patch("chibi.runners.ide_transport.handle_user_prompt", fake_gated_prompt):
                await instance._handle_message(initialize({"streaming": True}))
                await instance._handle_message(request("r1", 42))
                await wait_for_frame(recorder.frames, "r1", "status")
                await instance._handle_message({"type": "cancel", "request_id": "r1"})
                await wait_for_frame(recorder.frames, "r1", "error")
                await asyncio.sleep(DELTA_FLUSH_INTERVAL + 0.1)

            error_frame = [frame for frame in recorder.frames if frame.get("type") == "error"][0]
            assert error_frame["code"] == "cancelled"
            assert deltas(recorder.frames) == []
        finally:
            _gate.set()

    @pytest.mark.asyncio
    async def test_dropped_state_blocks_late_emission(self) -> None:
        """After retirement, a late chunk from the interface is a no-op."""
        global _gate
        _gate = asyncio.Event()
        _gate.set()
        instance, recorder = runner()
        RUNNERS.append(instance)

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_gated_prompt):
            await instance._handle_message(initialize({"streaming": True}))
            await instance._handle_message(request("r1", 42))
            await wait_for_frame(recorder.frames, "r1", "result")
            before = len(deltas(recorder.frames))
            instance._drop_deltas("r1")
            await instance._emit_delta("r1", 42, "late chunk")
            await asyncio.sleep(DELTA_FLUSH_INTERVAL + 0.1)

        assert before == 1
        assert len(deltas(recorder.frames)) == before


class TestInterfaceContract:
    """IDEInterface delta forwarding and closed-interface behavior."""

    @pytest.mark.asyncio
    async def test_interface_forwards_chunks_to_transport(self) -> None:
        """A streaming interface awaits the delta_emit callback inline."""
        capture = DeltaCapture()
        interface = IDEInterface(42, "p", {}, lambda text: None, delta_emit=capture)

        assert interface.streaming_enabled is True
        assert interface.delta_emitted is False
        await interface.send_delta("chunk")
        await interface.send_delta("chunk")

        assert capture.chunks == ["chunk", "chunk"]

    @pytest.mark.asyncio
    async def test_interface_without_callback_is_not_streaming(self) -> None:
        """A non-streaming interface exposes the flag and drops chunks."""
        interface = IDEInterface(42, "p", {}, lambda text: None)

        assert interface.streaming_enabled is False
        await interface.send_delta("ignored")

    @pytest.mark.asyncio
    async def test_closed_interface_drops_deltas_silently(self) -> None:
        """Chunks after mark_closed never reach the transport callback."""
        capture = DeltaCapture()
        interface = IDEInterface(42, "p", {}, lambda text: None, delta_emit=capture)
        interface.mark_closed()

        await interface.send_delta("late")

        assert capture.chunks == []

    @pytest.mark.asyncio
    async def test_closed_interface_never_routes_to_background(self) -> None:
        """A closed interface drops deltas instead of emitting message frames."""
        capture = DeltaCapture()
        background = BackgroundCapture()
        interface = IDEInterface(42, "p", {}, lambda text: None, background_emit=background, delta_emit=capture)
        interface.mark_closed()

        await interface.send_delta("late")

        assert capture.chunks == []
        assert background.payloads == []


class TestUserInterfaceDefault:
    """The shared UserInterface base keeps streaming opt-in off."""

    @pytest.mark.asyncio
    async def test_base_class_send_delta_is_noop(self) -> None:
        """Runners without streaming support inherit a silent no-op."""

        class BareInterface(UserInterface):
            """Minimal concrete UserInterface for the default contract."""

        interface = BareInterface()
        await interface.send_delta("anything")
