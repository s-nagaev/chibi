"""cwd_update frame coverage for the IDE stdio transport."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import COMMANDS, PROTOCOL_VERSION, IDEInterface, IDEStdioRunner
from chibi.services.cwd_events import cwd_tracker

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS},
}

LAUNCH_CWD = "/Users/dev/launch"
CHANGED_CWD = "/Users/dev/other"


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


def runner() -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    The recorder replaces ``_write_line`` so both regular writes and
    fire-and-forget emissions are captured.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write_line"] = recorder
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


def cwd_updates(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the cwd_update frames from the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The cwd_update frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == "cwd_update"]


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


async def wait_for_results(output: list[dict[str, Any]], count: int) -> None:
    """Wait until ``count`` result frames have been emitted.

    Args:
        output: Captured frame list.
        count: Number of terminal result frames to wait for.
    """
    for _ in range(300):
        if len([frame for frame in output if frame.get("type") == "result"]) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Expected {count} result frames: {output}")


RUNNERS: list[IDEStdioRunner] = []


@pytest.fixture(autouse=True)
async def cwd_event_cleanup() -> Any:
    """Release the tracker sink and settle emission tasks after each test."""
    yield
    cwd_tracker.set_sink(None)
    for instance in RUNNERS:
        await instance.drain_cwd_updates()
    RUNNERS.clear()


def patched_cwd(*values: str) -> Any:
    """Patch the transport's cwd resolution to fixed sequential values.

    Args:
        values: The effective working directories the resolution reports,
            one per call in call order.

    Returns:
        The patch context manager.
    """
    mock = AsyncMock(return_value=values[0])
    if len(values) > 1:
        mock.side_effect = list(values)
    return patch("chibi.runners.ide_transport._get_thread_cwd", new=mock)


class TestHandshake:
    """Capability parsing for the cwd_updates opt-in flag."""

    @pytest.mark.asyncio
    async def test_cwd_updates_capability_enables_emission(self) -> None:
        """Declaring cwd_updates enables the session flag and binds the sink."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize({"cwd_updates": True}))

        assert instance._cwd_updates_enabled is True
        assert cwd_tracker._sink is instance
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_v1_client_without_capability_never_binds_sink(self) -> None:
        """Clients that never send capabilities get no cwd sink and no frames."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize())

        assert instance._cwd_updates_enabled is False
        assert cwd_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_malformed_capabilities_do_not_enable_emission(self) -> None:
        """Non-object capabilities payloads keep emission off."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize("not-an-object"))

        assert instance._cwd_updates_enabled is False
        assert cwd_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_later_plain_session_releases_the_sink(self) -> None:
        """A session without the opt-in replaces a previously bound sink."""
        first, _ = runner()
        second, _ = runner()
        RUNNERS.extend([first, second])

        await first._handle_message(initialize({"cwd_updates": True}))
        await second._handle_message(initialize())

        assert first._cwd_updates_enabled is True
        assert second._cwd_updates_enabled is False
        assert cwd_tracker._sink is None


class TestFirstRequestEmission:
    """One-time per-thread cwd sync on the first validated request."""

    @pytest.mark.asyncio
    async def test_first_request_emits_cwd_before_result(self) -> None:
        """The first request on a thread emits exactly one cwd_update pre-result."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(LAUNCH_CWD):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        assert cwd_updates(output) == [{"type": "cwd_update", "thread_id": 42, "cwd": LAUNCH_CWD}]

    @pytest.mark.asyncio
    async def test_second_request_on_same_thread_does_not_re_emit(self) -> None:
        """The per-thread sync happens once; later requests emit nothing extra."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(LAUNCH_CWD):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await run_request(instance, "r2", 42)
            await wait_for_frame(output, "result")

        assert len(cwd_updates(output)) == 1

    @pytest.mark.asyncio
    async def test_second_thread_gets_its_own_sync(self) -> None:
        """A different thread triggers its own one-time cwd_update."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(LAUNCH_CWD):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await run_request(instance, "r1", 42)
            await run_request(instance, "r2", 7)
            await wait_for_results(output, 2)

        assert cwd_updates(output) == [
            {"type": "cwd_update", "thread_id": 42, "cwd": LAUNCH_CWD},
            {"type": "cwd_update", "thread_id": 7, "cwd": LAUNCH_CWD},
        ]

    @pytest.mark.asyncio
    async def test_disabled_capability_emits_nothing(self) -> None:
        """Old clients (no opt-in) see zero cwd_update frames."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(LAUNCH_CWD):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        assert cwd_updates(output) == []


class TestRuntimeChange:
    """Runtime cwd changes reported through the tracker sink."""

    @pytest.mark.asyncio
    async def test_set_working_dir_tool_change_is_emitted(self) -> None:
        """A runtime change via the tracker emits a cwd_update with the new value."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            cwd_tracker.cwd_changed(42)
            await asyncio.sleep(0)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(LAUNCH_CWD, CHANGED_CWD):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await instance.drain_cwd_updates()

        assert cwd_updates(output) == [
            {"type": "cwd_update", "thread_id": 42, "cwd": LAUNCH_CWD},
            {"type": "cwd_update", "thread_id": 42, "cwd": CHANGED_CWD},
        ]

    @pytest.mark.asyncio
    async def test_change_without_opt_in_is_silent(self) -> None:
        """Without the capability the tracker has no sink and nothing is emitted."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            cwd_tracker.cwd_changed(42)
            await asyncio.sleep(0)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt), patched_cwd(CHANGED_CWD):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            await instance.drain_cwd_updates()

        assert cwd_tracker._sink is None
        assert cwd_updates(output) == []


class TestGetCwd:
    """The get_cwd catch-up frame."""

    @pytest.mark.asyncio
    async def test_get_cwd_answers_with_cwd_update(self) -> None:
        """An opted-in client resyncs on demand."""
        instance, output = runner()
        RUNNERS.append(instance)

        with patched_cwd(LAUNCH_CWD):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await instance._handle_message({"type": "get_cwd", "thread_id": 42})

        assert cwd_updates(output) == [{"type": "cwd_update", "thread_id": 42, "cwd": LAUNCH_CWD}]

    @pytest.mark.asyncio
    async def test_get_cwd_without_capability_is_unknown_message(self) -> None:
        """Old sessions get the canonical unknown_message error."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize())
        await instance._handle_message({"type": "get_cwd", "thread_id": 42})

        error = wait_for_type(output, "error")
        assert error["code"] == "unknown_message"
        assert cwd_updates(output) == []

    @pytest.mark.asyncio
    async def test_get_cwd_before_initialize_is_rejected(self) -> None:
        """get_cwd before the handshake is a not_initialized error."""
        instance, output = runner()

        await instance._handle_message({"type": "get_cwd", "thread_id": 42})

        error = wait_for_type(output, "error")
        assert error["code"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_get_cwd_with_bad_thread_id_is_rejected(self) -> None:
        """A missing or negative thread_id is a malformed_request error."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize({"cwd_updates": True}))
        await instance._handle_message({"type": "get_cwd"})
        await instance._handle_message({"type": "get_cwd", "thread_id": -1})

        errors = [frame for frame in output if frame.get("type") == "error"]
        assert [error["code"] for error in errors] == ["malformed_request", "malformed_request"]
        assert cwd_updates(output) == []


class TestCloneEmission:
    """Thread cloning reports the destination thread's inherited cwd."""

    @pytest.mark.asyncio
    async def test_clone_emits_destination_cwd(self) -> None:
        """A successful clone emits a cwd_update for the destination thread."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_thread_messages(storage_id: int, thread_id: int) -> list[Any]:
            if thread_id == 7:
                return [object()]
            return []

        async def fake_clone(storage_id: int, old_thread_id: int, new_thread_id: int, name: Any = None) -> int:
            return 1

        with (
            patched_cwd(CHANGED_CWD),
            patch("chibi.runners.ide_transport._get_thread_messages", new=fake_thread_messages),
            patch("chibi.runners.ide_transport.clone_thread_messages", new=fake_clone),
        ):
            await instance._handle_message(initialize({"cwd_updates": True}))
            await run_request(instance, "r1", 42, prompt="/new_thread_with_current_context 7")
            await wait_for_results(output, 1)

        clone_updates = [
            frame for frame in cwd_updates(output) if frame["thread_id"] == 42 and frame["cwd"] == CHANGED_CWD
        ]
        assert clone_updates, "clone must emit a cwd_update for the destination thread"


def wait_for_type(output: list[dict[str, Any]], frame_type: str) -> dict[str, Any]:
    """Return the first frame of the given type or fail.

    Args:
        output: Captured frame list.
        frame_type: Frame type to look for.

    Returns:
        The first matching frame.
    """
    for frame in output:
        if frame.get("type") == frame_type:
            return frame
    raise AssertionError(f"Missing {frame_type} frame: {output}")
