"""thread_renamed frame coverage for the IDE stdio transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.config import application_settings
from chibi.runners.ide_transport import COMMANDS, PROTOCOL_VERSION, IDEInterface, IDEStdioRunner
from chibi.services.thread_name_events import thread_name_tracker

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS},
}

STORED_NAME = "Refactor auth module"


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


def thread_renamed_frames(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the thread_renamed frames from the captured output.

    Args:
        output: Captured frame list.

    Returns:
        The thread_renamed frames in wire order.
    """
    return [frame for frame in output if frame.get("type") == "thread_renamed"]


async def wait_for_results(output: list[dict[str, Any]], request_ids: list[str]) -> None:
    """Wait until every listed request has emitted its result frame.

    Args:
        output: Captured frame list.
        request_ids: Request identifiers that must all have result frames.
    """
    pending = set(request_ids)
    for _ in range(300):
        answered = {
            frame["request_id"] for frame in output if frame.get("type") == "result" and frame.get("request_id")
        }
        if pending <= answered:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Expected result frames for {sorted(pending)}: {output}")


def frame_of_type(output: list[dict[str, Any]], frame_type: str) -> dict[str, Any]:
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


RUNNERS: list[IDEStdioRunner] = []


@pytest.fixture(autouse=True)
async def thread_name_event_cleanup() -> Any:
    """Release the tracker sink, settle emission tasks and reap request tasks."""
    yield
    thread_name_tracker.set_sink(None)
    for instance in RUNNERS:
        leaked = [task for task in instance._tasks.values() if not task.done()]
        for task in leaked:
            task.cancel()
        if leaked:
            await asyncio.gather(*leaked, return_exceptions=True)
        await instance.drain_thread_name_events()
    RUNNERS.clear()


def patched_save_thread_name() -> Any:
    """Patch thread-name persistence with a recording mock.

    Returns:
        The patch context manager.
    """
    return patch("chibi.runners.ide_transport.save_thread_name", new=AsyncMock(return_value=None))


def patched_stored_name(value: str | None) -> Any:
    """Patch the transport's stored-name resolution to a fixed value.

    Args:
        value: The stored name to report (None for threads without a name).

    Returns:
        The patch context manager.
    """
    return patch("chibi.runners.ide_transport.get_thread_name", new=AsyncMock(return_value=value))


class TestHandshake:
    """Capability parsing for the thread_renames opt-in flag."""

    @pytest.mark.asyncio
    async def test_thread_renames_capability_enables_emission(self) -> None:
        """Declaring thread_renames enables the session flag and binds the sink."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize({"thread_renames": True}))

        assert instance._thread_renames_enabled is True
        assert thread_name_tracker._sink is instance
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_v1_client_without_capability_never_binds_sink(self) -> None:
        """Clients that never send capabilities get no thread-name sink and no frames."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize())

        assert instance._thread_renames_enabled is False
        assert thread_name_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_malformed_capabilities_do_not_enable_emission(self) -> None:
        """Non-object capabilities payloads keep emission off."""
        instance, output = runner()
        RUNNERS.append(instance)

        await instance._handle_message(initialize("not-an-object"))

        assert instance._thread_renames_enabled is False
        assert thread_name_tracker._sink is None
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_later_plain_session_releases_the_sink(self) -> None:
        """A session without the opt-in replaces a previously bound sink."""
        first, _ = runner()
        second, _ = runner()
        RUNNERS.extend([first, second])

        await first._handle_message(initialize({"thread_renames": True}))
        await second._handle_message(initialize())

        assert first._thread_renames_enabled is True
        assert second._thread_renames_enabled is False
        assert thread_name_tracker._sink is None

    @pytest.mark.asyncio
    async def test_run_shutdown_releases_the_sink(self) -> None:
        """The run() finally block releases the tracker sink on EOF."""
        instance, _ = runner()
        lines = [json.dumps(initialize({"thread_renames": True})), ""]
        object.__setattr__(instance, "_read_line", AsyncMock(side_effect=lines))
        original_gate = application_settings.scheduler_tool_enabled
        application_settings.scheduler_tool_enabled = False

        try:
            with patch("chibi.runners.ide_transport.task_manager") as task_manager_mock:
                task_manager_mock.shutdown = AsyncMock()
                await instance.run()
        finally:
            application_settings.scheduler_tool_enabled = original_gate

        assert thread_name_tracker._sink is None


class TestRenameEmission:
    """Rename frames fired through IDEInterface.rename_thread."""

    @pytest.mark.asyncio
    async def test_rename_thread_emits_frame_before_result(self) -> None:
        """A rename inside a request emits the frame ahead of the result frame."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            renamed = await interface.rename_thread(STORED_NAME)
            await interface.send_message("answer")
            assert renamed is True

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_save_thread_name() as save_mock,
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        save_mock.assert_awaited_once()
        assert save_mock.await_args.kwargs["thread_id"] == 42
        assert save_mock.await_args.kwargs["name"] == STORED_NAME
        frames = thread_renamed_frames(output)
        assert frames == [{"type": "thread_renamed", "thread_id": 42, "name": STORED_NAME}]
        assert output.index(frames[0]) < output.index(frame_of_type(output, "result"))

    @pytest.mark.asyncio
    async def test_rename_without_opt_in_saves_but_emits_nothing(self) -> None:
        """Without the capability the name persists but no frame reaches the wire."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            renamed = await interface.rename_thread(STORED_NAME)
            await interface.send_message("answer")
            assert renamed is True

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_save_thread_name() as save_mock,
        ):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        save_mock.assert_awaited_once()
        assert thread_renamed_frames(output) == []

    @pytest.mark.asyncio
    async def test_tracker_fire_without_rename_thread_still_emits(self) -> None:
        """A direct tracker fire (any future rename path) emits the frame."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            thread_name_tracker.thread_renamed(42, "Direct fire")
            await asyncio.sleep(0)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        assert thread_renamed_frames(output) == [{"type": "thread_renamed", "thread_id": 42, "name": "Direct fire"}]

    @pytest.mark.asyncio
    async def test_tracker_fire_without_sink_is_silent(self) -> None:
        """No session opted in: the tracker has no sink and nothing is emitted."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            thread_name_tracker.thread_renamed(42, "Ghost rename")
            await asyncio.sleep(0)
            await interface.send_message("answer")

        with patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        assert thread_name_tracker._sink is None
        assert thread_renamed_frames(output) == []


class TestRenameValidation:
    """Name validation at the rename fire site."""

    @pytest.mark.asyncio
    async def test_empty_name_is_rejected_without_persisting(self) -> None:
        """A whitespace-only name returns False and never persists or emits."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            renamed = await interface.rename_thread("   ")
            await interface.send_message("answer")
            assert renamed is False

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_save_thread_name() as save_mock,
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        save_mock.assert_not_awaited()
        assert thread_renamed_frames(output) == []

    @pytest.mark.asyncio
    async def test_overlong_name_is_rejected(self) -> None:
        """A name longer than 128 characters returns False and never persists."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            renamed = await interface.rename_thread("x" * 129)
            await interface.send_message("answer")
            assert renamed is False

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_save_thread_name() as save_mock,
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        save_mock.assert_not_awaited()
        assert thread_renamed_frames(output) == []

    @pytest.mark.asyncio
    async def test_name_is_trimmed_before_persist_and_emit(self) -> None:
        """Surrounding whitespace is trimmed; 128 characters is accepted."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            renamed = await interface.rename_thread(f"  {'y' * 128}  ")
            await interface.send_message("answer")
            assert renamed is True

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_save_thread_name() as save_mock,
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])
            await instance.drain_thread_name_events()

        assert save_mock.await_args.kwargs["name"] == "y" * 128
        assert thread_renamed_frames(output) == [{"type": "thread_renamed", "thread_id": 42, "name": "y" * 128}]


class TestInitialSync:
    """One-time per-thread stored-name sync on the first request."""

    @pytest.mark.asyncio
    async def test_first_request_emits_stored_name_before_result(self) -> None:
        """The first request on a thread emits exactly one thread_renamed pre-result."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_stored_name(STORED_NAME),
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])

        frames = thread_renamed_frames(output)
        assert frames == [{"type": "thread_renamed", "thread_id": 42, "name": STORED_NAME}]
        assert output.index(frames[0]) < output.index(frame_of_type(output, "result"))

    @pytest.mark.asyncio
    async def test_second_request_on_same_thread_does_not_re_emit(self) -> None:
        """The per-thread sync happens once; later requests emit nothing extra."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_stored_name(STORED_NAME),
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await run_request(instance, "r2", 42)
            await wait_for_results(output, ["r1", "r2"])

        assert len(thread_renamed_frames(output)) == 1

    @pytest.mark.asyncio
    async def test_second_thread_gets_its_own_sync(self) -> None:
        """A different thread triggers its own one-time thread_renamed."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_stored_name(STORED_NAME),
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await run_request(instance, "r2", 7)
            await wait_for_results(output, ["r1", "r2"])

        assert thread_renamed_frames(output) == [
            {"type": "thread_renamed", "thread_id": 42, "name": STORED_NAME},
            {"type": "thread_renamed", "thread_id": 7, "name": STORED_NAME},
        ]

    @pytest.mark.asyncio
    async def test_thread_without_stored_name_is_a_silent_no_op(self) -> None:
        """A thread with no stored name emits nothing: unknown ids drop silently."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_stored_name(None),
        ):
            await instance._handle_message(initialize({"thread_renames": True}))
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])

        assert thread_renamed_frames(output) == []

    @pytest.mark.asyncio
    async def test_disabled_capability_emits_nothing(self) -> None:
        """Old clients (no opt-in) see zero thread_renamed frames."""
        instance, output = runner()
        RUNNERS.append(instance)

        async def fake_prompt(interface: IDEInterface) -> None:
            await interface.send_message("answer")

        with (
            patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
            patched_stored_name(STORED_NAME),
        ):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_results(output, ["r1"])

        assert thread_renamed_frames(output) == []
