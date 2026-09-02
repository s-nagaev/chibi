"""Background message frame coverage for the IDE stdio transport."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import COMMANDS, PROTOCOL_VERSION, IDEInterface, IDEStdioRunner
from chibi.schemas.app import ChatResponseSchema
from chibi.services.bot import handle_tool_response
from chibi.services.providers.tools.schemas import ToolResponseSchema

READY_FRAME: dict[str, Any] = {
    "type": "ready",
    "protocol_version": PROTOCOL_VERSION,
    "server": {"name": "chibi", "version": "1.0.0"},
    "capabilities": {"commands": COMMANDS},
}

BACKGROUND_TASKS: list[asyncio.Task[None]] = []


class OutputRecorder:
    """Capture protocol frames through an async callable."""

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
            message: Emitted JSON-compatible protocol frame.
        """
        if message.get("type") in self._fail_on_types:
            raise RuntimeError("stdout pipe is broken")
        self.frames.append(message)


def runner(fail_on_types: set[str] | None = None) -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    Args:
        fail_on_types: Frame types whose writes must raise.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder(fail_on_types=fail_on_types)
    instance.__dict__["_write"] = recorder
    return instance, recorder.frames


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


def deliver_later(content: str, model: str | None = None, provider: str | None = None) -> Any:
    """Build a fake prompt handler that delivers an answer after the request closes.

    Args:
        content: Background answer text.
        model: Optional model name attached to the delivery.
        provider: Optional provider name attached to the delivery.

    Returns:
        An async handler compatible with the patched ``handle_user_prompt``.
    """

    async def fake_prompt(interface: IDEInterface) -> None:
        async def deliver() -> None:
            await wait_until_closed(interface)
            await interface.send_tool_answer(content, model=model, provider=provider)

        task = asyncio.create_task(deliver())
        BACKGROUND_TASKS.append(task)
        await interface.send_message("Foreground answer")

    return fake_prompt


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


@pytest.fixture(autouse=True)
async def background_task_cleanup() -> Any:
    """Await and clear spawned background tasks after each test."""
    yield
    tasks, BACKGROUND_TASKS[:] = BACKGROUND_TASKS[:], []
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class TestHandshake:
    """Handshake handling of the optional capabilities object."""

    @pytest.mark.asyncio
    async def test_background_messages_capability_is_stored(self) -> None:
        """Declaring background_messages enables the session flag."""
        instance, output = runner()

        await instance._handle_message(initialize({"background_messages": True}))

        assert instance._background_messages_enabled is True
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_ready_frame_unchanged_without_capabilities(self) -> None:
        """Clients that never send capabilities get the standard ready frame."""
        instance, output = runner()

        await instance._handle_message(initialize())

        assert instance._background_messages_enabled is False
        assert output[0] == READY_FRAME

    @pytest.mark.asyncio
    async def test_malformed_capabilities_do_not_enable_emission(self) -> None:
        """Non-object capabilities payloads keep emission off."""
        instance, output = runner()

        await instance._handle_message(initialize("not-an-object"))

        assert instance._background_messages_enabled is False
        assert output[0] == READY_FRAME


class TestBackgroundMessageFrames:
    """Emission of out-of-band message frames for background tool answers."""

    @pytest.mark.asyncio
    async def test_frame_emitted_with_thread_id_and_metadata(self) -> None:
        """A background answer after request completion becomes a message frame."""
        instance, output = runner()
        with patch(
            "chibi.runners.ide_transport.handle_user_prompt", deliver_later("Background answer", "gpt-x", "prov")
        ):
            await instance._handle_message(initialize({"background_messages": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "message")

        frame = next(f for f in output if f.get("type") == "message")
        assert frame == {
            "type": "message",
            "thread_id": 42,
            "content": "Background answer",
            "model": "gpt-x",
            "provider": "prov",
        }

    @pytest.mark.asyncio
    async def test_no_frame_without_capability(self) -> None:
        """Clients that did not declare the capability never see message frames."""
        instance, output = runner()
        with patch("chibi.runners.ide_transport.handle_user_prompt", deliver_later("Background answer")):
            await instance._handle_message(initialize())
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        assert not any(f.get("type") == "message" for f in output)

    @pytest.mark.asyncio
    async def test_optional_fields_omitted_when_unknown(self) -> None:
        """model and provider are dropped from the frame when not provided."""
        instance, output = runner()
        with patch("chibi.runners.ide_transport.handle_user_prompt", deliver_later("Background answer")):
            await instance._handle_message(initialize({"background_messages": True}))
            await run_request(instance, "r1", 7)
            await wait_for_frame(output, "message")

        frame = next(f for f in output if f.get("type") == "message")
        assert frame == {"type": "message", "thread_id": 7, "content": "Background answer"}


class TestRequestScopedContractUnchanged:
    """Foreground request behavior stays identical while the capability is on."""

    @pytest.mark.asyncio
    async def test_foreground_answer_still_lands_in_result(self) -> None:
        """The result frame still correlates the foreground answer by request_id."""
        instance, output = runner()
        with patch(
            "chibi.runners.ide_transport.handle_user_prompt", deliver_later("Background answer", "gpt-x", "prov")
        ):
            await instance._handle_message(initialize({"background_messages": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")

        result = next(f for f in output if f.get("type") == "result")
        assert result == {"type": "result", "request_id": "r1", "content": "Foreground answer"}
        statuses = [f for f in output if f.get("type") == "status"]
        assert statuses[0] == {"type": "status", "request_id": "r1", "state": "running"}


class TestEmissionFailureResilience:
    """A broken stdout during emission must not kill the background task."""

    @pytest.mark.asyncio
    async def test_write_failure_is_dropped_without_raising(self) -> None:
        """A failing message frame write raises nothing at the delivery site."""
        instance, _output = runner(fail_on_types={"message"})
        responses: list[str] = []

        async def emit_background(thread_id: int, content: str, model: str | None, provider: str | None) -> None:
            await instance._emit_background_message(thread_id, content, model, provider)

        interface = IDEInterface(3, "p", {}, responses.append, background_emit=emit_background)
        interface.mark_closed()

        await interface.send_tool_answer("Answer that will be dropped", model="m", provider="p")

        assert responses == []

    @pytest.mark.asyncio
    async def test_background_task_survives_write_failure(self) -> None:
        """The background delivery task completes even when the write fails."""
        instance, output = runner(fail_on_types={"message"})
        with patch("chibi.runners.ide_transport.handle_user_prompt", deliver_later("Doomed answer", "m", "p")):
            await instance._handle_message(initialize({"background_messages": True}))
            await run_request(instance, "r1", 42)
            await wait_for_frame(output, "result")
            for task in list(BACKGROUND_TASKS):
                await asyncio.gather(task)

        assert not any(f.get("type") == "message" for f in output)


class TestHandleToolResponsePlumbing:
    """handle_tool_response routes answers through send_tool_answer with metadata."""

    @pytest.mark.asyncio
    async def test_closed_interface_emits_frame_with_model_and_provider(self) -> None:
        """A closed IDE interface delivers the tool answer as a background frame."""
        instance, output = runner()
        emitted: list[tuple[int, str, str | None, str | None]] = []

        async def emit_background(thread_id: int, content: str, model: str | None, provider: str | None) -> None:
            emitted.append((thread_id, content, model, provider))
            await instance._emit_background_message(thread_id, content, model, provider)

        interface = IDEInterface(11, "p", {}, lambda _: None, background_emit=emit_background)
        interface.mark_closed()
        chat_response = ChatResponseSchema(answer="Tool answer", provider="prov", model="m", usage=None)

        with patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)):
            await handle_tool_response(
                tool_response=ToolResponseSchema(tool_name="t", status="ok", result="ok"), interface=interface
            )

        assert output == [
            {"type": "message", "thread_id": 11, "content": "Tool answer", "model": "m", "provider": "prov"}
        ]
        assert emitted == [(11, "Tool answer", "m", "prov")]

    @pytest.mark.asyncio
    async def test_open_interface_appends_to_request_responses(self) -> None:
        """An open request still collects the tool answer in its response buffer."""
        responses: list[str] = []
        interface = IDEInterface(11, "p", {}, responses.append)
        chat_response = ChatResponseSchema(answer="Tool answer", provider="prov", model="m", usage=None)

        with patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)):
            await handle_tool_response(
                tool_response=ToolResponseSchema(tool_name="t", status="ok", result="ok"), interface=interface
            )

        assert responses == ["Tool answer"]
