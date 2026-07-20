"""Contract coverage through the real ``chibi ide --stdio`` command."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import textwrap
import time
from typing import Any

_BOOTSTRAP = textwrap.dedent(
    """
    import asyncio

    from chibi.schemas.app import ModelChangeSchema
    import chibi.runners.ide_transport as transport

    MODELS = [ModelChangeSchema(provider="fake", name="fake-model", display_name="Fake Model", image_generation=False)]

    async def prompt(interface):
        value = await interface.get_text_prompt()
        interface.response_model = "fake-model"
        interface.response_provider = "fake"
        if value and value.startswith("slow"):
            await asyncio.sleep(0.3)
        await interface.send_message("deterministic response")

    async def models(user_id, thread_id=0, **kwargs):
        return MODELS

    async def reset(interface):
        await interface.send_message("reset ok")

    async def imagine(prompt, interface):
        await interface.send_images(["fake.png"])

    async def select_model(interface, model):
        return None

    async def info(user_id):
        return "fake info"

    transport.handle_user_prompt = prompt
    transport.get_models_available = models
    transport.handle_reset = reset
    transport.handle_image_generation = imagine
    transport.set_active_model = select_model
    transport.get_info = info

    from chibi.cli import main

    main(["ide", "--stdio"])
    """
)


class ProtocolProcess:
    """Manage one real CLI process and its unbuffered JSONL stream."""

    def __init__(self) -> None:
        """Start a deterministic ``chibi ide --stdio`` process."""
        self.process = subprocess.Popen(
            [sys.executable, "-c", _BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self.process.stdout is not None
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        self._buffer = b""
        self._frames: list[dict[str, Any]] = []

    def send(self, message: dict[str, Any] | str) -> None:
        """Send one JSONL client message.

        Args:
            message: Structured protocol message or intentionally malformed text.
        """
        assert self.process.stdin is not None
        encoded = message if isinstance(message, str) else json.dumps(message)
        self.process.stdin.write((encoded + "\n").encode())
        self.process.stdin.flush()

    def wait_for(
        self,
        frame_type: str,
        request_id: str | None = None,
        state: str | None = None,
        timeout: float = 12.0,
    ) -> dict[str, Any]:
        """Return and consume the first matching correlated frame.

        Args:
            frame_type: Required protocol frame type.
            request_id: Required request identifier, or None for a global frame.
            state: Optional required status state.
            timeout: Maximum wall-clock wait in seconds.

        Returns:
            The matching decoded protocol frame.

        Raises:
            AssertionError: If no matching frame arrives before timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            for index, frame in enumerate(self._frames):
                if (
                    frame.get("type") == frame_type
                    and (request_id is None or frame.get("request_id") == request_id)
                    and (state is None or frame.get("state") == state)
                ):
                    return self._frames.pop(index)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise AssertionError(
                    f"timed out waiting for type={frame_type}, request_id={request_id}, "
                    f"state={state}; buffered={self._frames}; stderr={self._stderr()}"
                )
            assert self.process.stdout is not None
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise AssertionError(f"IDE subprocess closed stdout; stderr={self._stderr()}")
            self._buffer += chunk
            lines = self._buffer.split(b"\n")
            self._buffer = lines.pop()
            self._frames.extend(json.loads(line) for line in lines if line)

    def shutdown(self) -> None:
        """Request graceful shutdown and verify successful process exit."""
        self.send({"type": "shutdown"})
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait(timeout=12)
        assert self.process.returncode == 0, self._stderr()

    def close(self) -> None:
        """Release selector resources and terminate a surviving process."""
        self._selector.close()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)

    def _stderr(self) -> str:
        """Return currently available stderr after process exit.

        Returns:
            Decoded stderr, or a marker while the process remains active.
        """
        if self.process.poll() is None:
            return "<process still running>"
        assert self.process.stderr is not None
        return self.process.stderr.read().decode(errors="replace")


def request(request_id: str, thread_id: int, prompt: str) -> dict[str, Any]:
    """Build a valid protocol request.

    Args:
        request_id: Correlation identifier.
        thread_id: IDE conversation identifier.
        prompt: Prompt or slash command.

    Returns:
        A valid request frame.
    """
    return {
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


def initialized_process() -> ProtocolProcess:
    """Start and initialize a protocol process.

    Returns:
        An initialized real CLI process.
    """
    client = ProtocolProcess()
    client.send({"type": "initialize", "protocol_version": 1})
    assert client.wait_for("ready")["protocol_version"] == 1
    return client


def test_real_cli_handshake_request_result_error_and_model() -> None:
    """Verify handshake, request lifecycle, errors, and ``/model``."""
    client = initialized_process()
    try:
        client.send(request("ok", 1, "hello"))
        assert client.wait_for("status", "ok")["state"] == "running"
        result = client.wait_for("result", "ok")
        assert result["content"] == "deterministic response"
        assert result["model"] == "fake-model"
        assert result["provider"] == "fake"
        client.send(request("model", 1, "/model"))
        assert "Fake Model" in client.wait_for("result", "model")["content"]
        client.send(request("bad", 1, "/unknown"))
        assert client.wait_for("error", "bad")["code"] == "request_failed"
        client.shutdown()
    finally:
        client.close()


def test_real_cli_different_threads_progress_concurrently() -> None:
    """Verify two threads reach running before either slow result completes."""
    client = initialized_process()
    try:
        client.send(request("d1", 10, "slow one"))
        client.send(request("d2", 11, "slow two"))
        assert client.wait_for("status", "d1", "running")["state"] == "running"
        assert client.wait_for("status", "d2", "running")["state"] == "running"
        assert client.wait_for("result", "d1")["content"] == "deterministic response"
        assert client.wait_for("result", "d2")["content"] == "deterministic response"
        client.shutdown()
    finally:
        client.close()


def test_real_cli_same_thread_queues_then_runs() -> None:
    """Verify a same-thread request transitions from queued to running."""
    client = initialized_process()
    try:
        client.send(request("q1", 20, "slow first"))
        client.send(request("q2", 20, "slow second"))
        assert client.wait_for("status", "q1", "running")["state"] == "running"
        assert client.wait_for("status", "q2", "queued")["state"] == "queued"
        assert client.wait_for("result", "q1")["content"] == "deterministic response"
        assert client.wait_for("status", "q2", "running")["state"] == "running"
        assert client.wait_for("result", "q2")["content"] == "deterministic response"
        client.shutdown()
    finally:
        client.close()


def test_real_cli_targeted_cancel() -> None:
    """Verify cancellation affects only the correlated request."""
    client = initialized_process()
    try:
        client.send(request("cancelled", 30, "slow cancel me"))
        client.send(request("survivor", 31, "hello"))
        assert client.wait_for("status", "cancelled", "running")["state"] == "running"
        assert client.wait_for("status", "survivor", "running")["state"] == "running"
        client.send({"type": "cancel", "request_id": "cancelled"})
        assert client.wait_for("error", "cancelled")["code"] == "cancelled"
        assert client.wait_for("result", "survivor")["content"] == "deterministic response"
        client.shutdown()
    finally:
        client.close()


def test_real_cli_malformed_input_and_shutdown() -> None:
    """Verify malformed input is recoverable and shutdown is graceful."""
    client = initialized_process()
    try:
        client.send("not valid json")
        malformed = client.wait_for("error")
        assert malformed["code"] == "malformed_request"
        assert malformed["request_id"] is None
        client.shutdown()
    finally:
        client.close()


def test_real_cli_quit_always_returns_terminal_frame() -> None:
    """Stress /quit so the terminal result frame is always observed before exit."""
    for _ in range(20):
        client = initialized_process()
        try:
            client.send(request("quit", 1, "/quit"))
            assert client.wait_for("status", "quit", "running")["state"] == "running"
            result = client.wait_for("result", "quit")
            assert result["content"] == "Bye!"
            assert client.process is not None
            assert client.process.wait(timeout=12) == 0, client._stderr()
        finally:
            client.close()
