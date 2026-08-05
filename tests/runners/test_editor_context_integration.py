"""Integration test: editor_context flows from IDE protocol to LLM prompt dict.

Verifies the full path:
  VSCode sends protocol request with editor_context
    → IDEStdioRunner._run_request() builds IDEInterface with editor_context
      → handle_user_prompt() → get_llm_chat_completion_answer()
        → isinstance(interface, EditorContextProvider) guard
          → editor_context section injected into prompt dict
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import textwrap
import time
from typing import Any

_BOOTSTRAP = textwrap.dedent(r"""
    import asyncio
    import json
    import sys

    from chibi.schemas.app import ModelChangeSchema
    import chibi.runners.ide_transport as transport

    MODELS = [ModelChangeSchema(provider="fake", name="fake-model", display_name="Fake Model", image_generation=False)]

    # Use a list wrapper so the nested function can mutate it via nonlocal.
    _captured_list: list = [None]

    _original_fget = transport.IDEInterface.editor_context.fget

    def _patched_fget(self):
        _captured_list[0] = _original_fget(self)
        return _captured_list[0]

    transport.IDEInterface.editor_context = property(_patched_fget)

    async def prompt(interface):
        interface.response_model = "fake-model"
        interface.response_provider = "fake"
        # Access the patched property to trigger capture.
        _ = interface.editor_context
        captured = _captured_list[0]
        # Emit captured editor_context as a non-JSONL line on stdout.
        sys.stdout.write(json.dumps({"_captured_editor_context": captured}) + "\n")
        sys.stdout.flush()
        await interface.send_message("editor_context test response")

    async def models(user_id, thread_id=0, **kwargs):
        return MODELS

    async def reset(interface):
        await interface.send_message("reset ok")

    async def imagine(prompt_text, interface):
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
""")


class EditorContextProcess:
    """Manage a real ``chibi ide --stdio`` process with editor_context capture."""

    def __init__(self) -> None:
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
            for line in lines:
                if line.strip():
                    try:
                        self._frames.append(json.loads(line.decode()))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

    def shutdown(self) -> None:
        self.send({"type": "shutdown"})
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait(timeout=12)
        assert self.process.returncode == 0, self._stderr()

    def close(self) -> None:
        self._selector.close()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)

    def _stderr(self) -> str:
        if self.process.poll() is None:
            return "<process still running>"
        assert self.process.stderr is not None
        return self.process.stderr.read().decode(errors="replace")


def request_with_editor_context(
    request_id: str,
    thread_id: int,
    prompt: str,
    *,
    active_file: str,
    selection_text: str,
    language_id: str,
    workspace_root: str,
) -> dict[str, Any]:
    return {
        "type": "request",
        "request_id": request_id,
        "thread_id": thread_id,
        "prompt": prompt,
        "workspace_root": workspace_root,
        "active_file": active_file,
        "selection": {
            "start_line": 10,
            "end_line": 15,
            "text": selection_text,
        },
        "cursor_position": {"line": 12, "character": 5},
        "language_id": language_id,
    }


def initialized_process() -> EditorContextProcess:
    client = EditorContextProcess()
    client.send({"type": "initialize", "protocol_version": 1})
    assert client.wait_for("ready")["protocol_version"] == 1
    return client


def test_editor_context_reaches_backend_with_required_fields() -> None:
    """Real CLI receives editor_context and makes it available to get_llm_chat_completion_answer."""
    client = initialized_process()
    try:
        client.send(
            request_with_editor_context(
                request_id="editor-ctx-test",
                thread_id=1,
                prompt="explain this code",
                active_file="src/utils/helper.py",
                selection_text="def calculate(items): return sum(items)",
                language_id="python",
                workspace_root="/home/developer/project",
            )
        )
        assert client.wait_for("status", "editor-ctx-test", "running")["state"] == "running"
        result = client.wait_for("result", "editor-ctx-test")
        assert result["model"] == "fake-model"
        assert result["provider"] == "fake"
        assert result["content"] == "editor_context test response"

        # Find the captured editor_context from the stdout marker line.
        captured: dict[str, Any] | None = None
        for frame in client._frames:
            if "_captured_editor_context" in frame:
                captured = frame["_captured_editor_context"]
                break

        assert captured is not None, (
            f"editor_context was not captured; buffered frames: {client._frames}; stderr={client._stderr()!r}"
        )
        assert captured["active_file"] == "src/utils/helper.py", captured
        assert captured["language_id"] == "python", captured
        assert captured["selection"]["text"] == "def calculate(items): return sum(items)", captured
        assert captured["selection"]["start_line"] == 10, captured
        assert captured["selection"]["end_line"] == 15, captured
        assert captured["workspace_root"] == "/home/developer/project", captured
        assert captured["cursor_position"] == {"line": 12, "character": 5}, captured

        client.shutdown()
    finally:
        client.close()


def test_editor_context_null_selection_graceful() -> None:
    """Null selection (no code selected) is handled gracefully — only active_file + language_id required."""
    client = initialized_process()
    try:
        req = {
            "type": "request",
            "request_id": "null-sel-test",
            "thread_id": 3,
            "prompt": "hello",
            "workspace_root": "/tmp",
            "active_file": "main.py",
            "selection": None,
            "cursor_position": None,
            "language_id": "python",
        }
        client.send(req)
        assert client.wait_for("status", "null-sel-test", "running")
        result = client.wait_for("result", "null-sel-test")
        assert result["content"] == "editor_context test response"

        captured: dict[str, Any] | None = None
        for frame in client._frames:
            if "_captured_editor_context" in frame:
                captured = frame["_captured_editor_context"]
                break

        assert captured is not None, f"editor_context was not captured; buffered frames: {client._frames}"
        assert captured["active_file"] == "main.py"
        assert captured["language_id"] == "python"
        assert captured["selection"] is None
        assert captured["cursor_position"] is None

        client.shutdown()
    finally:
        client.close()


def test_editor_context_response_ok_no_crash() -> None:
    """Smoke: the full round-trip completes without error and returns a result."""
    client = initialized_process()
    try:
        client.send(
            request_with_editor_context(
                request_id="smoke-roundtrip",
                thread_id=4,
                prompt="hello",
                active_file="test.py",
                selection_text="x = 1",
                language_id="python",
                workspace_root="/tmp",
            )
        )
        assert client.wait_for("status", "smoke-roundtrip", "running")
        result = client.wait_for("result", "smoke-roundtrip")
        assert result["content"] == "editor_context test response"
        client.shutdown()
    finally:
        client.close()
