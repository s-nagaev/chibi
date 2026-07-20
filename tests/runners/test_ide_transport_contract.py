"""Deterministic IDE JSONL contract coverage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import PROTOCOL_VERSION, IDEStdioRunner
from chibi.schemas.app import ModelChangeSchema

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ide_protocol"
MODELS = [
    ModelChangeSchema(provider="openai", name="gpt-example", display_name="🟢 GPT-Example", image_generation=False),
    ModelChangeSchema(provider="alibaba", name="qwen-fake", display_name="Qwen-Fake", image_generation=False),
]
_gate: asyncio.Event | None = None


async def fake_prompt(interface: Any) -> None:
    """Return the canonical fake-provider answer."""
    interface.response_model = "gpt-example"
    interface.response_provider = "openai"
    if _gate is not None:
        await _gate.wait()
    await interface.send_message("This function `foo` returns the integer `42`.")


async def fake_reset(interface: Any) -> None:
    """Return a deterministic reset answer."""
    await interface.send_message("Done!")


async def fake_imagine(prompt: str, interface: Any) -> None:
    """Return a deterministic image reference."""
    await interface.send_images(["fake_image_url.png"])


async def fake_models(user_id: int, thread_id: int = 0, **kwargs: Any) -> list[ModelChangeSchema]:
    """Return fake provider models."""
    return MODELS


async def fake_select(interface: Any, model: ModelChangeSchema) -> None:
    """Accept a fake model selection."""


async def fake_info(user_id: int) -> str:
    """Return deterministic user information."""
    return "Fake user info for testing"


def patches(gate: asyncio.Event | None = None) -> list[Any]:
    """Patch orchestration boundaries with deterministic fakes."""
    global _gate
    _gate = gate
    return [
        patch("chibi.runners.ide_transport.handle_user_prompt", fake_prompt),
        patch("chibi.runners.ide_transport.handle_reset", fake_reset),
        patch("chibi.runners.ide_transport.handle_image_generation", fake_imagine),
        patch("chibi.runners.ide_transport.get_models_available", fake_models),
        patch("chibi.runners.ide_transport.set_active_model", fake_select),
        patch("chibi.runners.ide_transport.get_info", fake_info),
    ]


def fixture(name: str) -> list[dict[str, Any]]:
    """Load JSONL fixture records."""
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines() if line.strip()]


class OutputRecorder:
    """Capture protocol frames through an async callable."""

    def __init__(self) -> None:
        """Initialize empty captured output."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        """Capture one protocol frame.

        Args:
            message: Emitted JSON-compatible protocol frame.
        """
        self.frames.append(message)


def runner() -> tuple[IDEStdioRunner, list[dict[str, Any]]]:
    """Create a runner with captured protocol output.

    Returns:
        The runner and its mutable captured-frame list.
    """
    instance = IDEStdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write"] = recorder
    return instance, recorder.frames


async def wait_for(output: list[dict[str, Any]], request_id: str, frame_type: str) -> None:
    """Wait until a correlated frame is emitted."""
    for _ in range(200):
        if any(
            frame.get("request_id") == request_id and (frame["type"] == frame_type or frame.get("state") == frame_type)
            for frame in output
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Missing {frame_type} for {request_id}: {output}")


def request(request_id: str, thread_id: int, prompt: str = "hello") -> dict[str, Any]:
    """Build a valid request."""
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


@pytest.mark.asyncio
async def test_canonical_session_output() -> None:
    """Replay the canonical session and compare every frame."""
    gate = asyncio.Event()
    gate.set()
    active = patches(gate)
    for item in active:
        item.start()
    try:
        instance, output = runner()
        for message in fixture("valid_session_input.jsonl"):
            await instance._handle_message(message)
        await wait_for(output, "01HXY9K1ABCDEFGH", "result")
        assert output == fixture("valid_session_output.jsonl")
    finally:
        for item in active:
            item.stop()


@pytest.mark.asyncio
async def test_canonical_cancel_and_shutdown_outputs() -> None:
    """Compare canonical cancellation and shutdown frames."""
    gate = asyncio.Event()
    active = patches(gate)
    for item in active:
        item.start()
    try:
        instance, output = runner()
        cancel_input = fixture("valid_cancel_input.jsonl")
        await instance._handle_message(cancel_input[0])
        await instance._handle_message(cancel_input[1])
        await wait_for(output, "01HXY9K2CANCELL0", "status")
        await instance._handle_message(cancel_input[2])
        await wait_for(output, "01HXY9K2CANCELL0", "error")
        assert output == fixture("valid_cancel_output.jsonl")
        second, second_output = runner()
        for message in fixture("valid_shutdown_input.jsonl"):
            await second._handle_message(message)
        assert second_output == fixture("valid_shutdown_output.jsonl")
    finally:
        gate.set()
        for item in active:
            item.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", fixture("invalid_cases.jsonl"), ids=lambda value: value["name"])
async def test_invalid_cases_match_canonical(case: dict[str, Any]) -> None:
    """Compare each invalid-case output fixture exactly."""
    active = patches()
    for item in active:
        item.start()
    try:
        instance, output = runner()
        for message in case["setup"]:
            await instance._handle_message(message)
        raw = case["input"]
        if isinstance(raw, str):
            await instance._error("malformed_request", "Malformed request.")
        else:
            await instance._handle_message(raw)
        assert output == case["expected_output"]
    finally:
        for item in active:
            item.stop()


@pytest.mark.asyncio
async def test_selection_schema_rejects_nested_shape() -> None:
    """Nested {start:{line,character}, end:{line,character}, text} is rejected."""
    instance, output = runner()
    await instance._handle_message({"type": "initialize", "protocol_version": PROTOCOL_VERSION})
    nested_selection = {
        "type": "request",
        "request_id": "nested-sel-test",
        "thread_id": 1,
        "prompt": "hello",
        "workspace_root": "/tmp",
        "active_file": None,
        "selection": {"start": {"line": 10, "character": 0}, "end": {"line": 15, "character": 12}, "text": "foo"},
        "cursor_position": None,
        "language_id": None,
    }
    await instance._handle_message(nested_selection)
    assert any(
        frame.get("code") == "malformed_request" and frame.get("request_id") == "nested-sel-test"
        for frame in output
    ), f"Expected malformed_request for nested selection, got: {output}"


@pytest.mark.asyncio
async def test_selection_schema_accepts_flat_shape() -> None:
    """Flat {start_line, end_line, text} is accepted; optional start_character/end_character tolerated."""
    gate = asyncio.Event()
    gate.set()
    active = patches(gate)
    for item in active:
        item.start()
    try:
        instance, output = runner()
        await instance._handle_message({"type": "initialize", "protocol_version": PROTOCOL_VERSION})
        flat_selection = {
            "type": "request",
            "request_id": "flat-sel-test",
            "thread_id": 1,
            "prompt": "hello",
            "workspace_root": "/tmp",
            "active_file": None,
            "selection": {"start_line": 10, "end_line": 15, "text": "foo"},
            "cursor_position": None,
            "language_id": None,
        }
        await instance._handle_message(flat_selection)
        await wait_for(output, "flat-sel-test", "result")
        assert any(
            frame.get("request_id") == "flat-sel-test" and frame["type"] == "result"
            for frame in output
        ), f"Expected result for flat selection, got: {output}"
    finally:
        for item in active:
            item.stop()


@pytest.mark.asyncio
async def test_selection_schema_accepts_flat_shape_with_optional_chars() -> None:
    """Flat shape with optional start_character/end_character is accepted."""
    gate = asyncio.Event()
    gate.set()
    active = patches(gate)
    for item in active:
        item.start()
    try:
        instance, output = runner()
        await instance._handle_message({"type": "initialize", "protocol_version": PROTOCOL_VERSION})
        flat_with_chars = {
            "type": "request",
            "request_id": "flat-sel-chars-test",
            "thread_id": 1,
            "prompt": "hello",
            "workspace_root": "/tmp",
            "active_file": None,
            "selection": {"start_line": 10, "end_line": 15, "start_character": 4, "end_character": 12, "text": "foo"},
            "cursor_position": None,
            "language_id": None,
        }
        await instance._handle_message(flat_with_chars)
        await wait_for(output, "flat-sel-chars-test", "result")
        assert any(
            frame.get("request_id") == "flat-sel-chars-test" and frame["type"] == "result"
            for frame in output
        ), f"Expected result for flat selection with optional chars, got: {output}"
    finally:
        for item in active:
            item.stop()


@pytest.mark.asyncio
async def test_commands_and_thread_scheduling() -> None:
    """Exercise commands, same-thread locking, different-thread overlap, and cancellation."""
    gate = asyncio.Event()
    active = patches(gate)
    for item in active:
        item.start()
    try:
        instance, output = runner()
        await instance._handle_message({"type": "initialize", "protocol_version": PROTOCOL_VERSION})
        await instance._handle_message(request("model", 1, "/model"))
        await wait_for(output, "model", "result")
        assert (
            "GPT-Example"
            in next(frame for frame in output if frame.get("request_id") == "model" and frame["type"] == "result")[
                "content"
            ]
        )
        await instance._handle_message(request("first", 2, "one"))
        await instance._handle_message(request("second", 2, "two"))
        await instance._handle_message(request("parallel", 3, "three"))
        assert any(frame.get("request_id") == "second" and frame.get("state") == "queued" for frame in output)
        await wait_for(output, "parallel", "status")
        assert any(frame.get("request_id") == "parallel" and frame.get("state") == "running" for frame in output)
        await instance._handle_message({"type": "cancel", "request_id": "first"})
        await wait_for(output, "first", "error")
        gate.set()
        await wait_for(output, "second", "running")
    finally:
        gate.set()
        for item in active:
            item.stop()
