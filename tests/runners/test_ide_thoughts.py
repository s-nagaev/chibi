"""Optional LLM thoughts in IDE result frames (v11_05).

Covers:
  provider reasoning → send_llm_thoughts capture seam → IDEInterface
    → result frame "thoughts" field (capability-gated, 256 KB capped)
plus the feature-quiet no-reasoning path and result-frame-only delivery.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.config.gpt import gpt_settings
from chibi.runners.ide_transport import MAX_THOUGHTS_BYTES, THOUGHTS_TRUNCATION_MARKER, IDEStdioRunner, _cap_thoughts
from chibi.services.providers.utils import send_llm_thoughts as deliver_thoughts


class OutputRecorder:
    """Capture protocol frames through an async callable."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.frames.append(message)


def runner_with(prompt_handler: Any) -> tuple[IDEStdioRunner, list[dict[str, Any]], list[Any]]:
    """Create a runner whose prompt handling is replaced by prompt_handler."""
    instance = IDEStdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write"] = recorder

    async def fake_reset(interface: Any) -> None:
        await interface.send_message("Done!")

    patches = [
        patch("chibi.runners.ide_transport.handle_user_prompt", prompt_handler),
        patch("chibi.runners.ide_transport.handle_reset", fake_reset),
        patch("chibi.runners.ide_transport.handle_image_generation", AsyncMock()),
        patch("chibi.runners.ide_transport.get_models_available", AsyncMock(return_value=[])),
        patch("chibi.runners.ide_transport.set_active_model", AsyncMock()),
        patch("chibi.runners.ide_transport.get_info", AsyncMock(return_value="info")),
    ]
    for item in patches:
        item.start()
    return instance, recorder.frames, patches


def stop_patches(patches: list[Any]) -> None:
    for item in patches:
        item.stop()


def request(request_id: str, prompt: str = "hello") -> dict[str, Any]:
    return {
        "type": "request",
        "request_id": request_id,
        "thread_id": 1,
        "prompt": prompt,
        "workspace_root": "/tmp",
        "active_file": None,
        "selection": None,
        "cursor_position": None,
        "language_id": None,
    }


def result_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "result"]


async def wait_for_result(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Wait until the runner emits the result frame for the request."""
    for _ in range(300):
        results = result_frames(frames)
        if results:
            return results[0]
        await asyncio.sleep(0.01)
    raise AssertionError(f"No result frame emitted: {frames}")


@pytest.mark.asyncio
async def test_result_frame_carries_thoughts_when_capability_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning captured through the shared seam lands on the result frame.

    The global show_llm_thoughts toggle stays off (its default): the IDE
    interface captures thoughts regardless, and the handshake capability
    alone gates emission.
    """
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts="chain of thought", interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert result["thoughts"] == "chain of thought"
        assert result["content"] == "Answer"
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_result_frame_omits_thoughts_without_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clients that never declare the thoughts capability never see the field."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts="chain of thought", interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message({"type": "initialize", "protocol_version": 1})
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert "thoughts" not in result
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_no_reasoning_path_keeps_v1_result_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quiet provider with the capability on emits a byte-identical v1 frame."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", True)

    async def prompt(interface: Any) -> None:
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert list(result.keys()) == ["type", "request_id", "content"]
        assert result == {"type": "result", "request_id": "r1", "content": "Answer"}
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_thoughts_accumulate_across_multiple_captures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoning reported in several pieces is joined in arrival order."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts="first piece", interface=interface)
        await deliver_thoughts(thoughts="second piece", interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert result["thoughts"] == "first piece\nsecond piece"
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_thoughts_capped_at_256kb_marker_at_head_tail_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized reasoning keeps the tail, drops the head, marker lands at the head."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    head_token = "HEADTOKEN"
    tail = "T" * 8192
    oversized = head_token + "A" * (MAX_THOUGHTS_BYTES + 4096 - len(head_token) - len(tail)) + tail

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts=oversized, interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        emitted = result["thoughts"]
        assert emitted.startswith(THOUGHTS_TRUNCATION_MARKER)
        assert emitted.endswith(tail)
        assert head_token not in emitted
        assert len(emitted.encode("utf-8")) == MAX_THOUGHTS_BYTES
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_thoughts_at_exact_cap_pass_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoning exactly at the cap is emitted verbatim with no marker."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    exact = "x" * MAX_THOUGHTS_BYTES

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts=exact, interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert result["thoughts"] == exact
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_multibyte_thoughts_truncation_stays_utf8_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A byte cut landing inside a multibyte character never emits invalid UTF-8."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    oversized = "я" * 140000

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts=oversized, interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        encoded = result["thoughts"].encode("utf-8")
        assert MAX_THOUGHTS_BYTES - 4 <= len(encoded) <= MAX_THOUGHTS_BYTES
        assert result["thoughts"].startswith(THOUGHTS_TRUNCATION_MARKER)
    finally:
        stop_patches(patches)


def test_cap_thoughts_keeps_tail_drops_head_and_prepends_marker() -> None:
    """Oversized text keeps only the tail bytes; the marker is prepended at the head."""
    head_token = "HEADTOKEN"
    tail = "T" * 8192
    oversized = head_token + "A" * (MAX_THOUGHTS_BYTES + 4096 - len(head_token) - len(tail)) + tail
    capped = _cap_thoughts(oversized)
    assert capped.startswith(THOUGHTS_TRUNCATION_MARKER)
    assert capped.endswith(tail)
    assert head_token not in capped
    assert len(capped.encode("utf-8")) == MAX_THOUGHTS_BYTES


def test_cap_thoughts_multibyte_straddle_stays_utf8_safe() -> None:
    """A cut landing inside a multibyte character drops the partial bytes and stays valid UTF-8."""
    marker_len = len(THOUGHTS_TRUNCATION_MARKER.encode("utf-8"))
    budget = MAX_THOUGHTS_BYTES - marker_len
    filler = (budget + 2) % 4
    oversized = "🦊" * (MAX_THOUGHTS_BYTES // 4 + 1) + "x" * filler
    capped = _cap_thoughts(oversized)
    assert capped.startswith(THOUGHTS_TRUNCATION_MARKER)
    assert len(capped.encode("utf-8")) == MAX_THOUGHTS_BYTES - 2
    assert capped.endswith("🦊🦊" + "x" * filler)
    assert "\ufffd" not in capped


def test_cap_thoughts_under_cap_passthrough() -> None:
    """Text at or under the cap is returned verbatim with no marker."""
    assert _cap_thoughts("chain of thought") == "chain of thought"
    exact = "x" * MAX_THOUGHTS_BYTES
    assert _cap_thoughts(exact) == exact


@pytest.mark.asyncio
async def test_thoughts_are_result_frame_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No frame other than the result carries a thoughts field."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)

    async def prompt(interface: Any) -> None:
        await deliver_thoughts(thoughts="chain of thought", interface=interface)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        await wait_for_result(frames)
        carriers = [frame["type"] for frame in frames if "thoughts" in frame]
        assert carriers == ["result"]
    finally:
        stop_patches(patches)
