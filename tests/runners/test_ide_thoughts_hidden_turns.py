"""Hidden-turn thought suppression in the IDE stdio transport (task 5).

Covers:
- the result frame omitting ``thoughts`` when the turn ended hidden;
- the continuation-snapshot re-baseline: a hidden background continuation
  turn's private reasoning must never ride a later background frame;
- the bot-level suppression path calling the absorb hook.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.config.gpt import gpt_settings
from chibi.interfaces.stdio_interface import StdioInterface
from chibi.interfaces.stdio_runner import StdioRunner
from chibi.schemas.app import ChatResponseSchema
from chibi.services.bot import handle_tool_response
from chibi.services.providers.tools.schemas import ToolResponseSchema


async def wait_for_result(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Wait until the runner emits the result frame for the request."""
    for _ in range(300):
        results = [frame for frame in frames if frame["type"] == "result"]
        if results:
            return results[0]
        await asyncio.sleep(0.01)
    raise AssertionError(f"No result frame emitted: {frames}")


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


class OutputRecorder:
    """Capture protocol frames through an async callable."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.frames.append(message)


def runner() -> tuple[StdioRunner, list[dict[str, Any]]]:
    instance = StdioRunner()
    recorder = OutputRecorder()
    instance.__dict__["_write"] = recorder
    return instance, recorder.frames


def closed_interface_with_parent_thoughts(
    instance: StdioRunner,
    parent_thoughts: str | None,
) -> StdioInterface:
    """Build a closed interface whose result frame already carried reasoning."""

    async def emit_background(
        thread_id: int, content: str, model: str | None, provider: str | None, thoughts: str | None = None
    ) -> None:
        await instance._emit_background_message(thread_id, content, model, provider, thoughts)

    interface = StdioInterface(11, "p", {}, lambda _: None, background_emit=emit_background)
    interface.response_thoughts = parent_thoughts
    interface.mark_closed()
    return interface


@pytest.mark.asyncio
async def test_result_frame_omits_thoughts_on_a_hidden_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn ended by ack_silently() must not ship its private CoT on the wire."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    instance, output = runner()
    instance._thoughts_enabled = True

    async def prompt(interface: Any) -> None:
        await interface.send_llm_thoughts("private reasoning the user must not see")
        interface.response_visibility = "hidden"

    with patch("chibi.interfaces.stdio_runner.handle_user_prompt", prompt):
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(output)

    assert "thoughts" not in result


@pytest.mark.asyncio
async def test_result_frame_keeps_thoughts_on_a_visible_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a visible turn still carries its reasoning on the result frame."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    instance, output = runner()
    instance._thoughts_enabled = True

    async def prompt(interface: Any) -> None:
        await interface.send_llm_thoughts("visible turn reasoning")
        interface.response_visibility = "visible"

    with patch("chibi.interfaces.stdio_runner.handle_user_prompt", prompt):
        await instance._handle_message(
            {"type": "initialize", "protocol_version": 1, "capabilities": {"thoughts": True}}
        )
        await instance._handle_message(request("r1"))
        result = await wait_for_result(output)

    assert result["thoughts"] == "visible turn reasoning"


@pytest.mark.asyncio
async def test_absorb_rebaselines_so_hidden_continuation_cot_stays_off_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning captured by a hidden background turn must not ride the next frame's delta."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    instance, output = runner()
    instance._thoughts_enabled = True
    interface = closed_interface_with_parent_thoughts(instance, "parent step one")

    # A background continuation turn ends hidden: its reasoning is captured,
    # then the bot-level gate suppresses delivery and absorbs the snapshot.
    await interface.send_llm_thoughts("hidden continuation CoT")
    await interface.absorb_hidden_turn_reasoning()

    # The next background continuation is visible and delivered.
    await interface.send_llm_thoughts("fresh continuation reasoning")
    await interface.send_tool_answer("Continuation answer", model="m", provider="prov")

    messages = [frame for frame in output if frame["type"] == "message"]
    assert len(messages) == 1
    assert messages[0]["thoughts"] == "fresh continuation reasoning"


@pytest.mark.asyncio
async def test_bot_hidden_tool_response_absorbs_reasoning_and_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bot-level hidden gate routes through the IDE absorb hook end to end."""
    monkeypatch.setattr(gpt_settings, "show_llm_thoughts", False)
    instance, output = runner()
    instance._thoughts_enabled = True
    interface = closed_interface_with_parent_thoughts(instance, "parent step one")

    hidden_response = ChatResponseSchema(answer="", provider="prov", model="m", usage=None, visibility="hidden")
    with patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=hidden_response)):
        await handle_tool_response(
            tool_response=ToolResponseSchema(tool_name="t", status="ok", result="ok"), interface=interface
        )

    assert not any(frame["type"] == "message" for frame in output)

    # The hidden turn's reasoning is absorbed: a later delivered continuation
    # carries only its own fresh reasoning, never the hidden turn's CoT.
    await interface.send_llm_thoughts("later visible continuation reasoning")
    await interface.send_tool_answer("Later answer", model="m", provider="prov")
    messages = [frame for frame in output if frame["type"] == "message"]
    assert len(messages) == 1
    assert messages[0]["thoughts"] == "later visible continuation reasoning"
    assert "parent step one" not in (messages[0].get("thoughts") or "")
