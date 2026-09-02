"""Usage data in IDE result frames (v11_02).

Covers:
  provider-reported usage → ChatResponseSchema → handle_user_prompt
    → interface.response_usage → result frame "usage" object
plus the curated model context-window lookup.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai.types import CompletionUsage

import chibi.config  # noqa: F401
from chibi.config.gpt import gpt_settings
from chibi.models import get_model_context_window
from chibi.runners.ide_transport import IDEStdioRunner, build_usage_payload
from chibi.schemas.app import ChatResponseSchema, UsageSchema

# In-process harness reused from the contract tests' style: patch the
# orchestration boundary and replay protocol frames through the runner.


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
async def test_result_frame_carries_usage_when_provider_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider-reported usage lands on the result frame with a known window.

    The model window (128000 for gpt-4o) is below MAX_HISTORY_TOKENS here, so
    no clamp applies: this is the no-clamp ordering of the effective-ceiling rule.
    """
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)

    async def prompt(interface: Any) -> None:
        await interface.send_message("Answer")

    def respond(interface: Any) -> ChatResponseSchema:
        return ChatResponseSchema(
            answer="Answer",
            provider="openai",
            model="gpt-4o",
            usage=UsageSchema(prompt_tokens=1200, completion_tokens=340, total_tokens=1540),
        )

    def wrapped(interface: Any) -> Any:
        chat_response = respond(interface)
        interface.response_model = chat_response.model
        interface.response_provider = chat_response.provider
        interface.response_usage = chat_response.usage
        return prompt(interface)

    instance, frames, patches = runner_with(wrapped)
    try:
        await instance._handle_message({"type": "initialize", "protocol_version": 1})
        await instance._handle_message(request("r1"))
        result = await wait_for_result(frames)
        assert result["usage"] == {"input_tokens": 1200, "output_tokens": 340, "context_window": 128000}
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_result_frame_omits_usage_when_provider_reports_none() -> None:
    """No usage from the provider means the field is absent, not null."""

    async def prompt(interface: Any) -> None:
        interface.response_model = "some-model"
        interface.response_provider = "openai"
        interface.response_usage = None
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message({"type": "initialize", "protocol_version": 1})
        await instance._handle_message(request("r2"))
        result = await wait_for_result(frames)
        assert "usage" not in result
        assert result["model"] == "some-model"
    finally:
        stop_patches(patches)


@pytest.mark.asyncio
async def test_result_frame_usage_context_window_is_null_for_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown models still report real token counts with a null window.

    Null stays null even when the clamp ceiling is tiny: an unknown model is
    never replaced by the MAX_HISTORY_TOKENS value.
    """
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 50)

    async def prompt(interface: Any) -> None:
        interface.response_model = "totally-unknown-model"
        interface.response_provider = "openai"
        interface.response_usage = UsageSchema(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        await interface.send_message("Answer")

    instance, frames, patches = runner_with(prompt)
    try:
        await instance._handle_message({"type": "initialize", "protocol_version": 1})
        await instance._handle_message(request("r3"))
        result = await wait_for_result(frames)
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5, "context_window": None}
    finally:
        stop_patches(patches)


def test_build_usage_payload_adds_anthropic_cache_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic-style usage reports cached input outside input_tokens.

    The model window equals MAX_HISTORY_TOKENS here (boundary case of the
    clamp: min is a no-op when both sides are equal).
    """
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)
    usage = UsageSchema(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cache_creation_input_tokens=30,
        cache_read_input_tokens=400,
    )
    payload = build_usage_payload(usage=usage, provider="Anthropic", model="claude-sonnet-4-5-20250929")
    assert payload == {"input_tokens": 530, "output_tokens": 20, "context_window": 200000}


def test_build_usage_payload_does_not_double_count_openai_cached_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI already includes cached tokens inside prompt_tokens."""
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)
    usage = UsageSchema(prompt_tokens=100, completion_tokens=20, total_tokens=120, cache_read_input_tokens=40)
    payload = build_usage_payload(usage=usage, provider="openai", model="gpt-4o")
    assert payload == {"input_tokens": 100, "output_tokens": 20, "context_window": 128000}


def test_build_usage_payload_accepts_raw_completion_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw openai CompletionUsage (the other half of the schema union) works too."""
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 1000000)
    usage = CompletionUsage(prompt_tokens=77, completion_tokens=13, total_tokens=90)
    payload = build_usage_payload(usage=usage, provider="openai", model="gpt-5.2-2026-01-15")
    assert payload == {"input_tokens": 77, "output_tokens": 13, "context_window": 400000}


def test_build_usage_payload_clamps_to_max_history_tokens() -> None:
    """Clamp active: a model window above MAX_HISTORY_TOKENS is capped at it.

    Default settings (MAX_HISTORY_TOKENS=100000): proactive summarization
    fires at that threshold, so 100000 is the effective ceiling, not 400000.
    """
    usage = CompletionUsage(prompt_tokens=77, completion_tokens=13, total_tokens=90)
    payload = build_usage_payload(usage=usage, provider="openai", model="gpt-5.2-2026-01-15")
    assert payload == {"input_tokens": 77, "output_tokens": 13, "context_window": gpt_settings.max_history_tokens}


def test_build_usage_payload_null_window_survives_clamp() -> None:
    """Unknown model: null context_window is emitted as-is, never clamped to a number."""
    usage = CompletionUsage(prompt_tokens=77, completion_tokens=13, total_tokens=90)
    payload = build_usage_payload(usage=usage, provider="openai", model="definitely-not-a-model")
    assert payload == {"input_tokens": 77, "output_tokens": 13, "context_window": None}


def test_build_usage_payload_returns_none_without_usage() -> None:
    assert build_usage_payload(usage=None, provider="openai", model="gpt-4o") is None


class TestModelContextWindowLookup:
    """Curated context-window map behavior."""

    def test_exact_match(self) -> None:
        assert get_model_context_window("deepseek-chat") == 128000

    def test_case_insensitive_exact_match(self) -> None:
        assert get_model_context_window("MiniMax-M2.7") == 204800

    def test_dated_variant_resolves_through_prefix(self) -> None:
        assert get_model_context_window("gpt-4o-2024-11-20") == 128000

    def test_longest_prefix_wins(self) -> None:
        assert get_model_context_window("o3-mini") == 200000

    def test_unknown_model_returns_none(self) -> None:
        assert get_model_context_window("definitely-not-a-model") is None

    def test_none_and_empty_return_none(self) -> None:
        assert get_model_context_window(None) is None
        assert get_model_context_window("") is None


class _FakeInterfaceForBot:
    """Minimal interface surface handle_user_prompt touches."""

    def __init__(self) -> None:
        self.response_model: str | None = None
        self.response_provider: str | None = None
        self.response_usage: UsageSchema | CompletionUsage | None = None

    user_data = "user-1"
    chat_data = "chat-1"
    storage_id = 1
    thread_id = 0

    async def get_text_prompt(self) -> str | None:
        return "hello"

    async def get_voice_prompt(self) -> str | None:
        return None

    async def get_caption(self) -> str | None:
        return None

    async def send_action_typing(self) -> None:
        return None

    async def send_message(self, message: str, **kwargs: Any) -> None:
        return None


class TestHandleUserPromptUsagePlumbing:
    """handle_user_prompt copies chat_response.usage onto the interface."""

    @pytest.mark.asyncio
    async def test_response_usage_set_from_chat_response(self) -> None:
        from chibi.services.bot import handle_user_prompt

        interface = _FakeInterfaceForBot()
        chat_response = ChatResponseSchema(
            answer="Answer",
            provider="anthropic",
            model="claude-sonnet-4-5",
            usage=UsageSchema(prompt_tokens=500, completion_tokens=60, total_tokens=560),
        )

        with (
            patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=interface)

        assert interface.response_usage is chat_response.usage
        assert interface.response_model == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_response_usage_stays_none_without_provider_usage(self) -> None:
        from chibi.services.bot import handle_user_prompt

        interface = _FakeInterfaceForBot()
        chat_response = ChatResponseSchema(answer="Answer", provider="openai", model="gpt-4o", usage=None)

        with (
            patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(return_value=chat_response)),
            patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
        ):
            await handle_user_prompt(interface=interface)

        assert interface.response_usage is None
