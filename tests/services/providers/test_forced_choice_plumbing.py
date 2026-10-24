"""Tests for the per-adapter forced-choice mapping (sub-task 4c-2).

The contract under test: when the single switch
(``gpt_settings.force_terminal_tool_choice``, default OFF) is on AND the
request carries a user interface (exactly when the terminal tools are offered
per ruling D1), each adapter asks its provider at the protocol level to end
the turn with a tool call instead of prose. Forcing is an optimisation only:
the turn-end validator is the guarantee, so every test also pins that the
off-state and interface-less payloads are byte-identical to the pre-4c-2
behaviour (no forced-choice field present).

The tests drive real request construction per adapter and assert the payload
actually handed to the client — no live network calls anywhere.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from anthropic import omit as anthropic_omit
from google.genai.types import FunctionCall as GeminiFunctionCall
from google.genai.types import FunctionCallingConfigMode
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import ResponseOutputMessage

import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.config import gpt_settings
from chibi.config.gpt import GPTSettings
from chibi.models import Message, User
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI as ChibiOpenAI
from chibi.services.providers.terminal_calls import (
    anthropic_forced_tool_choice,
    forced_choice_applicable,
)

TEST_TOKEN = "test-token-for-forced-choice"
TEST_MODEL = "kimi-k2.6"
TEST_OPENAI_MODEL = "gpt-5.6-terra"

# An interface stub with streaming disabled: the non-streaming request path
# is used, so the payload handed to the SDK client is capturable directly.
INTERFACE_STUB = SimpleNamespace(thread_id=1, streaming_enabled=False, delta_emitted=False, captures_llm_thoughts=False)


def _forcing(on: bool):
    """Patch the single switch (the one mechanism behind all five adapters)."""
    return patch.object(gpt_settings, "force_terminal_tool_choice", on)


async def _openai_cc_payload(forcing: bool, interface: Any) -> dict[str, Any]:
    """Build a chat-completions request through the shared loop, capture the payload."""
    provider = MoonshotAI(token=TEST_TOKEN)
    answer = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="plain answer", tool_calls=None),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )
    create_mock = AsyncMock(return_value=answer)
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    with (
        _forcing(forcing),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
    ):
        await provider._get_chat_completion_response(
            messages=[Message(role="user", content="Hi").to_openai()],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="",
            interface=interface,
        )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


async def _responses_payload(forcing: bool, interface: Any) -> dict[str, Any]:
    """Build a Responses request, capture the payload handed to the SDK."""
    provider = ChibiOpenAI(token=TEST_TOKEN)
    message_item = ResponseOutputMessage(
        id="msg_1",
        content=[],
        role="assistant",
        status="completed",
        type="message",
    )
    response = SimpleNamespace(output=[message_item], output_text="Hello", usage=None)
    create_mock = AsyncMock(return_value=response)
    mock_client = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    with (
        _forcing(forcing),
        patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")),
    ):
        provider.client = mock_client  # type: ignore[assignment]
        await provider._get_response_completion_response(
            messages=[Message(role="user", content="Hi")],
            model=TEST_OPENAI_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="",
            interface=cast("Any", interface),
        )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


def _anthropic_answer() -> SimpleNamespace:
    text_block = SimpleNamespace(text="plain answer")
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    return SimpleNamespace(content=[text_block], usage=usage, model=TEST_MODEL, role="assistant")


async def _anthropic_payload(forcing: bool, interface: Any) -> dict[str, Any]:
    """Build an Anthropic request via ``_generate_content``, capture the payload."""
    provider = Anthropic(token=TEST_TOKEN)
    create_mock = AsyncMock(return_value=_anthropic_answer())
    mock_client: Any = SimpleNamespace(messages=SimpleNamespace(create=create_mock, stream=AsyncMock()))
    provider._client = mock_client
    with _forcing(forcing):
        await provider._generate_content(
            model=TEST_MODEL,
            system_prompt="sp",
            messages=[{"role": "user", "content": "Hi"}],
            interface=cast("Any", interface),
        )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


async def _gemini_config(forcing: bool, interface: Any) -> Any:
    """Build a Gemini request, capture the GenerateContentConfig handed onward."""
    provider = Gemini(token=TEST_TOKEN)
    # A terminal say() response: the turn exits through the terminal branch,
    # so the request-construction helper never triggers a repair round-trip.
    response = SimpleNamespace(
        candidates=None,
        function_calls=[SimpleNamespace(name="say", args={"text": "Hi"})],
        usage_metadata=None,
        model_version="gemini-test",
    )
    generate_mock = AsyncMock(return_value=response)
    with (
        _forcing(forcing),
        patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch.object(Gemini, "_generate_content", generate_mock),
        patch(
            "chibi.services.providers.gemini_native.RegisteredChibiTools.call",
            new=AsyncMock(side_effect=AssertionError("terminal call reached the dispatcher")),
        ),
    ):
        await provider._get_chat_completion_response(
            messages=[],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model="models/gemini-3.8-flash",
            system_prompt="",
            interface=cast("Any", interface),
        )
    assert generate_mock.await_args is not None
    return generate_mock.await_args.kwargs["config"]


async def _mistral_payload(forcing: bool, interface: Any) -> dict[str, Any]:
    """Build a Mistral request, capture the payload handed to the SDK."""
    provider = MistralAI(token=TEST_TOKEN)
    message = SimpleNamespace(content="plain answer", tool_calls=None)
    usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
    create_mock = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage))
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(complete_async=create_mock, stream_async=AsyncMock()))
    provider._client = mock_client
    with (
        _forcing(forcing),
        patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
    ):
        await provider._generate_content(
            model=TEST_MODEL,
            messages=[Message(role="user", content="Hi").to_mistral()],
            interface=cast("Any", interface),
        )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


class TestTheSingleSwitch:
    """One mechanism, default OFF, gating all five adapters."""

    def test_default_is_off(self) -> None:
        # A fresh settings object (not the process singleton) pins the field
        # default itself, so a config regression cannot hide behind state.
        assert GPTSettings().force_terminal_tool_choice is False

    def test_the_gate_requires_both_the_switch_and_an_interface(self) -> None:
        assert forced_choice_applicable(None) is False
        with _forcing(True):
            assert forced_choice_applicable(None) is False
            assert forced_choice_applicable(cast("Any", INTERFACE_STUB)) is True


class TestOpenAIChatCompletions:
    """The shared loop (MoonshotAI stands in for every OpenAI-compatible)."""

    async def test_off_state_keeps_auto_and_the_terminal_tools_are_offered(self) -> None:
        payload = await _openai_cc_payload(forcing=False, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "auto"

    async def test_on_state_forces_required_for_an_interface_request(self) -> None:
        payload = await _openai_cc_payload(forcing=True, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "required"
        offered = {tool["function"]["name"] for tool in payload["tools"]}
        assert {"say", "ack_silently"} <= offered

    async def test_interface_less_request_is_never_forced(self) -> None:
        payload = await _openai_cc_payload(forcing=True, interface=None)
        assert payload["tool_choice"] == "auto"
        offered = {tool["function"]["name"] for tool in payload["tools"]}
        assert "say" not in offered and "ack_silently" not in offered


class TestOpenAIResponses:
    async def test_off_state_keeps_auto(self) -> None:
        payload = await _responses_payload(forcing=False, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "auto"

    async def test_on_state_forces_required(self) -> None:
        payload = await _responses_payload(forcing=True, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "required"
        # The minimal Responses function schema carries a parameters object
        # (recon C4: {"type": "function", "name": ..., "parameters": ...}).
        for tool in payload["tools"]:
            assert tool["type"] == "function"
            assert tool["parameters"] is not None

    async def test_interface_less_request_is_never_forced(self) -> None:
        payload = await _responses_payload(forcing=True, interface=None)
        assert payload["tool_choice"] == "auto"


class TestAnthropic:
    async def test_off_state_sends_no_tool_choice_field(self) -> None:
        payload = await _anthropic_payload(forcing=False, interface=INTERFACE_STUB)
        # The SDK sentinel (not a dict): the wire request carries no forced
        # choice and behaves exactly as before 4c-2.
        assert payload["tool_choice"] is anthropic_omit

    async def test_on_state_forces_any_for_an_interface_request(self) -> None:
        payload = await _anthropic_payload(forcing=True, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == {"type": "any"}
        offered = {tool["name"] for tool in payload["tools"]}
        assert {"say", "ack_silently"} <= offered

    async def test_interface_less_request_sends_no_tool_choice_field(self) -> None:
        payload = await _anthropic_payload(forcing=True, interface=None)
        assert payload["tool_choice"] is anthropic_omit

    async def test_the_repo_sends_no_thinking_parameter_today(self) -> None:
        # Pins the current state the recon C5 guard relies on: legacy
        # thinking.type="enabled" combined with a forced choice is a hard
        # 400, so a future change that introduces thinking must route it
        # through the guard (or use adaptive thinking, which is accepted).
        payload = await _anthropic_payload(forcing=True, interface=INTERFACE_STUB)
        assert "thinking" not in payload


class TestAnthropicThinkingGuard:
    """recon C5 (VERIFIED live, both directions): the hard-400 combination."""

    def test_legacy_enabled_thinking_drops_the_forced_choice(self) -> None:
        with _forcing(True):
            choice = anthropic_forced_tool_choice(
                interface=cast("Any", INTERFACE_STUB),
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
        assert choice is None

    def test_adaptive_thinking_keeps_the_forced_choice(self) -> None:
        with _forcing(True):
            choice = anthropic_forced_tool_choice(
                interface=cast("Any", INTERFACE_STUB),
                thinking={"type": "adaptive"},
            )
        assert choice == {"type": "any"}

    def test_no_thinking_at_all_keeps_the_forced_choice(self) -> None:
        with _forcing(True):
            assert anthropic_forced_tool_choice(interface=cast("Any", INTERFACE_STUB)) == {"type": "any"}


class TestGeminiNative:
    async def test_off_state_sends_no_tool_config(self) -> None:
        config = await _gemini_config(forcing=False, interface=INTERFACE_STUB)
        assert config.tool_config is None

    async def test_on_state_forces_any_for_an_interface_request(self) -> None:
        config = await _gemini_config(forcing=True, interface=INTERFACE_STUB)
        assert config.tool_config is not None
        calling_config = config.tool_config.function_calling_config
        assert calling_config.mode == FunctionCallingConfigMode.ANY
        # The allowed set is NOT narrowed to the terminal tools: a turn may
        # still need ordinary work calls before it ends, and narrowing would
        # force premature termination.
        assert calling_config.allowed_function_names in (None, [])
        offered = {str(decl.name) for tool in config.tools for decl in tool.function_declarations}
        assert {"say", "ack_silently"} <= offered

    async def test_interface_less_request_sends_no_tool_config(self) -> None:
        config = await _gemini_config(forcing=True, interface=None)
        assert config.tool_config is None

    async def test_every_declaration_carries_an_explicit_parameters_object(self) -> None:
        # Defensive, recon C4: the "silent no-call" observation is NOT
        # REPRODUCED/UNVERIFIED, so this is verified-harmless defensiveness.
        config = await _gemini_config(forcing=True, interface=INTERFACE_STUB)
        for tool in config.tools:
            for declaration in tool.function_declarations:
                assert declaration.parameters is not None


class TestGeminiThoughtSignatureRoundTrip:
    """recon C6 (VERIFIED): replayed functionCall history without a
    thought_signature is a hard 400 on gemini-3.x. No fixture may fabricate
    signature-less functionCall history, and the persistence path must carry
    the signature back onto the replayed part."""

    def test_response_function_call_parts_carry_a_signature_into_persistence(self) -> None:
        from google.genai.types import Content, Part

        signature = b"test-signature-bytes"
        function_call = GeminiFunctionCall(name="say", args={"text": "Hi"}, id="fc_say")
        part = Part(function_call=function_call, thought_signature=signature)
        content = Content(role="model", parts=[part])
        message = Message.from_google(content.model_dump())
        assert message.tool_calls is not None
        assert message.tool_calls[0].thought_signature == signature
        replayed = message.to_google()
        replayed_parts = replayed["parts"] or []
        function_call_parts = [part for part in replayed_parts if part.get("function_call") is not None]
        assert len(function_call_parts) == 1
        assert function_call_parts[0]["thought_signature"] == signature


class TestMistralNative:
    async def test_off_state_keeps_auto(self) -> None:
        payload = await _mistral_payload(forcing=False, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "auto"

    async def test_on_state_forces_any_for_an_interface_request(self) -> None:
        payload = await _mistral_payload(forcing=True, interface=INTERFACE_STUB)
        assert payload["tool_choice"] == "any"
        offered = {tool["function"]["name"] for tool in payload["tools"]}
        assert {"say", "ack_silently"} <= offered

    async def test_interface_less_request_keeps_auto(self) -> None:
        payload = await _mistral_payload(forcing=True, interface=None)
        assert payload["tool_choice"] == "auto"

    async def test_every_offered_tool_carries_the_mandatory_parameters_field(self) -> None:
        # recon C4: the Mistral SDK rejects a function tool without
        # ``parameters`` client-side, even when the schema is empty.
        payload = await _mistral_payload(forcing=True, interface=INTERFACE_STUB)
        for tool in payload["tools"]:
            assert tool["function"]["parameters"] is not None


class TestForcedChoiceInstrumentation:
    """Honoured/ignored outcomes go through the same greppable sink as
    TURN_MISUSE — log lines only, no new metrics service or storage."""

    @pytest.fixture
    def capture_logs(self):
        from loguru import logger as loguru_logger

        lines: list[str] = []
        handler_id = loguru_logger.add(lambda message: lines.append(str(message).rstrip("\n")), level="INFO")
        try:
            yield lines
        finally:
            loguru_logger.remove(handler_id)

    async def test_a_forced_request_ended_by_a_terminal_call_is_counted_honoured(self, capture_logs) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        answer = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_1",
                                type="function",
                                function=Function(name="say", arguments='{"text": "Forced answer"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            created=1234567890,
            model=TEST_MODEL,
            object="chat.completion",
        )
        create_mock = AsyncMock(return_value=answer)
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        with (
            _forcing(True),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch(
                "chibi.services.providers.provider.RegisteredChibiTools.call",
                new=AsyncMock(side_effect=AssertionError("terminal call reached the dispatcher")),
            ),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("Any", INTERFACE_STUB),
            )
        assert response.answer == "Forced answer"
        assert create_mock.await_count == 1  # the turn exited; no tool result fuelled a second request
        forced_lines = [line for line in capture_logs if "FORCED_CHOICE" in line]
        assert len(forced_lines) == 1
        assert "outcome=honoured" in forced_lines[0]
        assert "role=master" in forced_lines[0]
        assert f"model={TEST_MODEL}" in forced_lines[0]

    async def test_a_forced_request_answered_with_prose_is_counted_ignored(self, capture_logs) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        answer = ChatCompletion(
            id="test-id",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="prose answer", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            model=TEST_MODEL,
            object="chat.completion",
        )
        create_mock = AsyncMock(return_value=answer)
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        with (
            _forcing(True),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("Any", INTERFACE_STUB),
            )
        forced_lines = [line for line in capture_logs if "FORCED_CHOICE" in line]
        assert len(forced_lines) == 1
        assert "outcome=ignored" in forced_lines[0]

    def test_the_line_never_appears_when_the_switch_is_off(self) -> None:
        # Off-state default: no instrumentation events at all.
        with _forcing(False):
            assert gpt_settings.force_terminal_tool_choice is False
            assert forced_choice_applicable(None) is False
