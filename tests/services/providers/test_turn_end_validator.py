"""Tests for the turn-end validator, repair round-trip and graceful degradation
(sub-task 4c).

The validator is the guarantee of the mechanical-silence contract: forcing is
only an optimisation. At every turn-end seam it judges the OUTCOME (not the
offer): (a) an interface turn with plain text and no terminal call degrades to
visible (behaviour preserved); (b) a turn that ended with nothing delivered
costs one bounded repair round-trip and then a loud failure; (c) an
ack_silently() with nothing pending is a misuse; and an interface-less turn
may end only with a non-empty text report, wherever its emptiness came from —
including a hallucinated terminal call, because terminal interception is
offer-blind.
"""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import Usage
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from google.genai.types import FunctionCall as GeminiFunctionCall
from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.exceptions import SilentTurnError
from chibi.models import Message, User
from chibi.services.interface import UserInterface
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI as ChibiOpenAI
from chibi.services.providers.terminal_calls import (
    REPAIR_NUDGE_TYPE,
    background_work_pending,
    detect_wake_type,
    evaluate_turn_end,
    honored_terminal_name,
    log_turn_misuse,
    repair_nudge_text,
)
from chibi.services.providers.tools.schemas import ToolCallSchema

TEST_TOKEN = "test-token-for-turn-end-validator"
TEST_MODEL = "kimi-k2.6"
TEST_OPENAI_MODEL = "gpt-5.6-terra"

INTERFACE_STUB = SimpleNamespace(thread_id=1, streaming_enabled=False, delta_emitted=False, captures_llm_thoughts=False)


def _interface() -> UserInterface:
    return cast("UserInterface", INTERFACE_STUB)


def _wake(wake_type: str = "user message") -> str:
    return json.dumps({"type": wake_type, "desc": "wake", "datetime_now": "2026-10-20 12:00:00 UTC+0000"})


def _conversation(*contents: str) -> list[Message]:
    return [Message(role="user", content=content) for content in contents]


# ---------------------------------------------------------------------------
# Shared-predicate unit coverage
# ---------------------------------------------------------------------------


class TestEvaluateTurnEnd:
    """Unit coverage for the shared ending rule."""

    def test_say_turn_with_text_is_ok_for_both_roles(self) -> None:
        assert (
            evaluate_turn_end(
                interface=_interface(), answer="Hi", ended_by_terminal_call=True, terminal_tool_name="say"
            ).outcome
            == "ok"
        )
        assert (
            evaluate_turn_end(
                interface=None, answer="report", ended_by_terminal_call=True, terminal_tool_name="say"
            ).outcome
            == "ok"
        )

    def test_case_a_interface_plain_text_degrades_to_visible(self) -> None:
        # Today's behaviour preserved: delivered, and measured as under-use.
        verdict = evaluate_turn_end(
            interface=_interface(), answer="plain answer", ended_by_terminal_call=False, terminal_tool_name=None
        )
        assert verdict.outcome == "degraded_visible"
        assert verdict.misuse == "no_terminal_call_text"

    def test_interface_less_plain_text_report_is_ok(self) -> None:
        verdict = evaluate_turn_end(
            interface=None, answer="plain report", ended_by_terminal_call=False, terminal_tool_name=None
        )
        assert verdict.outcome == "ok"
        assert verdict.misuse is None

    def test_case_b_empty_ending_is_a_misuse_for_both_roles(self) -> None:
        for interface in (None, _interface()):
            verdict = evaluate_turn_end(
                interface=interface, answer="", ended_by_terminal_call=False, terminal_tool_name=None
            )
            assert verdict.outcome == "misuse"
            assert verdict.misuse == "empty_turn"

    def test_empty_say_is_an_empty_ending(self) -> None:
        verdict = evaluate_turn_end(
            interface=_interface(), answer="", ended_by_terminal_call=True, terminal_tool_name="say"
        )
        assert verdict.outcome == "misuse"
        assert verdict.misuse == "empty_turn"

    def test_interface_less_ack_is_a_hallucinated_terminal_call(self) -> None:
        # Class-aware ending rule: silence has no receiver one floor down.
        verdict = evaluate_turn_end(
            interface=None, answer="", ended_by_terminal_call=True, terminal_tool_name="ack_silently"
        )
        assert verdict.outcome == "misuse"
        assert verdict.misuse == "hallucinated_ack"

    def test_case_c_ack_without_pending_work_is_over_eager(self) -> None:
        verdict = evaluate_turn_end(
            interface=_interface(),
            answer="",
            ended_by_terminal_call=True,
            terminal_tool_name="ack_silently",
            conversation=_conversation(_wake("user message")),
        )
        assert verdict.outcome == "misuse"
        assert verdict.misuse == "overeager_ack"

    def test_tool_response_wake_makes_the_ack_legitimate(self) -> None:
        verdict = evaluate_turn_end(
            interface=_interface(),
            answer="",
            ended_by_terminal_call=True,
            terminal_tool_name="ack_silently",
            conversation=_conversation(_wake("tool response")),
        )
        assert verdict.outcome == "ok"

    def test_scheduler_wake_makes_the_ack_legitimate(self) -> None:
        verdict = evaluate_turn_end(
            interface=_interface(),
            answer="",
            ended_by_terminal_call=True,
            terminal_tool_name="ack_silently",
            conversation=_conversation(_wake("scheduled_trigger")),
        )
        assert verdict.outcome == "ok"

    def test_unanswered_background_dispatch_makes_the_ack_legitimate(self) -> None:
        dispatch = json.dumps({"tool_name": "delegate_task", "status": "tool is running in background", "result": {}})
        verdict = evaluate_turn_end(
            interface=_interface(),
            answer="",
            ended_by_terminal_call=True,
            terminal_tool_name="ack_silently",
            conversation=[Message(role="user", content=_wake("user message")), Message(role="tool", content=dispatch)],
        )
        assert verdict.outcome == "ok"

    def test_unrecognized_wake_is_treated_strictly(self) -> None:
        verdict = evaluate_turn_end(
            interface=_interface(),
            answer="",
            ended_by_terminal_call=True,
            terminal_tool_name="ack_silently",
            conversation=_conversation("plain text, not a wake prompt"),
        )
        assert verdict.outcome == "misuse"
        assert verdict.misuse == "overeager_ack"


class TestWakeAndPendingDetection:
    """Unit coverage for the wake-type and pending-background-work signals."""

    def test_detect_wake_type_reads_the_last_user_message(self) -> None:
        conversation = [
            Message(role="user", content=_wake("user message")),
            Message(role="assistant", content="working"),
            Message(role="user", content=_wake("scheduled_trigger")),
        ]
        assert detect_wake_type(conversation) == "scheduled_trigger"

    def test_detect_wake_type_skips_tool_result_messages(self) -> None:
        dispatch = json.dumps({"tool_name": "delegate_task", "status": "tool is running in background", "result": {}})
        conversation = [
            Message(role="user", content=_wake("user message")),
            Message(role="tool", content=dispatch, tool_call_id="t1"),
        ]
        assert detect_wake_type(conversation) == "user message"

    def test_detect_wake_type_returns_none_for_non_json_prompts(self) -> None:
        assert detect_wake_type(_conversation("Hi")) is None
        assert detect_wake_type([]) is None

    def test_pending_background_work_requires_an_unanswered_dispatch(self) -> None:
        dispatch = json.dumps({"tool_name": "delegate_task", "status": "tool is running in background", "result": {}})
        assert background_work_pending([Message(role="tool", content=dispatch, tool_call_id="t1")]) is True

        result_wake = json.dumps(
            {
                "type": "tool response",
                "desc": "background task is done",
                "tool_name": "delegate_task",
                "tool_response": {"tool_name": "delegate_task", "status": "ok", "result": {}},
            }
        )
        conversation = [
            Message(role="tool", content=dispatch, tool_call_id="t1"),
            Message(role="user", content=result_wake),
        ]
        assert background_work_pending(conversation) is False

    def test_plain_text_history_is_never_pending(self) -> None:
        assert background_work_pending(_conversation("Hi", "there")) is False


class TestHonoredTerminalNameAndNudge:
    def test_say_is_honoured_regardless_of_position(self) -> None:
        say = ToolCallSchema(tool_name="say", args={"text": "x"})
        ack = ToolCallSchema(tool_name="ack_silently", args={})
        assert honored_terminal_name([ack, say]) == "say"
        assert honored_terminal_name([ack, ack]) == "ack_silently"

    def test_repair_nudge_is_a_json_wake_prompt_per_role(self) -> None:
        for interface_flag, expected_fragment in ((True, "say"), (False, "plain text")):
            payload = json.loads(repair_nudge_text("empty_turn", interface=interface_flag))
            assert payload["type"] == REPAIR_NUDGE_TYPE
            assert expected_fragment in payload["desc"]


# ---------------------------------------------------------------------------
# Behavioural coverage through the real shared loop (OpenAI-compatible seam)
# ---------------------------------------------------------------------------


def _completion_with_answer(answer: str) -> ChatCompletion:
    choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=answer, tool_calls=None),
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _completion_with_tool_calls(tool_calls: list[ChatCompletionMessageToolCall], content: str = "") -> ChatCompletion:
    choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason="tool_calls",
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


class _CaptureLogs:
    """Collect formatted loguru lines between setup and teardown."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._handler_id: int | None = None

    def __enter__(self) -> "list[str]":
        self._handler_id = logger.add(lambda message: self.lines.append(str(message).rstrip("\n")), level="INFO")
        return self.lines

    def __exit__(self, *exc: Any) -> None:
        assert self._handler_id is not None
        logger.remove(self._handler_id)


class TestTurnEndValidatorSharedLoop:
    """Behavioural tests through the shared OpenAI-compatible loop (MoonshotAI)."""

    @pytest.mark.asyncio
    async def test_case_a_interface_turn_with_plain_text_still_delivers(self) -> None:
        # Graceful degradation: no terminal call, non-empty text, nothing
        # pending — the text is treated as visible exactly as today, with no
        # repair round-trip and no second model call.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("plain answer"))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=_wake("user message")).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=_interface(),
                conversation_messages=conversation,
            )

        assert response.answer == "plain answer"
        assert create_mock.await_count == 1
        assert [msg for msg in conversation if msg.role == "assistant"]

    @pytest.mark.asyncio
    async def test_case_b_interface_empty_ending_costs_one_repair_and_recovers(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(side_effect=[_completion_with_answer(""), _completion_with_answer("Recovered answer")])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=_wake("user message")).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=_interface(),
                conversation_messages=conversation,
            )

        assert response.answer == "Recovered answer"
        # Exactly one repair round-trip: the model was contacted twice.
        assert create_mock.await_count == 2
        # The repair nudge is a JSON wake prompt persisted in the conversation.
        nudges = [
            msg
            for msg in conversation
            if msg.role == "user" and json.loads(msg.content).get("type") == REPAIR_NUDGE_TYPE
        ]
        assert len(nudges) == 1

    @pytest.mark.asyncio
    async def test_repair_cap_a_model_that_misbehaves_every_time_fails_loudly(self) -> None:
        # The cap proof: the third call must never happen. If the cap is
        # removed, the sentinel AssertionError on the third response fires
        # instead of SilentTurnError and this test fails.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            side_effect=[
                _completion_with_answer(""),
                _completion_with_answer(""),
                AssertionError("repair cap exceeded"),
            ]
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[Message(role="user", content=_wake("user message")).to_openai()],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=1,
                    system_prompt="",
                    interface=_interface(),
                    conversation_messages=conversation,
                )

        assert create_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_case_c_over_eager_ack_costs_one_repair_then_say_recovers(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            side_effect=[
                _completion_with_tool_calls([_tool_call("ack_silently", "{}")]),
                _completion_with_tool_calls([_tool_call("say", '{"text": "Here you go"}')]),
            ]
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(
                MoonshotAI,
                "call_functions",
                AsyncMock(side_effect=AssertionError("a terminal call reached the dispatcher")),
            ),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=_wake("user message")).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=_interface(),
                conversation_messages=conversation,
            )

        assert response.answer == "Here you go"
        assert create_mock.await_count == 2
        # No tool result may exist for either terminal call.
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_case_c_over_eager_ack_that_acks_again_fails_loudly(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        ack_response = _completion_with_tool_calls([_tool_call("ack_silently", "{}")])
        create_mock = AsyncMock(side_effect=[ack_response, ack_response, AssertionError("repair cap exceeded")])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[Message(role="user", content=_wake("user message")).to_openai()],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=1,
                    system_prompt="",
                    interface=_interface(),
                    conversation_messages=conversation,
                )

        assert create_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_legitimate_ack_after_a_scheduler_wake_needs_no_repair(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="nothing to report")
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("scheduled_trigger"))]

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=_wake("scheduled_trigger")).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=_interface(),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert create_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].visibility == "hidden"


# ---------------------------------------------------------------------------
# Per-seam reach: the validator is reachable from every adapter turn-end seam
# ---------------------------------------------------------------------------


def _anthropic_message(text: str | None) -> AnthropicMessage:
    from anthropic.types import TextBlock

    content = [TextBlock(type="text", text=text)] if text is not None else []
    return AnthropicMessage(
        id="msg_1",
        role="assistant",
        model=TEST_MODEL,
        content=content,
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class TestValidatorReachesEverySeam:
    """One real-path test per remaining adapter seam (OpenAI CC is above)."""

    @pytest.mark.asyncio
    async def test_anthropic_interface_less_empty_report_fails_loudly_after_one_repair(self) -> None:
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        generate_mock = AsyncMock(
            side_effect=[
                _anthropic_message(None),
                _anthropic_message(None),
                AssertionError("repair cap exceeded"),
            ]
        )
        conversation: list[Message] = [Message(role="user", content="task prompt")]

        with (
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Anthropic, "_generate_content", generate_mock),
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[{"role": "user", "content": "task prompt"}],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    system_prompt="",
                    conversation_messages=conversation,
                )

        assert generate_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_gemini_interface_less_hallucinated_ack_fails_loudly_after_one_repair(self) -> None:
        provider = Gemini(token=TEST_TOKEN)
        generate_mock = AsyncMock(
            side_effect=[
                _gemini_response(None, [("ack_silently", {})]),
                _gemini_response(None, [("ack_silently", {})]),
                AssertionError("repair cap exceeded"),
            ]
        )
        conversation: list[Message] = [Message(role="user", content="task prompt")]

        with (
            patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Gemini, "_generate_content", generate_mock),
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[],
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    model=TEST_MODEL,
                    system_prompt="",
                    conversation_messages=conversation,
                )

        assert generate_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_mistral_interface_less_empty_report_fails_loudly_after_one_repair(self) -> None:
        provider = MistralAI(token=TEST_TOKEN)
        empty_message = SimpleNamespace(content="", tool_calls=None)
        usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        empty_response = SimpleNamespace(choices=[SimpleNamespace(message=empty_message)], usage=usage)
        generate_mock = AsyncMock(side_effect=[empty_response, empty_response, AssertionError("repair cap exceeded")])
        conversation: list[Message] = [Message(role="user", content="task prompt")]

        with (
            patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(MistralAI, "_generate_content", generate_mock),
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[Message(role="user", content="task prompt").to_mistral()],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    system_prompt="",
                    conversation_messages=conversation,
                )

        assert generate_mock.await_count == 2

    @pytest.mark.asyncio
    async def test_responses_interface_empty_ending_costs_one_repair_and_recovers(self) -> None:
        provider = ChibiOpenAI(token=TEST_TOKEN)

        def _response(output_text: str) -> SimpleNamespace:
            response = SimpleNamespace()
            response.output = [SimpleNamespace(type="message")]
            response.output_text = output_text
            response.usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
            return response

        create_mock = AsyncMock(side_effect=[_response(""), _response("Recovered report")])
        mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
        provider.client = mock_client
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")):
            response, _messages = await provider._get_response_completion_response(
                messages=conversation,
                model=TEST_OPENAI_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=_interface(),
            )

        assert response.answer == "Recovered report"
        assert create_mock.await_count == 2
        assert any(
            msg.role == "user" and json.loads(msg.content).get("type") == REPAIR_NUDGE_TYPE for msg in conversation
        )


def _gemini_response(text: str | None, function_calls: list[tuple[str, dict[str, Any]]]) -> GenerateContentResponse:
    parts: list[Any] = []
    if text:
        parts.append(Part(text=text))
    for name, payload in function_calls:
        parts.append(Part(function_call=GeminiFunctionCall(name=name, args=payload, id=f"fc_{name}")))
    return GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=parts))])


# ---------------------------------------------------------------------------
# Misuse instrumentation
# ---------------------------------------------------------------------------


class TestMisuseInstrumentation:
    """The logger is the counter sink: one greppable line per event."""

    def test_log_line_format_is_greppable_per_model_and_role(self) -> None:
        with _CaptureLogs() as lines:
            log_turn_misuse(
                model="m-1", provider_name="OpenAI", interface=None, kind="hallucinated_ack", action="repair"
            )
            log_turn_misuse(model="m-1", provider_name="OpenAI", interface=_interface(), kind=None, action="recovered")
        assert any(
            "TURN_MISUSE" in line
            and "model=m-1" in line
            and "role=delegated" in line
            and "kind=hallucinated_ack" in line
            and "action=repair" in line
            and "provider=OpenAI" in line
            for line in lines
        )
        assert any("role=master" in line and "action=recovered" in line for line in lines)

    @pytest.mark.asyncio
    async def test_repair_and_loud_failure_are_counted_on_the_real_path(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        empty = _completion_with_answer("")
        create_mock = AsyncMock(side_effect=[empty, empty, AssertionError("repair cap exceeded")])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content="task prompt")]

        with _CaptureLogs() as lines:
            with (
                patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
                patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            ):
                with pytest.raises(SilentTurnError):
                    await provider._get_chat_completion_response(
                        messages=[Message(role="user", content="task prompt").to_openai()],
                        model=TEST_MODEL,
                        user=User(id=1),
                        caller_storage_id=1,
                        caller_thread_id=0,
                        system_prompt="",
                        conversation_messages=conversation,
                    )

        events = [line for line in lines if "TURN_MISUSE" in line]
        assert any(
            f"model={TEST_MODEL}" in line and "role=delegated" in line and "action=repair" in line for line in events
        )
        assert any(f"model={TEST_MODEL}" in line and "action=loud_failure" in line for line in events)

    @pytest.mark.asyncio
    async def test_degraded_visible_ending_is_counted_on_the_real_path(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("plain answer"))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content=_wake("user message"))]

        with _CaptureLogs() as lines:
            with (
                patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
                patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            ):
                await provider._get_chat_completion_response(
                    messages=[Message(role="user", content=_wake("user message")).to_openai()],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=1,
                    system_prompt="",
                    interface=_interface(),
                    conversation_messages=conversation,
                )

        assert any(
            "TURN_MISUSE" in line
            and f"model={TEST_MODEL}" in line
            and "role=master" in line
            and "kind=no_terminal_call_text" in line
            and "action=degraded_visible" in line
            for line in lines
        )
