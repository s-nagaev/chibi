"""Tests for the say()/ack_silently() terminal-call interception (sub-task 4b).

The invariant under test: a terminal call ends the turn in every adapter
path, is normalized at ingest into a plain assistant ``Message``, and NEVER
receives a tool result — a result is the fuel of the infinite
``say() → result → ack_silently() → ack_silently() → ...`` loop (recon C7).

Since sub-task 4c the turn-end validator judges every ending: a legitimate
``ack_silently()`` (interface turn absorbing a background event) still
persists the hidden message exactly as 4b normalized it, while an
interface-less or over-eager ack now costs a repair round-trip and then a
loud failure — those paths are covered by
``tests/services/providers/test_turn_end_validator.py``.
"""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from google.genai.types import FunctionCall as GeminiFunctionCall
from mistralai.client.models import FunctionCall as MistralFunctionCall
from mistralai.client.models import ToolCall as MistralToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import ResponseFunctionToolCall

import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.exceptions import SilentTurnError
from chibi.models import Message, User
from chibi.services.interface import UserInterface
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI as ChibiOpenAI
from chibi.services.providers.terminal_calls import (
    TERMINAL_TOOL_NAMES,
    background_work_pending,
    fold_terminal_calls_into_thoughts,
    offered_tool_definitions,
    offered_tools,
    partition_terminal_calls,
    plan_terminal_turn,
)
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.schemas import ToolCallSchema, ToolResponseSchema

TEST_TOKEN = "test-token-for-terminal-calls"
TEST_MODEL = "kimi-k2.6"
TEST_OPENAI_MODEL = "gpt-5.6-terra"

# Interface stub for validator-era ack tests: an ack ending is legitimate
# only for an interface turn absorbing a background event, so the ack
# fixtures carry the same JSON tool-response wake the master path appends.
INTERFACE_STUB = SimpleNamespace(thread_id=1, streaming_enabled=False, delta_emitted=False, captures_llm_thoughts=False)


def _tool_response_wake(tool_name: str = "delegate_task") -> str:
    """The JSON wake prompt a completed background task produces (user.py)."""
    return json.dumps(
        {
            "type": "tool response",
            "desc": "background task is done",
            "tool_name": tool_name,
            "tool_response": {"tool_name": tool_name, "status": "ok", "result": {}},
            "datetime_now": "2026-10-20 12:00:00 UTC+0000",
        }
    )


def _user_message_wake() -> str:
    """The JSON wake prompt a plain user message produces (user.py)."""
    return json.dumps(
        {
            "type": "user message",
            "user": {"first_name": "Tester"},
            "prompt": "do the thing",
            "datetime_now": "2026-10-20 12:00:00 UTC+0000",
        }
    )


def _conversation_with_delivered_answer(wake: str) -> list[Message]:
    """A user turn that already delivered a visible answer, then the wake.

    Under the answer-owed admission rule a silent ack is legitimate when the
    user already has the substance of the user turn, so the ack fixtures seed
    a delivered visible answer before the tool-response wake. A bare
    tool-response wake with nothing delivered and nothing pending is exactly
    the over-eager ack the validator must refuse.
    """
    return [
        Message(role="user", content=_user_message_wake()),
        Message(role="assistant", content="The substantive answer", visibility="visible"),
        Message(role="user", content=wake),
    ]


def _say_call(text: str = "Hello there") -> ToolCallSchema:
    return ToolCallSchema(tool_name="say", args={"text": text})


def _ack_call() -> ToolCallSchema:
    return ToolCallSchema(tool_name="ack_silently", args={})


def _work_call() -> ToolCallSchema:
    return ToolCallSchema(tool_name="get_current_datetime", args={})


class TestTerminalCallHelpers:
    """Unit coverage for the shared interception helpers."""

    def test_partition_splits_terminal_from_work_calls(self) -> None:
        terminal, work = partition_terminal_calls([_work_call(), _say_call(), _ack_call(), _work_call()])
        assert [call.tool_name for call in terminal] == ["say", "ack_silently"]
        assert [call.tool_name for call in work] == ["get_current_datetime", "get_current_datetime"]

    def test_partition_of_pure_work_turn_is_empty(self) -> None:
        terminal, work = partition_terminal_calls([_work_call()])
        assert terminal == []
        assert len(work) == 1

    def test_every_registered_terminal_tool_name_is_known(self) -> None:
        assert TERMINAL_TOOL_NAMES == {"say", "ack_silently"}

    def test_plan_say_turn_persists_a_visible_message_carrying_the_text(self) -> None:
        plan = plan_terminal_turn(terminal_calls=[_say_call(text="Answer")], thoughts="private CoT")
        assert plan.answer == "Answer"
        assert plan.message is not None
        assert plan.message.role == "assistant"
        assert plan.message.content == "Answer"
        assert plan.message.visibility == "visible"
        assert plan.message.tool_calls is None

    def test_plan_say_turn_with_empty_text_omits_the_message(self) -> None:
        plan = plan_terminal_turn(terminal_calls=[_say_call(text="")], thoughts="CoT")
        assert plan.answer == ""
        assert plan.message is None

    def test_plan_ack_turn_persists_a_hidden_message_with_the_cot(self) -> None:
        plan = plan_terminal_turn(terminal_calls=[_ack_call()], thoughts="still waiting for results")
        assert plan.answer == ""
        assert plan.message is not None
        assert plan.message.content == "still waiting for results"
        assert plan.message.visibility == "hidden"
        assert plan.message.tool_calls is None

    def test_plan_degenerate_ack_turn_omits_the_assistant_turn(self) -> None:
        # Design R2: never persist an empty text block — omit the turn.
        plan = plan_terminal_turn(terminal_calls=[_ack_call()], thoughts=None)
        assert plan.answer == ""
        assert plan.message is None

    def test_plan_ignores_redundant_terminal_calls(self) -> None:
        plan = plan_terminal_turn(terminal_calls=[_ack_call(), _ack_call()], thoughts="CoT")
        assert plan.message is not None
        assert plan.message.visibility == "hidden"

    def test_fold_appends_the_say_text_to_the_thoughts(self) -> None:
        folded = fold_terminal_calls_into_thoughts("thinking", [_say_call(text="answer text")])
        assert folded == "thinking\nanswer text"

    def test_fold_without_a_say_call_keeps_the_thoughts(self) -> None:
        assert fold_terminal_calls_into_thoughts("thinking", [_ack_call()]) == "thinking"
        assert fold_terminal_calls_into_thoughts(None, [_ack_call()]) == ""


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


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def _forbid_terminal_dispatch() -> Any:
    """Patch both dispatch seams; any await fails the no-tool-result invariant."""
    registry_mock = AsyncMock(side_effect=AssertionError("A terminal call reached the dispatcher"))
    call_functions_mock = AsyncMock(side_effect=AssertionError("call_functions executed a terminal call"))
    return (
        patch.object(RegisteredChibiTools, "call", registry_mock),
        call_functions_mock,
    )


@pytest.mark.asyncio
async def test_dispatcher_still_manufactures_a_result_when_reached() -> None:
    """Documents the pre-4b failure mode: reaching the dispatcher feeds the loop.

    Before the interception existed, a terminal call went through
    RegisteredChibiTools.call and came back as an error tool result
    (NotImplementedError), which was appended to the history and sent back to
    the model — the exact shape reconnaissance reproduced as an infinite
    say/ack loop (recon C7). The interception must make that path
    unreachable; this test only pins what the dispatcher itself would do.
    """
    result = await RegisteredChibiTools.call(tool_name="say", tools_args={"text": "Hi"})
    assert result.status == "error"


class TestSharedOpenAICompatibleLoop:
    """MoonshotAI stands in for every OpenAI-compatible chat-completions loop."""

    @pytest.mark.asyncio
    async def test_say_call_ends_the_turn_without_a_tool_result(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("say", '{"text": "Final answer"}')])
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                conversation_messages=conversation,
            )

        assert response.answer == "Final answer"
        # The turn exited: the model was contacted exactly once — no second
        # request, which is what a manufactured tool result would have caused.
        assert create_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "Final answer"
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_ack_call_ends_the_turn_with_a_hidden_message(self) -> None:
        # Legitimate ack: an interface turn absorbing a trailing background
        # tool result after the user already got the answer. The validator
        # admits it and the 4b normalization holds.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="waiting for results")
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert create_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"]
        assert len(persisted) == 1
        assert persisted[0].content == "waiting for results"
        assert persisted[0].visibility == "hidden"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_degenerate_ack_call_omits_the_assistant_turn(self) -> None:
        # Legitimate ack with nothing to persist: design R2 omits the turn.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="")
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"] == []

    @pytest.mark.asyncio
    async def test_get_chat_response_returns_the_normalized_messages_with_visibility(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            side_effect=[
                _completion_with_tool_calls([_tool_call("say", '{"text": "Delivered"}')]),
            ]
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, new_messages = await provider.get_chat_response(
                messages=conversation,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                model=TEST_MODEL,
                system_prompt="base",
            )

        assert response.answer == "Delivered"
        assert [msg.visibility for msg in new_messages] == ["visible"]
        assert new_messages[0].content == "Delivered"
        assert new_messages[0].role == "assistant"

    @pytest.mark.asyncio
    async def test_terminal_call_does_not_fuel_the_ack_loop(self) -> None:
        # Recon C7 loop shape: the model emits ack_silently forever as long as
        # it keeps receiving a tool result. With the interception in place the
        # first legitimate terminal call must end the turn regardless of what
        # the mock would keep returning.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="thoughts")
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            await provider._get_chat_completion_response(
                messages=[Message(role="user", content=wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert create_mock.await_count == 1
        # (b) A normalized message is persisted, with its visibility respected.
        persisted = [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"]
        assert len(persisted) == 1
        assert persisted[0].content == "thoughts"
        assert persisted[0].visibility == "hidden"
        assert persisted[0].tool_calls is None
        # (c) No tool result exists for the terminal call.
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_mixed_turn_runs_work_calls_and_drops_the_terminal_entry(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        mixed = _completion_with_tool_calls(
            [
                _tool_call("get_current_datetime", "{}", call_id="call_work"),
                _tool_call("say", '{"text": "should not be executed"}', call_id="call_say"),
            ],
            content="working",
        )
        create_mock = AsyncMock(side_effect=[mixed, _completion_with_answer("Done")])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        call_functions_mock = AsyncMock(
            return_value=[ToolResponseSchema(tool_name="get_current_datetime", status="ok", result={})]
        )
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(provider, "call_functions", call_functions_mock),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                conversation_messages=conversation,
            )

        assert response.answer == "Done"
        assert call_functions_mock.await_count == 1
        assert call_functions_mock.await_args is not None
        executed_names = [call.tool_name for call in call_functions_mock.await_args.kwargs["calls"]]
        assert executed_names == ["get_current_datetime"]
        # No say/ack tool call may appear anywhere in the persisted history.
        for msg in conversation:
            for tool_call in msg.tool_calls or []:
                assert tool_call.function.name not in TERMINAL_TOOL_NAMES
        # The terminal text is folded into the assistant content, not lost.
        assert any("should not be executed" in msg.content for msg in conversation if msg.role == "assistant")


def _responses_output(*items: Any) -> SimpleNamespace:
    response = SimpleNamespace()
    response.output = list(items)
    response.output_text = ""
    response.usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
    return response


class TestOpenAIResponsesLoop:
    """The OpenAI Responses API path."""

    @pytest.mark.asyncio
    async def test_say_call_ends_the_turn_without_a_tool_result(self) -> None:
        provider = ChibiOpenAI(token=TEST_TOKEN)
        tool_call = ResponseFunctionToolCall(
            type="function_call",
            name="say",
            arguments='{"text": "Responses answer"}',
            call_id="call_1",
            status="completed",
        )
        create_mock = AsyncMock(return_value=_responses_output(tool_call))
        mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
        provider.client = mock_client
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _messages = await provider._get_response_completion_response(
                messages=conversation,
                model=TEST_OPENAI_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
            )

        assert response.answer == "Responses answer"
        assert create_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "Responses answer"
        assert persisted[0].visibility == "visible"
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_degenerate_ack_call_appends_nothing(self) -> None:
        # Legitimate ack (interface turn absorbing a background result) with
        # nothing to persist: design R2 — no empty assistant message.
        provider = ChibiOpenAI(token=TEST_TOKEN)
        tool_call = ResponseFunctionToolCall(
            type="function_call", name="ack_silently", arguments="{}", call_id="call_1", status="completed"
        )
        create_mock = AsyncMock(return_value=_responses_output(tool_call))
        mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
        provider.client = mock_client
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _messages = await provider._get_response_completion_response(
                messages=conversation,
                model=TEST_OPENAI_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"] == []

    @pytest.mark.asyncio
    async def test_get_chat_response_persists_the_terminal_message(self) -> None:
        # The Responses wrapper used to compare the conversation list with
        # itself and return an empty delta (nothing was ever persisted). The
        # delta over the call is the persisted form now — verified with a
        # terminal message that must carry its visibility.
        provider = ChibiOpenAI(token=TEST_TOKEN)
        tool_call = ResponseFunctionToolCall(
            type="function_call", name="say", arguments='{"text": "Persisted"}', call_id="call_1", status="completed"
        )
        create_mock = AsyncMock(
            side_effect=[
                _responses_output(
                    SimpleNamespace(type="message"),
                    tool_call,
                )
            ]
        )
        mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
        provider.client = mock_client
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, new_messages = await provider.get_chat_response(
                messages=conversation,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                model=TEST_OPENAI_MODEL,
                system_prompt="base",
            )

        assert response.answer == "Persisted"
        assert len(new_messages) == 1
        assert new_messages[0].visibility == "visible"
        assert new_messages[0].content == "Persisted"


def _anthropic_message(
    text: str | None, tool_calls: list[tuple[str, dict[str, Any]]] | None = None
) -> AnthropicMessage:
    content: list[Any] = []
    if text is not None:
        content.append(TextBlock(type="text", text=text))
    for name, payload in tool_calls or []:
        content.append(ToolUseBlock(type="tool_use", id=f"toolu_{name}", name=name, input=payload))
    return AnthropicMessage(
        id="msg_1",
        role="assistant",
        model=TEST_MODEL,
        content=content,
        stop_reason="tool_use" if tool_calls else "end_turn",
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class TestAnthropicLoop:
    """The Anthropic native path, including the empty-text-block rule (R2)."""

    @pytest.mark.asyncio
    async def test_say_call_ends_the_turn_without_a_tool_result(self) -> None:
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_anthropic_message("thinking", [("say", {"text": "Anthropic answer"})]))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Anthropic, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[{"role": "user", "content": "Hi"}],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                conversation_messages=conversation,
            )

        assert response.answer == "Anthropic answer"
        assert generate_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "Anthropic answer"
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_ack_turn_with_thoughts_persists_a_hidden_message(self) -> None:
        # Legitimate ack: interface turn absorbing a trailing background tool
        # result after the user already got the answer.
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_anthropic_message("absorbing the tool result", [("ack_silently", {})]))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Anthropic, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[{"role": "user", "content": wake}],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        persisted = [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"]
        assert len(persisted) == 1
        assert persisted[0].content == "absorbing the tool result"
        assert persisted[0].visibility == "hidden"

    @pytest.mark.asyncio
    async def test_ack_turn_without_thoughts_omits_the_assistant_turn(self) -> None:
        # Design R2: a silent Anthropic turn must not persist an empty text
        # block — the assistant turn is omitted entirely.
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_anthropic_message(None, [("ack_silently", {})]))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Anthropic, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[{"role": "user", "content": wake}],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"] == []


def _gemini_response(text: str | None, function_calls: list[tuple[str, dict[str, Any]]]) -> GenerateContentResponse:
    parts: list[Any] = []
    if text:
        parts.append(Part(text=text))
    for name, payload in function_calls:
        parts.append(Part(function_call=GeminiFunctionCall(name=name, args=payload, id=f"fc_{name}")))
    return GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=parts))])


class TestGeminiLoop:
    """The Gemini native path."""

    @pytest.mark.asyncio
    async def test_say_call_ends_the_turn_without_a_tool_result(self) -> None:
        provider = Gemini(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_gemini_response("thinking", [("say", {"text": "Gemini answer"})]))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Gemini, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[],
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                model=TEST_MODEL,
                system_prompt="",
                conversation_messages=conversation,
            )

        assert response.answer == "Gemini answer"
        assert generate_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "Gemini answer"
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_ack_turn_persists_a_hidden_message_with_the_cot(self) -> None:
        # Legitimate ack: interface turn absorbing a trailing background tool
        # result after the user already got the answer.
        provider = Gemini(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_gemini_response("waiting for more results", [("ack_silently", {})]))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Gemini, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[],
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                model=TEST_MODEL,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        persisted = [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"]
        assert len(persisted) == 1
        assert persisted[0].content == "waiting for more results"
        assert persisted[0].visibility == "hidden"


class TestMistralLoop:
    """The Mistral native path."""

    @pytest.mark.asyncio
    async def test_say_call_ends_the_turn_without_a_tool_result(self) -> None:
        provider = MistralAI(token=TEST_TOKEN)
        message = SimpleNamespace(
            content="thinking",
            tool_calls=[
                MistralToolCall(
                    id="t1", function=MistralFunctionCall(name="say", arguments='{"text": "Mistral answer"}')
                )
            ],
        )
        usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        generate_mock = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(MistralAI, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_mistral()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                conversation_messages=conversation,
            )

        assert response.answer == "Mistral answer"
        assert generate_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "Mistral answer"
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_ack_turn_without_thoughts_omits_the_assistant_turn(self) -> None:
        # Legitimate ack (interface turn absorbing a trailing background
        # result after the user already got the answer) with nothing to
        # persist: design R2 — the assistant turn is omitted.
        provider = MistralAI(token=TEST_TOKEN)
        message = SimpleNamespace(
            content="",
            tool_calls=[MistralToolCall(id="t1", function=MistralFunctionCall(name="ack_silently", arguments="{}"))],
        )
        usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        generate_mock = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage))
        wake = _tool_response_wake()
        conversation: list[Message] = _conversation_with_delivered_answer(wake)

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(MistralAI, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=wake).to_mistral()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "hidden"] == []


class TestVisibilityMetadataOnTheWire:
    """The visibility field is platform metadata and never reaches an API payload."""

    def test_to_openai_strips_the_visibility_field(self) -> None:
        wire = Message(role="assistant", content="hi", visibility="hidden").to_openai()
        assert "visibility" not in dict(wire)

    def test_legacy_message_without_the_field_reads_back_visible(self) -> None:
        legacy = {"role": "assistant", "content": "old message", "id": 1}
        message = Message(**legacy)
        assert message.visibility == "visible"


class TestRoleScopedToolOffers:
    """Ruling D1: the terminal tools are offered iff the request carries a user interface.

    The offer is filtered at the request seam (the completion kwargs built by
    the loops), driven through the single helper
    ``terminal_calls.offered_tools``. Interface-less turns — delegated
    sub-agents at any depth and the emergency summarizer — are offered neither
    ``say`` nor ``ack_silently``.
    """

    interface_stub: Any = SimpleNamespace(thread_id=1, streaming_enabled=False, delta_emitted=False)

    @staticmethod
    def _offered_names(create_mock: AsyncMock) -> set[str]:
        tools = create_mock.await_args.kwargs["tools"]  # type: ignore[union-attr]
        return {definition["function"]["name"] for definition in tools}

    @pytest.mark.asyncio
    async def test_interface_less_request_is_offered_neither_terminal_tool(self) -> None:
        """The sub-agent shape (interface=None): no say/ack_silently in the offered list."""
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("plain report"))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))

        with (
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
                conversation_messages=[Message(role="user", content="Hi")],
            )

        offered = self._offered_names(create_mock)
        assert "say" not in offered
        assert "ack_silently" not in offered
        # Everything else is unaffected: the interface-less list is the global
        # list minus exactly the two terminal tools.
        global_names = {definition["function"]["name"] for definition in RegisteredChibiTools.get_tool_definitions()}
        assert offered == global_names - {"say", "ack_silently"}
        assert "delegate_task" in offered

    @pytest.mark.asyncio
    async def test_interface_request_still_receives_both_terminal_tools(self) -> None:
        """Control: a user-carrying request keeps the approved contract unchanged."""
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("answer"))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        ):
            await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=cast("UserInterface", self.interface_stub),
                conversation_messages=[Message(role="user", content="Hi")],
            )

        offered = self._offered_names(create_mock)
        assert {"say", "ack_silently"} <= offered

    @pytest.mark.asyncio
    async def test_hallucinated_ack_costs_a_repair_then_fails_loudly(self) -> None:
        """The offer-blind interception is now judged by the turn-end validator.

        Terminal interception is by tool-name membership and never consults the
        offered list: an interface-less request that was NOT offered
        ``ack_silently`` but emits it anyway still ends the turn empty — and
        the validator rejects that outcome identically to an empty text
        report: one bounded repair round-trip, then a loud ``SilentTurnError``
        (the loud-empty contract, now enforced in the loop, not only at the
        delegation boundary).
        """
        provider = MoonshotAI(token=TEST_TOKEN)
        ack_response = _completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="")
        create_mock = AsyncMock(side_effect=[ack_response, ack_response, AssertionError("repair cap exceeded")])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            with pytest.raises(SilentTurnError):
                await provider._get_chat_completion_response(
                    messages=[Message(role="user", content="Hi").to_openai()],
                    model=TEST_MODEL,
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    system_prompt="",
                    conversation_messages=[Message(role="user", content="Hi")],
                )

        # The tool was not offered...
        assert "ack_silently" not in self._offered_names(create_mock)
        # ...the hallucinated call still ended the turn, the validator spent
        # exactly ONE repair attempt, and the turn never passed silently.
        assert create_mock.await_count == 2


class TestOfferedToolsHelper:
    """Unit coverage for the single filter helper behind the D1 rule."""

    def test_interface_less_list_is_the_global_list_minus_the_terminal_tools(self) -> None:
        definitions = RegisteredChibiTools.get_tool_definitions()
        offered = offered_tool_definitions(definitions=definitions, interface=None)
        names = {definition["function"]["name"] for definition in offered}
        assert names == {definition["function"]["name"] for definition in definitions} - {"say", "ack_silently"}

    def test_interface_presence_keeps_every_tool(self) -> None:
        definitions = RegisteredChibiTools.get_tool_definitions()
        offered = offered_tool_definitions(definitions=definitions, interface=cast("UserInterface", SimpleNamespace()))
        assert len(offered) == len(definitions)

    def test_name_extractor_supports_non_openai_tool_shapes(self) -> None:
        anthropic_like = [{"name": "say"}, {"name": "get_current_datetime"}]
        assert [tool["name"] for tool in offered_tools(anthropic_like, None, name_of=lambda tool: tool["name"])] == [
            "get_current_datetime"
        ]

        class _GeminiTool:
            def __init__(self, name: str) -> None:
                self.function_declarations = [SimpleNamespace(name=name)]

        gemini_like = [_GeminiTool("say"), _GeminiTool("ack_silently"), _GeminiTool("get_current_datetime")]
        assert [
            tool.function_declarations[0].name
            for tool in offered_tools(gemini_like, None, name_of=lambda tool: tool.function_declarations[0].name)
        ] == ["get_current_datetime"]

    def test_the_signal_is_interface_presence_not_a_role(self) -> None:
        """Pins the D1 signal assumption: any non-None interface re-enables the tools.

        The ruling keys on ``interface is None`` because at the three loop
        entry points that condition holds exactly when the request has no user
        channel. If a future actor class breaks that equivalence, this test
        and the helper's docstring are the places that must be revisited.
        """
        definitions = RegisteredChibiTools.get_tool_definitions()
        bare_object_interface = object()
        offered = offered_tool_definitions(
            definitions=definitions, interface=cast("UserInterface", bare_object_interface)
        )
        assert len(offered) == len(definitions)


class TestLoopTrapRegression:
    """The centrepiece loop-trap regression (recon C7, plan recon 13.6).

    Conditions of the reproduced infinite-ack loop: forcing enabled plus both
    terminal tools offered, prompt "answer by calling say()". Observed live on
    OpenAI chat-completions (``tool_choice: required``) and Anthropic
    (``tool_choice: any``): ``say("2 + 2 = 4.")`` → tool result
    ``{"status": "delivered"}`` → ``ack_silently {}`` → ``ack_silently {}`` →
    … with no natural stop. Under the interception the first terminal call
    must exit the loop: no second request, a normalized message persisted
    with its ``visibility`` respected, and no tool result for the terminal
    call. The ack-forever variant on the CC seam is pinned (and extended with
    the persistence and no-tool-result assertions) by
    ``test_terminal_call_does_not_fuel_the_ack_loop`` above.

    Seam coverage: the two seams C7 was reproduced live on — the OpenAI
    chat-completions seam (MoonshotAI stand-in) and the Anthropic-compatible
    seam. The remaining seams (OpenAI Responses, Gemini, Mistral) share the
    same terminal-interception branch and already carry per-seam
    no-tool-result persistence tests; a third loop-trap variant would assert
    nothing the shared branch does not already pin.
    """

    @pytest.mark.asyncio
    async def test_cc_seam_say_then_infinite_acks_exits_on_the_terminal_call(self) -> None:
        """The C7 sequence shape: say() first, then an endless ack supply."""
        provider = MoonshotAI(token=TEST_TOKEN)
        say_response = _completion_with_tool_calls([_tool_call("say", '{"text": "2 + 2 = 4."}')])
        ack_response = _completion_with_tool_calls(
            [_tool_call("ack_silently", "{}")], content="semantically reasonable ack"
        )
        create_mock = AsyncMock(side_effect=[say_response] + [ack_response] * 9)
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        wake = _tool_response_wake()
        conversation: list[Message] = [Message(role="user", content=wake)]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        # (a) The loop exits on the terminal call: one request, never a second.
        assert response.answer == "2 + 2 = 4."
        assert create_mock.await_count == 1
        # (b) A normalized message is persisted, visibility respected.
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "2 + 2 = 4."
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        # (c) No tool result exists for the terminal call.
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_anthropic_seam_say_then_infinite_acks_exits_on_the_terminal_call(self) -> None:
        """The same C7 sequence on the Anthropic-compatible seam (repro'd live there)."""
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        say_response = _anthropic_message("thinking", [("say", {"text": "2 + 2 = 4."})])
        ack_response = _anthropic_message("semantically reasonable ack", [("ack_silently", {})])
        generate_mock = AsyncMock(side_effect=[say_response] + [ack_response] * 9)
        wake = _tool_response_wake()
        conversation: list[Message] = [Message(role="user", content=wake)]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch.object(Anthropic, "_generate_content", generate_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[{"role": "user", "content": wake}],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        # (a) The loop exits on the terminal call.
        assert response.answer == "2 + 2 = 4."
        assert generate_mock.await_count == 1
        # (b) A normalized message is persisted, visibility respected.
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "2 + 2 = 4."
        assert persisted[0].visibility == "visible"
        assert persisted[0].tool_calls is None
        # (c) No tool result exists for the terminal call.
        assert not [msg for msg in conversation if msg.role == "tool"]


def _background_dispatch(tool_name: str) -> Message:
    """The in-loop tool result a background dispatch produces (tool.py)."""
    return Message(
        role="tool",
        content=json.dumps(
            {
                "tool_name": tool_name,
                "status": "tool is running in background",
                "result": "in progress",
            }
        ),
    )


class TestStragglerAndPendingObligation:
    """Evidence-file fixture class (evidence_cot_wait_case.md, case B).

    Two parallel background obligations; the first result is a straggler that
    changes nothing, the second is still pending. The evidence file predates
    the tool-channel design: its "ACK" is now the ``ack_silently()`` terminal
    call, and per-obligation tracking is the validator's own
    ``background_work_pending`` scan over unanswered background dispatches.
    Assertions, translated: (1) the straggler is absorbed by a legitimate
    hidden ack turn, (2) the pending obligation is still tracked as
    unsatisfied and nothing is delivered for it, (3) when the second result
    arrives it is delivered via a visible ``say()`` message.
    """

    @pytest.mark.asyncio
    async def test_straggler_is_acked_while_the_pending_obligation_stays_open(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        straggler_wake = _tool_response_wake("generate_image")
        second_wake = _tool_response_wake("generate_video")
        conversation: list[Message] = [
            _background_dispatch("generate_image"),
            _background_dispatch("generate_video"),
            Message(role="user", content=straggler_wake),
        ]

        # Round 1: the straggler result arrived; the video is still running.
        ack_completion = _completion_with_tool_calls(
            [_tool_call("ack_silently", "{}")],
            content="one of two results arrived; it changes nothing, the other is still running",
        )
        ack_mock = AsyncMock(return_value=ack_completion)
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=ack_mock)))
        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=straggler_wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        # (1) The straggler is ACKed: the turn ends on the terminal call, the
        # private reasoning persists hidden, no tool result, nothing delivered.
        assert response.answer == ""
        assert ack_mock.await_count == 1
        tool_results_before = [msg for msg in conversation if msg.role == "tool"]
        assert len(tool_results_before) == 2  # only the seeded background dispatches
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].visibility == "hidden"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "visible"]
        # (2) The pending obligation is NOT considered satisfied: the
        # validator's own tracker still sees the unanswered video dispatch.
        assert background_work_pending(conversation) is True

        # Round 2: the second result arrives; it must be delivered.
        conversation.append(Message(role="user", content=second_wake))
        say_completion = _completion_with_tool_calls(
            [_tool_call("say", '{"text": "Both results are in: the comparison verdict."}')]
        )
        say_mock = AsyncMock(return_value=say_completion)
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=say_mock)))
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content=second_wake).to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        # (3) The second result IS delivered: visible message, no new tool
        # result, and no tool result for the terminal calls.
        assert response.answer == "Both results are in: the comparison verdict."
        assert say_mock.await_count == 1
        assert [msg for msg in conversation if msg.role == "tool"] == tool_results_before
        delivered = [msg for msg in conversation if msg.role == "assistant" and msg.visibility == "visible"]
        assert len(delivered) == 1
        assert delivered[0].content == "Both results are in: the comparison verdict."
        # Every background obligation has now been answered.
        assert background_work_pending(conversation) is False


class TestTurnMisuseAccountingPins:
    """Pins the current turn-misuse counter labels (PM ruling on the 4c-1 routed note).

    On an interface turn, a repair that recovers via plain text is logged as
    ``action="degraded_visible" kind="no_terminal_call_text"`` — not
    ``action="recovered"`` — because the no-tool-call exit checks
    ``verdict.misuse is not None`` before the ``repair_used`` branch
    (``provider.py`` no-tool-call exit; same shape on sibling seams). The
    ruling: pin the labels with instrumentation tests, do not reorder the
    checks. A recovery through a terminal call keeps the ``recovered`` label
    (the terminal-call branch checks ``repair_used`` directly).
    """

    @pytest.mark.asyncio
    async def test_plain_text_recovery_after_a_repair_logs_degraded_visible_not_recovered(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        empty_prose = _completion_with_answer("")
        recovered_prose = _completion_with_answer("Recovered via plain text")
        create_mock = AsyncMock(side_effect=[empty_prose, recovered_prose])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        log_mock = Mock()

        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch("chibi.services.providers.provider.log_turn_misuse", log_mock),
            patch("chibi.services.providers.terminal_calls.log_turn_misuse", log_mock),
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=[Message(role="user", content="Hi")],
            )

        # The repair happened (exactly one round-trip) and recovered via text.
        assert response.answer == "Recovered via plain text"
        assert create_mock.await_count == 2
        logged = [call.kwargs for call in log_mock.call_args_list]
        assert logged[0].get("action") == "repair"
        # The recovered ending is accounted as degraded_visible, never recovered.
        assert any(
            entry.get("action") == "degraded_visible" and entry.get("kind") == "no_terminal_call_text"
            for entry in logged
        )
        assert not any(entry.get("action") == "recovered" for entry in logged)

    @pytest.mark.asyncio
    async def test_terminal_call_recovery_after_a_repair_logs_recovered(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        empty_prose = _completion_with_answer("")
        say_completion = _completion_with_tool_calls([_tool_call("say", '{"text": "Final answer"}')])
        create_mock = AsyncMock(side_effect=[empty_prose, say_completion])
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        log_mock = Mock()
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            patch("chibi.services.providers.provider.log_turn_misuse", log_mock),
            patch("chibi.services.providers.terminal_calls.log_turn_misuse", log_mock),
            registry_patch,
        ):
            response, _wire = await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=1,
                system_prompt="",
                interface=cast("UserInterface", INTERFACE_STUB),
                conversation_messages=conversation,
            )

        assert response.answer == "Final answer"
        assert create_mock.await_count == 2
        logged = [call.kwargs for call in log_mock.call_args_list]
        assert any(entry.get("action") == "recovered" and entry.get("kind") is None for entry in logged)
        # The terminal call received no tool result.
        assert not [msg for msg in conversation if msg.role == "tool"]
