"""Tests for the say()/ack_silently() terminal-call interception (sub-task 4b).

The invariant under test: a terminal call ends the turn in every adapter
path, is normalized at ingest into a plain assistant ``Message``, and NEVER
receives a tool result — a result is the fuel of the infinite
``say() → result → ack_silently() → ack_silently() → ...`` loop (recon C7).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from google.genai.types import FunctionCall as GeminiFunctionCall
from mistralai.models import FunctionCall as MistralFunctionCall
from mistralai.models import ToolCall as MistralToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import ResponseFunctionToolCall

import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.models import Message, User
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI as ChibiOpenAI
from chibi.services.providers.terminal_calls import (
    TERMINAL_TOOL_NAMES,
    fold_terminal_calls_into_thoughts,
    partition_terminal_calls,
    plan_terminal_turn,
)
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.schemas import ToolCallSchema, ToolResponseSchema

TEST_TOKEN = "test-token-for-terminal-calls"
TEST_MODEL = "kimi-k2.6"
TEST_OPENAI_MODEL = "gpt-5.6-terra"


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
                system_prompt=None,
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
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="waiting for results")
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert create_mock.await_count == 1
        persisted = [msg for msg in conversation if msg.role == "assistant"]
        assert len(persisted) == 1
        assert persisted[0].content == "waiting for results"
        assert persisted[0].visibility == "hidden"
        assert persisted[0].tool_calls is None
        assert not [msg for msg in conversation if msg.role == "tool"]

    @pytest.mark.asyncio
    async def test_degenerate_ack_call_omits_the_assistant_turn(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="")
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant"] == []

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
        # first terminal call must end the turn regardless of what the mock
        # would keep returning.
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(
            return_value=_completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="thoughts")
        )
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
        conversation: list[Message] = [Message(role="user", content="Hi")]

        registry_patch, _ = _forbid_terminal_dispatch()
        with (
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
            registry_patch,
        ):
            await provider._get_chat_completion_response(
                messages=[Message(role="user", content="Hi").to_openai()],
                model=TEST_MODEL,
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert create_mock.await_count == 1

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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == "Done"
        assert call_functions_mock.await_count == 1
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
                system_prompt=None,
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
        provider = ChibiOpenAI(token=TEST_TOKEN)
        tool_call = ResponseFunctionToolCall(
            type="function_call", name="ack_silently", arguments="{}", call_id="call_1", status="completed"
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
                system_prompt=None,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant"] == []

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
                system_prompt=None,
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
        from chibi.services.providers.anthropic import Anthropic

        provider = Anthropic(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_anthropic_message("absorbing the tool result", [("ack_silently", {})]))
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        persisted = [msg for msg in conversation if msg.role == "assistant"]
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant"] == []


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
                system_prompt=None,
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
        provider = Gemini(token=TEST_TOKEN)
        generate_mock = AsyncMock(return_value=_gemini_response("waiting for more results", [("ack_silently", {})]))
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        persisted = [msg for msg in conversation if msg.role == "assistant"]
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
                system_prompt=None,
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
        provider = MistralAI(token=TEST_TOKEN)
        message = SimpleNamespace(
            content="",
            tool_calls=[MistralToolCall(id="t1", function=MistralFunctionCall(name="ack_silently", arguments="{}"))],
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
                system_prompt=None,
                conversation_messages=conversation,
            )

        assert response.answer == ""
        assert [msg for msg in conversation if msg.role == "assistant"] == []


class TestVisibilityMetadataOnTheWire:
    """The visibility field is platform metadata and never reaches an API payload."""

    def test_to_openai_strips_the_visibility_field(self) -> None:
        wire = Message(role="assistant", content="hi", visibility="hidden").to_openai()
        assert "visibility" not in dict(wire)

    def test_legacy_message_without_the_field_reads_back_visible(self) -> None:
        legacy = {"role": "assistant", "content": "old message", "id": 1}
        message = Message(**legacy)
        assert message.visibility == "visible"
