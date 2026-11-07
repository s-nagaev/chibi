"""Parametric tests that a per-call ``max_tokens`` reaches each provider completion payload."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from anthropic.types import TextBlock, ToolUseBlock
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    Part,
)
from mistralai.client.models import FunctionCall as MistralFunctionCall
from mistralai.client.models import ToolCall, UserMessage
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import ResponseFunctionToolCall

from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.context_measurement import SizeProvenance
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI as ChibiOpenAI
from chibi.services.providers.tools.schemas import ToolResponseSchema

TEST_TOKEN = "test-token-for-max-tokens"
TEST_MODEL = "test-model"
TEST_OPENAI_MODEL = "gpt-5.6-terra"


def _make_chat_completion(answer: str) -> ChatCompletion:
    """Build a ChatCompletion containing a final answer without tool calls.

    Args:
        answer: Assistant message content.

    Returns:
        A ChatCompletion object with a single choice and no tool calls.
    """
    mock_choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=answer, tool_calls=None),
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _make_anthropic_message() -> SimpleNamespace:
    """Build a minimal non-empty Anthropic response message.

    Returns:
        An object satisfying the ``response_message.content`` checks in the provider.
    """
    text_block = SimpleNamespace(text="Hello")
    return SimpleNamespace(content=[text_block])


async def _run_openai_friendly_chat_completions(max_tokens: int | None) -> dict[str, Any]:
    """Run the OpenAI-friendly chat-completions path and capture the request payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        The completion-call kwargs captured from the mocked SDK client.
    """
    provider = MoonshotAI(token=TEST_TOKEN)
    create_mock = AsyncMock(return_value=_make_chat_completion("Hello"))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    with patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client):
        await provider._get_chat_completion_response(
            messages=[Message(role="user", content="Hi").to_openai()],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt=None,
            max_tokens=max_tokens,
        )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


async def _run_openai_responses(max_tokens: int | None) -> dict[str, Any]:
    """Run the OpenAI Responses API path and capture the request payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        The completion-call kwargs captured from the mocked SDK client.
    """
    provider = ChibiOpenAI(token=TEST_TOKEN)
    response = MagicMock()
    response.output = [SimpleNamespace(type="message")]
    response.output_text = "Hello"
    response.usage = None
    create_mock = AsyncMock(return_value=response)
    mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    provider.client = mock_client

    await provider._get_response_completion_response(
        messages=[Message(role="user", content="Hi")],
        model=TEST_OPENAI_MODEL,
        user=User(id=1),
        caller_storage_id=1,
        caller_thread_id=0,
        system_prompt=None,
        max_tokens=max_tokens,
    )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


async def _run_anthropic_non_stream(max_tokens: int | None) -> dict[str, Any]:
    """Run the Anthropic non-streaming path and capture the request payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        The completion-call kwargs captured from the mocked SDK client.
    """
    provider = Anthropic(token=TEST_TOKEN)
    create_mock = AsyncMock(return_value=_make_anthropic_message())
    mock_client: Any = SimpleNamespace(messages=SimpleNamespace(create=create_mock, stream=Mock()))
    provider._client = mock_client

    await provider._generate_content(
        model=TEST_MODEL,
        system_prompt="sp",
        messages=[{"role": "user", "content": "Hi"}],
        interface=None,
        max_tokens=max_tokens,
    )
    assert create_mock.await_args is not None
    return dict(create_mock.await_args.kwargs)


class _FakeAnthropicStream:
    """Async context manager emulating an Anthropic event stream with no events."""

    def __init__(self, final_message: SimpleNamespace) -> None:
        """Store the final message to return.

        Args:
            final_message: The aggregated message returned by ``get_final_message``.
        """
        self._final_message = final_message

    async def __aenter__(self) -> "_FakeAnthropicStream":
        """Enter the streaming context.

        Returns:
            The stream itself.
        """
        return self

    async def __aexit__(self, *_exc_info: Any) -> bool:
        """Close the streaming context.

        Args:
            *_exc_info: Exception type, value and traceback, if any.

        Returns:
            Always False, so exceptions propagate.
        """
        return False

    def __aiter__(self) -> "_FakeAnthropicStream":
        """Return an async iterator over zero events.

        Returns:
            The stream itself.
        """
        return self

    async def __anext__(self) -> Any:
        """Signal the end of the event stream.

        Raises:
            StopAsyncIteration: Always, the fake stream carries no events.
        """
        raise StopAsyncIteration

    async def get_final_message(self) -> SimpleNamespace:
        """Return the aggregated final message.

        Returns:
            The stored final message.
        """
        return self._final_message


async def _run_anthropic_stream(max_tokens: int | None) -> dict[str, Any]:
    """Run the Anthropic streaming path and capture the request payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        The stream-call kwargs captured from the mocked SDK client.
    """
    provider = Anthropic(token=TEST_TOKEN)
    stream_mock = Mock(return_value=_FakeAnthropicStream(_make_anthropic_message()))
    mock_client: Any = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(), stream=stream_mock))
    provider._client = mock_client

    await provider._stream_generate_content(
        model=TEST_MODEL,
        system=[{"type": "text", "text": "sp"}],
        messages=[{"role": "user", "content": "Hi"}],
        interface=None,
        max_tokens=max_tokens,
    )
    return dict(stream_mock.call_args.kwargs)


async def _run_gemini(max_tokens: int | None) -> dict[str, Any]:
    """Run the Gemini native path and capture the generation config payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        A dict with the ``max_output_tokens`` value extracted from the captured config.
    """
    provider = Gemini(token=TEST_TOKEN)
    response = GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text="Hello")]))]
    )
    generate_mock = AsyncMock(return_value=response)
    with (
        patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch.object(Gemini, "_generate_content", generate_mock),
    ):
        await provider._get_chat_completion_response(
            messages=[],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
            system_prompt="sp",
            max_tokens=max_tokens,
        )
    assert generate_mock.await_args is not None
    config = generate_mock.await_args.kwargs["config"]
    return {"max_output_tokens": config.max_output_tokens}


async def _run_mistral_non_stream(max_tokens: int | None) -> dict[str, Any]:
    """Run the Mistral non-streaming path and capture the request payload.

    Args:
        max_tokens: The per-call max_tokens value to forward.

    Returns:
        The completion-call kwargs captured from the mocked SDK client.
    """
    provider = MistralAI(token=TEST_TOKEN)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello", tool_calls=None))])
    complete_mock = AsyncMock(return_value=response)
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(complete_async=complete_mock, stream_async=Mock()))
    provider._client = mock_client
    user_message = UserMessage(role="user", content="Hi")

    await provider._generate_content(
        model=TEST_MODEL,
        messages=[user_message],
        interface=None,
        max_tokens=max_tokens,
    )
    assert complete_mock.await_args is not None
    return dict(complete_mock.await_args.kwargs)


MAX_TOKENS_CASES = {
    "openai_friendly_chat_completions": (
        _run_openai_friendly_chat_completions,
        "max_tokens",
        lambda: MoonshotAI(token=TEST_TOKEN)._get_max_tokens_value(model_name=TEST_MODEL),
    ),
    "openai_responses": (
        _run_openai_responses,
        "max_output_tokens",
        lambda: ChibiOpenAI(token=TEST_TOKEN)._get_max_tokens_value(model_name=TEST_OPENAI_MODEL),
    ),
    "anthropic_non_stream": (
        _run_anthropic_non_stream,
        "max_tokens",
        lambda: Anthropic(token=TEST_TOKEN).max_tokens,
    ),
    "anthropic_stream": (
        _run_anthropic_stream,
        "max_tokens",
        lambda: Anthropic(token=TEST_TOKEN).max_tokens,
    ),
    "gemini": (
        _run_gemini,
        "max_output_tokens",
        lambda: Gemini(token=TEST_TOKEN).max_tokens,
    ),
    "mistral_non_stream": (
        _run_mistral_non_stream,
        "max_tokens",
        lambda: MistralAI(token=TEST_TOKEN).max_tokens,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(MAX_TOKENS_CASES))
async def test_provided_max_tokens_overrides_instance_value(case: str) -> None:
    """An explicitly provided max_tokens must reach the completion payload verbatim."""
    runner, payload_key, _fallback_factory = MAX_TOKENS_CASES[case]

    payload = await runner(4242)

    assert payload[payload_key] == 4242


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(MAX_TOKENS_CASES))
async def test_none_max_tokens_keeps_old_value(case: str) -> None:
    """max_tokens=None must leave the payload on the pre-existing instance/config value."""
    runner, payload_key, fallback_factory = MAX_TOKENS_CASES[case]

    payload = await runner(None)

    assert payload[payload_key] == fallback_factory()


@pytest.mark.asyncio
async def test_gemini_tool_call_continuation_forwards_max_tokens() -> None:
    """Regression: the recursive continuation after Gemini tool calls must forward max_tokens."""
    provider = Gemini(token=TEST_TOKEN)
    tool_call_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model", parts=[Part(function_call=FunctionCall(name="test_tool", args={}, id="call_1"))]
                )
            )
        ]
    )
    final_response = GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text="Done")]))]
    )
    generate_mock = AsyncMock(side_effect=[tool_call_response, final_response])

    with (
        patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch.object(Gemini, "_generate_content", generate_mock),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=[Message(role="user", content="Use the tool")],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
            system_prompt="sp",
            max_tokens=41600,
        )

    assert chat_response == ChatResponseSchema(
        answer="Done", provider="Gemini", model=TEST_MODEL, usage=chat_response.usage
    )
    assert generate_mock.await_count == 2
    first_config = generate_mock.await_args_list[0].kwargs["config"]
    second_config = generate_mock.await_args_list[1].kwargs["config"]
    assert first_config.max_output_tokens == 41600
    assert second_config.max_output_tokens == 41600


# ---- In-loop budget check wiring (one test per provider loop path) ----
# Each test drives a one-exchange tool loop and asserts that the in-loop
# budget check runs before the recursive in-loop request and that the mask it
# returns is applied to the provider-format request list (a mask dropping the
# first conversation message must remove it from the second request payload,
# while the fresh tool pair stays intact). The check itself is mocked at the
# provider module's import site; its decision logic is covered in
# tests/services/test_user.py and test_provider_context_size.py.

_IN_LOOP_DROP_FIRST_MASK = [False, True, True]


def _tool_response_schema() -> ToolResponseSchema:
    """Return the canned tool execution result used by the loop tests."""
    return ToolResponseSchema(tool_name="test_tool", status="ok", result={})


@pytest.mark.asyncio
async def test_in_loop_check_applied_on_openai_friendly_path() -> None:
    """The Chat Completions loop applies the in-loop mask to its request list."""
    provider = MoonshotAI(token=TEST_TOKEN)
    tool_call = ChatCompletionMessageToolCall(
        id="call_1", type="function", function=Function(name="test_tool", arguments="{}")
    )
    tool_response = ChatCompletion(
        id="test-id",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Calling tool", tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
        created=1,
        model=TEST_MODEL,
        object="chat.completion",
    )
    final_response = _make_chat_completion("Done")
    create_mock = AsyncMock(side_effect=[tool_response, final_response])
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))

    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch(
            "chibi.services.providers.provider.enforce_in_loop_budget",
            new=AsyncMock(return_value=_IN_LOOP_DROP_FIRST_MASK),
        ),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[Message(role="user", content="Hi").to_openai()],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt=None,
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert create_mock.await_count == 2
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    # The recorded list keeps growing after the call (the final assistant turn
    # is appended to the same object), so assert on the request-time prefix.
    assert [item["role"] for item in second_messages[:2]] == ["assistant", "tool"]
    assert "Hi" not in json.dumps(second_messages)
    assert second_messages[0]["tool_calls"][0]["id"] == second_messages[1]["tool_call_id"]


@pytest.mark.asyncio
async def test_in_loop_check_applied_on_openai_responses_path() -> None:
    """The Responses API loop applies the in-loop mask to its Message list."""
    provider = ChibiOpenAI(token=TEST_TOKEN)
    tool_item = ResponseFunctionToolCall(
        id="fc_1", call_id="call_1", name="test_tool", arguments="{}", type="function_call"
    )
    tool_response = MagicMock()
    tool_response.output = [tool_item]
    tool_response.output_text = "Calling tool"
    tool_response.usage = None
    final_response = MagicMock()
    final_response.output = [SimpleNamespace(type="message")]
    final_response.output_text = "Done"
    final_response.usage = None
    create_mock = AsyncMock(side_effect=[tool_response, final_response])
    mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    provider.client = mock_client

    with (
        patch(
            "chibi.services.providers.openai.enforce_in_loop_budget",
            new=AsyncMock(return_value=_IN_LOOP_DROP_FIRST_MASK),
        ),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_response_completion_response(
            messages=[Message(role="user", content="Hi")],
            model=TEST_OPENAI_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt=None,
        )

    assert create_mock.await_count == 2
    second_input = create_mock.await_args_list[1].kwargs["input"]
    assert "Hi" not in json.dumps(second_input, default=str)
    assert "call_1" in json.dumps(second_input, default=str)


@pytest.mark.asyncio
async def test_in_loop_check_applied_on_anthropic_path() -> None:
    """The Anthropic loop applies the in-loop mask to its request list."""
    provider = Anthropic(token=TEST_TOKEN)
    tool_block = ToolUseBlock(id="call_1", name="test_tool", input={}, type="tool_use")
    text_block = TextBlock(text="Done", type="text")
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    generate_mock = AsyncMock(
        side_effect=[
            SimpleNamespace(content=[tool_block], usage=usage),
            SimpleNamespace(content=[text_block], usage=usage),
        ]
    )

    with (
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch(
            "chibi.services.providers.provider.enforce_in_loop_budget",
            new=AsyncMock(return_value=_IN_LOOP_DROP_FIRST_MASK),
        ),
        patch.object(Anthropic, "_generate_content", generate_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert generate_mock.await_count == 2
    second_messages = generate_mock.await_args_list[1].kwargs["messages"]
    # The recorded list keeps growing after the call; assert on the prefix.
    assert second_messages[0]["role"] == "assistant"
    assert "Hi" not in json.dumps(second_messages)
    assert "call_1" in json.dumps(second_messages)


@pytest.mark.asyncio
async def test_in_loop_check_applied_on_gemini_path() -> None:
    """The Gemini loop applies the in-loop mask to its contents list."""
    provider = Gemini(token=TEST_TOKEN)
    tool_call_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model", parts=[Part(function_call=FunctionCall(name="test_tool", args={}, id="call_1"))]
                )
            )
        ]
    )
    final_response = GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text="Done")]))]
    )
    generate_mock = AsyncMock(side_effect=[tool_call_response, final_response])

    with (
        patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch(
            "chibi.services.providers.gemini_native.enforce_in_loop_budget",
            new=AsyncMock(return_value=_IN_LOOP_DROP_FIRST_MASK),
        ),
        patch.object(Gemini, "_generate_content", generate_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[Message(role="user", content="Hi").to_google()],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert generate_mock.await_count == 2
    second_contents = generate_mock.await_args_list[1].kwargs["contents"]
    # The recorded list keeps growing after the call; assert on the prefix.
    assert second_contents[0]["role"] == "model"
    assert "Hi" not in json.dumps(second_contents, default=str)
    assert "call_1" in json.dumps(second_contents, default=str)


@pytest.mark.asyncio
async def test_in_loop_check_applied_on_mistral_path() -> None:
    """The Mistral loop applies the in-loop mask past its system-message offset."""
    provider = MistralAI(token=TEST_TOKEN)
    tool_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Calling tool",
                    tool_calls=[ToolCall(id="call_1", function=MistralFunctionCall(name="test_tool", arguments="{}"))],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    final_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    complete_mock = AsyncMock(side_effect=[tool_response, final_response])
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(complete_async=complete_mock, stream_async=Mock()))
    provider._client = mock_client

    with (
        patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch(
            "chibi.services.providers.mistralai_native.enforce_in_loop_budget",
            new=AsyncMock(return_value=_IN_LOOP_DROP_FIRST_MASK),
        ),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[UserMessage(role="user", content="Hi")],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert complete_mock.await_count == 2
    second_messages = complete_mock.await_args_list[1].kwargs["messages"]
    # The recorded list keeps growing after the call; assert on the prefix.
    assert "Hi" not in json.dumps(second_messages, default=str)
    assert "call_1" in json.dumps(second_messages, default=str)
    assert any(getattr(item, "tool_call_id", None) == "call_1" for item in second_messages)


def _assert_anthropic_pair_integrity(items: list[dict[str, Any]]) -> None:
    """Assert Anthropic pair integrity: no orphan tool_result, no result-less tool_use.

    Every ``tool_result`` block must follow the assistant message carrying
    the ``tool_use`` block with the same id, and every ``tool_use`` block
    must have its result somewhere after it.
    """
    pending: set[str] = set()
    for item in items:
        for block in item.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                pending.add(block["id"])
            elif block.get("type") == "tool_result":
                assert block.get("tool_use_id") in pending, (
                    f"orphan tool_result {block.get('tool_use_id')!r}: its assistant tool_use "
                    "carrier is not part of the request"
                )
                pending.discard(block["tool_use_id"])
    assert not pending, f"tool_use blocks without results: {pending}"


def _in_loop_two_drop_masks() -> list[list[bool]]:
    """Masks for a mocked two-drop turn: drop the head, then drop the old pair.

    Check 1 runs over ``[head, carrier_1, result_1]`` and drops the head;
    check 2 runs over ``[carrier_1, result_1, carrier_2, result_2]`` (the
    mirror already filtered by mask 1, plus the second exchange) and drops
    the FIRST pair whole, keeping the fresh one.
    """
    return [
        [False, True, True],
        [False, False, True, True],
    ]


@pytest.mark.asyncio
async def test_in_loop_second_drop_stays_aligned_on_anthropic_path() -> None:
    """A second real in-loop drop on the Anthropic path masks the list actually sent.

    Regression for the positional-mask misalignment: the keep-mask used to be
    computed over the never-filtered mirror, so from the second drop onward
    the request could lose its fresh tool pair or carry an orphan
    ``tool_result``. The loop now filters the mirror with the same mask and
    passes the filtered mirror into the recursion.
    """
    provider = Anthropic(token=TEST_TOKEN)
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    responses = [
        SimpleNamespace(content=[ToolUseBlock(id="call_1", name="test_tool", input={}, type="tool_use")], usage=usage),
        SimpleNamespace(content=[ToolUseBlock(id="call_2", name="test_tool", input={}, type="tool_use")], usage=usage),
        SimpleNamespace(content=[TextBlock(text="Done", type="text")], usage=usage),
    ]
    generate_mock = AsyncMock(side_effect=responses)
    conversation = [
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second question"),
    ]

    with (
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.get_model_context_window", new=Mock(return_value=1000)),
        patch.object(Anthropic, "_get_max_tokens_value", new=Mock(return_value=100)),
        patch.object(Anthropic, "_generate_content", generate_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
            system_prompt="base",
        )

    assert chat_response.answer == "Done"
    assert generate_mock.await_count == 3

    # First in-loop request (after drop 1): head + the fresh first pair.
    second_request = generate_mock.await_args_list[1].kwargs["messages"][:3]
    assert [item["role"] for item in second_request] == ["user", "assistant", "user"]
    _assert_anthropic_pair_integrity(second_request)
    assert "First answer" not in json.dumps(second_request)
    assert "call_1" in json.dumps(second_request)

    # Second in-loop request (after drop 2): the first pair is gone whole,
    # the fresh pair of the last exchange is intact, the head survives.
    third_request = generate_mock.await_args_list[2].kwargs["messages"][:3]
    assert [item["role"] for item in third_request] == ["user", "assistant", "user"]
    _assert_anthropic_pair_integrity(third_request)
    assert third_request[0]["content"][0]["text"] == "First question"
    assert any(
        isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id") == "call_2"
        for block in third_request[1]["content"]
    )
    assert any(
        isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id") == "call_2"
        for block in third_request[2]["content"]
    )
    assert "call_1" not in json.dumps(third_request)
    assert "First answer" not in json.dumps(third_request)
    assert "Second question" not in json.dumps(third_request)


@pytest.mark.asyncio
async def test_in_loop_mask_stays_in_lockstep_on_anthropic_path() -> None:
    """The Anthropic loop filters the mirror with the same mask as its request list.

    The second check must receive the already-filtered mirror (4 items), not
    the stale full one (5 items), and the second drop must remove the first
    pair whole from the request.
    """
    provider = Anthropic(token=TEST_TOKEN)
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    responses = [
        SimpleNamespace(content=[ToolUseBlock(id="call_1", name="test_tool", input={}, type="tool_use")], usage=usage),
        SimpleNamespace(content=[ToolUseBlock(id="call_2", name="test_tool", input={}, type="tool_use")], usage=usage),
        SimpleNamespace(content=[TextBlock(text="Done", type="text")], usage=usage),
    ]
    generate_mock = AsyncMock(side_effect=responses)
    check_mock = AsyncMock(side_effect=_in_loop_two_drop_masks())

    with (
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.provider.enforce_in_loop_budget", new=check_mock),
        patch.object(Anthropic, "_generate_content", generate_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert check_mock.await_count == 2
    # The mirror handed to the second check is the filtered one (4 items):
    # head dropped by mask 1, then the second exchange appended. Without the
    # lockstep filtering it would still carry the dropped head (5 items).
    assert len(check_mock.await_args_list[1].kwargs["conversation_messages"]) == 4

    assert generate_mock.await_count == 3
    third_request = generate_mock.await_args_list[2].kwargs["messages"][:2]
    assert [item["role"] for item in third_request] == ["assistant", "user"]
    _assert_anthropic_pair_integrity(third_request)
    assert "call_2" in json.dumps(third_request)
    assert "call_1" not in json.dumps(third_request)
    assert "Hi" not in json.dumps(third_request)


@pytest.mark.asyncio
async def test_in_loop_mask_stays_in_lockstep_on_gemini_path() -> None:
    """The Gemini loop filters the mirror with the same mask as its contents list."""
    provider = Gemini(token=TEST_TOKEN)
    tool_call_response_1 = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model", parts=[Part(function_call=FunctionCall(name="test_tool", args={}, id="call_1"))]
                )
            )
        ]
    )
    tool_call_response_2 = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model", parts=[Part(function_call=FunctionCall(name="test_tool", args={}, id="call_2"))]
                )
            )
        ]
    )
    final_response = GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text="Done")]))]
    )
    generate_mock = AsyncMock(side_effect=[tool_call_response_1, tool_call_response_2, final_response])
    check_mock = AsyncMock(side_effect=_in_loop_two_drop_masks())

    with (
        patch("chibi.services.providers.gemini_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.gemini_native.enforce_in_loop_budget", new=check_mock),
        patch.object(Gemini, "_generate_content", generate_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[Message(role="user", content="Hi").to_google()],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert check_mock.await_count == 2
    assert len(check_mock.await_args_list[1].kwargs["conversation_messages"]) == 4

    assert generate_mock.await_count == 3
    third_contents = generate_mock.await_args_list[2].kwargs["contents"][:2]
    assert [item["role"] for item in third_contents] == ["model", "user"]
    assert "call_2" in json.dumps(third_contents, default=str)
    assert "call_1" not in json.dumps(third_contents, default=str)
    assert "Hi" not in json.dumps(third_contents, default=str)
    # The fresh pair is intact: the function_call id matches the function_response id.
    assert any(
        part.get("function_call", {}).get("id") == "call_2" for item in third_contents for part in item.get("parts", [])
    )
    assert any(
        part.get("function_response", {}).get("id") == "call_2"
        for item in third_contents
        for part in item.get("parts", [])
    )


@pytest.mark.asyncio
async def test_in_loop_mask_stays_in_lockstep_on_mistral_path() -> None:
    """The Mistral loop filters the mirror with the same mask, past its system offset."""
    provider = MistralAI(token=TEST_TOKEN)

    def _mistral_tool_response(call_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Calling tool",
                        tool_calls=[
                            ToolCall(id=call_id, function=MistralFunctionCall(name="test_tool", arguments="{}"))
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    final_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    complete_mock = AsyncMock(
        side_effect=[_mistral_tool_response("call_1"), _mistral_tool_response("call_2"), final_response]
    )
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(complete_async=complete_mock, stream_async=Mock()))
    provider._client = mock_client
    check_mock = AsyncMock(side_effect=_in_loop_two_drop_masks())

    with (
        patch("chibi.services.providers.mistralai_native.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.mistralai_native.enforce_in_loop_budget", new=check_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_chat_completion_response(
            messages=[UserMessage(role="user", content="Hi")],
            model=TEST_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt="sp",
            conversation_messages=[Message(role="user", content="Hi")],
        )

    assert check_mock.await_count == 2
    assert len(check_mock.await_args_list[1].kwargs["conversation_messages"]) == 4

    assert complete_mock.await_count == 3
    third_messages = complete_mock.await_args_list[2].kwargs["messages"][:3]
    assert len(third_messages) == 3
    assert "call_2" in json.dumps(third_messages, default=str)
    assert "call_1" not in json.dumps(third_messages, default=str)
    assert "Hi" not in json.dumps(third_messages, default=str)
    assert any(getattr(item, "role", None) == "system" for item in third_messages)
    assert any(getattr(item, "tool_call_id", None) == "call_2" for item in third_messages)


@pytest.mark.asyncio
async def test_in_loop_mask_stays_in_lockstep_on_openai_responses_path() -> None:
    """The Responses loop keeps its self-aligned list filtered on every drop."""
    provider = ChibiOpenAI(token=TEST_TOKEN)

    def _responses_tool_response(call_id: str) -> MagicMock:
        tool_item = ResponseFunctionToolCall(
            id=f"fc_{call_id}", call_id=call_id, name="test_tool", arguments="{}", type="function_call"
        )
        response = MagicMock()
        response.output = [tool_item]
        response.output_text = "Calling tool"
        response.usage = None
        return response

    final_response = MagicMock()
    final_response.output = [SimpleNamespace(type="message")]
    final_response.output_text = "Done"
    final_response.usage = None
    create_mock = AsyncMock(
        side_effect=[_responses_tool_response("call_1"), _responses_tool_response("call_2"), final_response]
    )
    mock_client: Any = SimpleNamespace(responses=SimpleNamespace(create=create_mock))
    provider.client = mock_client
    check_mock = AsyncMock(side_effect=_in_loop_two_drop_masks())

    with (
        patch("chibi.services.providers.openai.enforce_in_loop_budget", new=check_mock),
        patch.object(provider, "call_functions", new=AsyncMock(return_value=[_tool_response_schema()])),
    ):
        await provider._get_response_completion_response(
            messages=[Message(role="user", content="Hi")],
            model=TEST_OPENAI_MODEL,
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            system_prompt=None,
        )

    assert check_mock.await_count == 2
    assert len(check_mock.await_args_list[1].kwargs["conversation_messages"]) == 4

    assert create_mock.await_count == 3
    third_input = create_mock.await_args_list[2].kwargs["input"]
    assert "call_2" in json.dumps(third_input, default=str)
    assert "call_1" not in json.dumps(third_input, default=str)
    assert "Hi" not in json.dumps(third_input, default=str)
