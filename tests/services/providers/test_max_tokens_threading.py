"""Parametric tests that a per-call ``max_tokens`` reaches each provider completion payload."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    Part,
)
from mistralai.models import UserMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
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
