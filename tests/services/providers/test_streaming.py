"""Unit tests for live response-delta streaming in the provider layer.

Reachability tests go through the real public chain
(``_get_chat_completion_response``/``_get_response_completion_response``) so
that wiring bugs (a call site forgetting to pass ``interface``) and
unreachable fallbacks (the ``__getattribute__`` error-conversion wrapper
swallowing raw SDK errors) actually fail the suite. Pure-logic unit tests
(accumulator, latch) complement them.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, PropertyMock, patch

import httpx
import httpx2
import pytest
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from anthropic.types import (
    Usage as AnthropicUsage,
)
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    TextChunk,
    ThinkChunk,
    UsageInfo,
)
from openai import BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import ResponseTextDeltaEvent
from tenacity import wait_fixed

from chibi.config import gpt_settings
from chibi.exceptions import ServiceRateLimitError, ServiceResponseError
from chibi.models import Message, User
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.minimax import Minimax
from chibi.services.providers.mistralai_native import MistralAI, _is_retryable_mistral_error
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import (
    AnthropicFriendlyProvider,
    OpenAIFriendlyProvider,
    _OpenAIChatStreamAccumulator,
)
from chibi.services.providers.streaming import (
    AckSentinelStreamSession,
    delta_latched,
    delta_streaming_allowed,
    mark_non_streaming_fallback_spent,
    non_streaming_fallback_spent_error,
)

TEST_TOKEN = "test-streaming-token"


def _make_interface(**overrides: Any) -> Any:
    """Build a streaming-capable interface stub."""
    defaults: dict[str, Any] = {
        "streaming_enabled": True,
        "delta_emitted": False,
        "thread_id": 0,
        "send_delta": AsyncMock(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_chunk(
    text: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: Any = None,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> Any:
    """Build a minimal OpenAI ChatCompletionChunk-like object."""
    delta = SimpleNamespace(content=text, tool_calls=tool_calls, reasoning_content=reasoning_content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id="chatcmpl-1", model="m-1", created=42, choices=[choice], usage=usage)


def _make_bad_request_error() -> BadRequestError:
    """Build a real openai.BadRequestError (400, non-context-length)."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": "stream_options unsupported"}})
    return BadRequestError("stream_options unsupported", response=response, body=None)


def _make_rate_limit_error() -> RateLimitError:
    """Build a real openai.RateLimitError (429)."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "rate limited"}})
    return RateLimitError("rate limited", response=response, body=None)


def _make_mistral_sdk_error(status_code: int) -> SDKError:
    """Build a real mistralai SDKError carrying the given HTTP status."""
    request = httpx2.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx2.Response(status_code, request=request, json={"message": "stream_options unsupported"})
    return SDKError("API error occurred", response, None)


def _make_mistral_completion(text: str = "Hello") -> ChatCompletionResponse:
    """Build a real non-streaming Mistral response for fallback assertions."""
    return ChatCompletionResponse(
        id="cmpl-2",
        object="chat.completion",
        model="mistral",
        created=7,
        usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        choices=[ChatCompletionChoice(index=0, finish_reason="stop", message=AssistantMessage(content=text))],
    )


def _make_non_streaming_completion(text: str = "Hello") -> ChatCompletion:
    """Build a real non-streaming ChatCompletion for fallback assertions."""
    return ChatCompletion(
        id="chatcmpl-2",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content=text))],
        created=42,
        model="m-1",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _text_delta_event(delta: str) -> ResponseTextDeltaEvent:
    """Build a real Responses API output-text delta event."""
    return ResponseTextDeltaEvent(
        sequence_number=0,
        type="response.output_text.delta",
        item_id="item-1",
        output_index=0,
        content_index=0,
        delta=delta,
        logprobs=[],
    )


def _fake_responses_api_response(text: str) -> Any:
    """Build a minimal non-streaming Responses API response stand-in."""
    return SimpleNamespace(output=[SimpleNamespace(type="message")], output_text=text, usage=None)


class _FakeOpenAIStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "_FakeOpenAIStream":
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        item = self._chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


class _FakeResponsesStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = list(events)
        self._final = final
        self.closed = False

    async def __aenter__(self) -> "_FakeResponsesStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    def __aiter__(self) -> "_FakeResponsesStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_response(self) -> Any:
        return self._final


class _RejectingResponsesStream:
    """Responses stream context manager whose entry raises a SDK error."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def __aenter__(self) -> "_RejectingResponsesStream":
        raise self._error

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _AbortingResponsesStream:
    """Responses stream that yields one text delta, then raises a SDK error."""

    def __init__(self, delta: str, error: BaseException) -> None:
        self._events: list[Any] = [_text_delta_event(delta), error]
        self.closed = False

    async def __aenter__(self) -> "_AbortingResponsesStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    def __aiter__(self) -> "_AbortingResponsesStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        item = self._events.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeAnthropicStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = list(events)
        self._final = final
        self.closed = False

    async def __aenter__(self) -> "_FakeAnthropicStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    def __aiter__(self) -> "_FakeAnthropicStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def get_final_message(self) -> Any:
        return self._final


class _FakeMistralStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)
        self.closed = False

    async def __aenter__(self) -> "_FakeMistralStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    def __aiter__(self) -> "_FakeMistralStream":
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        item = self._events.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeGeminiClient:
    """Stand-in for ``google.genai.Client`` whose ``.aio`` is self-enterable."""

    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        async def _agen() -> Any:
            for chunk in chunks:
                yield chunk

        async def generate_content_stream(**kwargs: Any) -> Any:
            return _agen()

        self.models = SimpleNamespace(generate_content_stream=generate_content_stream)

    @property
    def aio(self) -> "_FakeGeminiClient":
        return self

    async def __aenter__(self) -> "_FakeGeminiClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class TestOpenAIChatStreamAccumulator:
    async def test_rebuilds_fragmented_arguments(self) -> None:
        accumulator = _OpenAIChatStreamAccumulator()
        accumulator.process_chunk(_make_chunk(text="Hello"))
        accumulator.process_chunk(
            _make_chunk(
                tool_calls=[
                    SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="get_time", arguments='{"'))
                ]
            )
        )
        accumulator.process_chunk(
            _make_chunk(
                tool_calls=[
                    SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments='tz": "UTC"}'))
                ]
            )
        )
        calls = accumulator.build_tool_calls()
        assert len(calls) == 1
        assert calls[0].id == "call-1"
        assert calls[0].function.name == "get_time"
        assert calls[0].function.arguments == '{"tz": "UTC"}'

    async def test_handles_two_parallel_tool_calls(self) -> None:
        accumulator = _OpenAIChatStreamAccumulator()
        accumulator.process_chunk(
            _make_chunk(
                tool_calls=[
                    SimpleNamespace(index=0, id="call-a", function=SimpleNamespace(name="f1", arguments="{}")),
                    SimpleNamespace(index=1, id="call-b", function=SimpleNamespace(name="f2", arguments="{}")),
                ]
            )
        )
        calls = accumulator.build_tool_calls()
        assert [c.function.name for c in calls] == ["f1", "f2"]
        assert [c.id for c in calls] == ["call-a", "call-b"]

    async def test_usage_only_chunk_is_captured_without_raising(self) -> None:
        accumulator = _OpenAIChatStreamAccumulator()
        accumulator.process_chunk(_make_chunk(text="Hi"))
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        usage_chunk = cast(Any, SimpleNamespace(id="chatcmpl-1", model="m-1", created=42, choices=[], usage=usage))
        accumulator.process_chunk(usage_chunk)
        response = accumulator.build_response()
        assert response.usage is usage
        assert response.choices[0].message.content == "Hi"

    async def test_reasoning_content_is_kept_out_of_body_text(self) -> None:
        accumulator = _OpenAIChatStreamAccumulator()
        accumulator.process_chunk(_make_chunk(reasoning_content="thinking..."))
        accumulator.process_chunk(_make_chunk(text="Answer"))
        response = accumulator.build_response()
        assert response.choices[0].message.content == "Answer"
        assert "".join(accumulator.reasoning_parts) == "thinking..."

    async def test_tool_call_signal(self) -> None:
        accumulator = _OpenAIChatStreamAccumulator()
        accumulator.process_chunk(_make_chunk(text="before"))
        assert accumulator.tool_call_signal is False
        accumulator.process_chunk(
            _make_chunk(
                tool_calls=[SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name="f", arguments=""))]
            )
        )
        assert accumulator.tool_call_signal is True


class TestOpenAIChatStreaming:
    """Reachability tests through the real ``_get_chat_completion_response`` chain."""

    async def _call_public_path(self, provider: OpenAI, mock_client: Any, interface: Any) -> Any:
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            return await provider._get_chat_completion_response(
                messages=[],
                model="m-1",
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt=None,
                interface=interface,
            )

    async def test_streams_deltas_and_captures_usage_via_public_path(self) -> None:
        usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _make_chunk(text="Hel"),
            _make_chunk(text="lo"),
            SimpleNamespace(id="chatcmpl-1", model="m-1", created=42, choices=[], usage=usage),
        ]
        stream = _FakeOpenAIStream(chunks)
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        completions = SimpleNamespace(create=AsyncMock(return_value=stream))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result, _ = await self._call_public_path(provider, mock_client, interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert interface.delta_emitted is True
        assert stream.closed is True
        assert result.answer == "Hello"
        assert result.usage.total_tokens == 15
        create_kwargs = completions.create.await_args.kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["stream_options"] == {"include_usage": True}

    async def test_rejecting_gateway_falls_back_to_single_non_streaming_call(self) -> None:
        error = _make_bad_request_error()
        non_streaming = _make_non_streaming_completion()
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        completions = SimpleNamespace(create=AsyncMock(side_effect=[error, non_streaming]))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        result, _ = await self._call_public_path(provider, mock_client, interface)

        assert completions.create.await_count == 2
        first_kwargs = completions.create.await_args_list[0].kwargs
        second_kwargs = completions.create.await_args_list[1].kwargs
        assert first_kwargs["stream"] is True
        assert "stream" not in second_kwargs
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert result.answer == "Hello"
        assert result.usage.total_tokens == 15

    async def test_no_fallback_after_first_emitted_delta(self) -> None:
        error = _make_bad_request_error()
        stream = _FakeOpenAIStream([_make_chunk(text="Hel"), error])
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        completions = SimpleNamespace(create=AsyncMock(return_value=stream))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with pytest.raises(ServiceResponseError):
            await self._call_public_path(provider, mock_client, interface)

        assert completions.create.await_count == 1
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel"]

    async def test_latched_rejection_is_not_retried(self) -> None:
        error = _make_bad_request_error()
        interface = _make_interface(delta_emitted=True)
        provider = OpenAI(token=TEST_TOKEN)
        completions = SimpleNamespace(create=AsyncMock(side_effect=error))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(ServiceResponseError):
                await provider._stream_chat_completion(
                    completion_kwargs={"model": "m-1", "messages": []},
                    interface=interface,
                )

        assert completions.create.await_count == 1

    async def test_cancellation_closes_stream_and_stops_deltas(self) -> None:
        chunks: list[Any] = [_make_chunk(text="safe"), asyncio.CancelledError(), _make_chunk(text="never")]
        stream = _FakeOpenAIStream(chunks)
        provider = OpenAI(token=TEST_TOKEN)
        interface = _make_interface()
        completions = SimpleNamespace(create=AsyncMock(return_value=stream))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(asyncio.CancelledError):
                await provider._stream_chat_completion(
                    completion_kwargs={"model": "m-1", "messages": []},
                    interface=interface,
                )
        assert stream.closed is True
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["safe"]

    async def test_latched_request_uses_non_streaming_call(self) -> None:
        final = _make_non_streaming_completion(text="ok")
        completions = SimpleNamespace(create=AsyncMock(return_value=final))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = OpenAI(token=TEST_TOKEN)
        interface = _make_interface(delta_emitted=True)
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            result, _ = await provider._get_chat_completion_response(
                messages=[],
                model="m-1",
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt=None,
                interface=interface,
            )
        assert completions.create.await_count == 1
        assert "stream" not in completions.create.await_args.kwargs
        interface.send_delta.assert_not_awaited()
        assert result.answer == "ok"


class TestOpenAIResponsesStreaming:
    """Reachability tests through the real ``_get_response_completion_response`` chain."""

    async def _call_public_path(self, provider: OpenAI, mock_client: Any, interface: Any) -> Any:
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            return await provider._get_response_completion_response(
                messages=[Message(role="user", content="hi")],
                model="gpt-5-mini",
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                system_prompt=None,
                interface=interface,
            )

    async def test_streams_deltas_via_public_path(self) -> None:
        final = _fake_responses_api_response("Hello")
        stream = _FakeResponsesStream([_text_delta_event("Hel"), _text_delta_event("lo")], final)
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        mock_client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: stream))

        result, _ = await self._call_public_path(provider, mock_client, interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert stream.closed is True
        assert interface.delta_emitted is True
        assert result.answer == "Hello"

    async def test_rejecting_gateway_falls_back_to_non_streaming_call(self) -> None:
        error = _make_bad_request_error()
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_fake_responses_api_response("Hello"))
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _RejectingResponsesStream(error),
                create=create_mock,
            )
        )

        result, _ = await self._call_public_path(provider, mock_client, interface)

        assert create_mock.await_count == 1
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert result.answer == "Hello"


class TestAnthropicStreaming:
    """Reachability tests through the real ``_get_chat_completion_response`` chain."""

    @staticmethod
    def _final_message(text: str = "Hello") -> AnthropicMessage:
        return AnthropicMessage(
            id="msg-1",
            content=[TextBlock(text=text, type="text")],
            model="claude-sonnet-5",
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
            usage=AnthropicUsage(input_tokens=1, output_tokens=2),
        )

    async def _call_public_path(
        self,
        provider: Anthropic,
        fake_client: Any,
        interface: Any,
        messages: list[Any] | None = None,
    ) -> Any:
        with patch.object(type(provider), "client", new_callable=PropertyMock, return_value=fake_client):
            with patch(
                "chibi.services.providers.provider.prepare_system_prompt",
                new=AsyncMock(return_value="sp"),
            ):
                return await provider._get_chat_completion_response(
                    messages=messages or [],
                    model="claude-sonnet-5",
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    interface=interface,
                )

    async def test_streams_deltas_via_public_path_and_never_streams_thinking(self) -> None:
        events: list[Any] = [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=ThinkingDelta(type="thinking_delta", thinking="hmm")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hel")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="lo")
            ),
        ]
        fake_stream = _FakeAnthropicStream(events, self._final_message())
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream))
        provider = Anthropic(token=TEST_TOKEN)
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, fake_client, interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert interface.delta_emitted is True
        assert result.answer == "Hello"
        assert result.usage is not None
        assert result.usage.total_tokens == 3

    async def test_latched_request_uses_non_streaming_call(self) -> None:
        create_mock = AsyncMock(return_value=self._final_message(text="ok"))
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
        provider = Anthropic(token=TEST_TOKEN)
        interface = _make_interface(delta_emitted=True)

        result, _ = await self._call_public_path(provider, fake_client, interface)

        assert create_mock.await_count == 1
        interface.send_delta.assert_not_awaited()
        assert result.answer == "ok"


class TestAnthropicStreamAccumulation:
    async def test_never_streams_thinking_and_handles_parallel_tool_calls(self) -> None:
        events: list[Any] = [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=ThinkingDelta(type="thinking_delta", thinking="hmm")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hi ")
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=1,
                content_block=ToolUseBlock(id="t1", name="f1", input={}, type="tool_use"),
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=2,
                content_block=ToolUseBlock(id="t2", name="f2", input={}, type="tool_use"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="late")
            ),
        ]
        final = SimpleNamespace(
            content=[
                TextBlock(text="Hi ", type="text"),
                ToolUseBlock(id="t1", name="f1", input={}, type="tool_use"),
                ToolUseBlock(id="t2", name="f2", input={}, type="tool_use"),
            ]
        )
        provider = AnthropicFriendlyProvider(token=TEST_TOKEN)
        interface = _make_interface()
        fake_stream = _FakeAnthropicStream(events, final)
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream))
        with patch.object(AnthropicFriendlyProvider, "client", new_callable=PropertyMock, return_value=fake_client):
            message = await provider._stream_generate_content(
                model="claude",
                system=[],
                messages=[],
                interface=interface,
            )
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hi "]
        assert fake_stream.closed is True
        tool_parts = [part for part in message.content if isinstance(part, ToolUseBlock)]
        assert [part.name for part in tool_parts] == ["f1", "f2"]


class TestGeminiStreaming:
    """Reachability tests through the real ``_get_chat_completion_response`` chain."""

    async def test_streams_deltas_via_public_path_and_filters_thoughts(self) -> None:
        chunks = [
            GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text="Hel")]))]),
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="hidden thought", thought=True)]))]
            ),
            GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text="lo")]))]),
            GenerateContentResponse(
                usage_metadata=GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    candidates_token_count=2,
                    total_token_count=3,
                )
            ),
        ]
        provider = Gemini(token=TEST_TOKEN)
        interface = _make_interface()
        fake_client = _FakeGeminiClient(chunks)
        with patch("chibi.services.providers.gemini_native.Client", lambda **kwargs: fake_client):
            with patch(
                "chibi.services.providers.gemini_native.prepare_system_prompt",
                new=AsyncMock(return_value="sp"),
            ):
                result, _ = await provider._get_chat_completion_response(
                    messages=[],
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    interface=interface,
                )

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert interface.delta_emitted is True
        assert result.answer == "Hello"
        assert result.usage is not None
        assert result.usage.total_tokens == 3


class TestMistralStreaming:
    """Reachability tests through the real ``_get_chat_completion_response`` chain."""

    def _chunk(
        self,
        content: str | list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        usage: Any = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        chunk = SimpleNamespace(id="cmpl-1", model="mistral", created=7, choices=[choice], usage=usage)
        return SimpleNamespace(data=chunk)

    async def _call_public_path(self, provider: MistralAI, fake_client: Any, interface: Any) -> Any:
        with patch(
            "chibi.services.providers.mistralai_native.prepare_system_prompt",
            new=AsyncMock(return_value="sp"),
        ):
            return await provider._get_chat_completion_response(
                messages=[],
                model="mistral-medium-latest",
                user=User(id=1),
                caller_storage_id=1,
                caller_thread_id=0,
                interface=interface,
            )

    async def test_streams_deltas_via_public_path_and_captures_usage(self) -> None:
        usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        events = [
            self._chunk(content="Hel"),
            self._chunk(content="lo"),
            self._chunk(finish_reason="stop", usage=usage),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert fake_stream.closed is True
        assert interface.delta_emitted is True
        assert result.answer == "Hello"
        assert result.usage.total_tokens == 3

    async def test_thinkchunk_delta_does_not_crash_and_stays_out_of_the_answer(self) -> None:
        events = [
            self._chunk(content=[ThinkChunk(thinking=[TextChunk(text="secret reasoning")])]),
            self._chunk(content=[TextChunk(text="Hel")]),
            self._chunk(content=[TextChunk(text="lo")]),
            self._chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert fake_stream.closed is True
        assert result.answer == "Hello"
        assert "secret reasoning" not in result.answer
        for call in interface.send_delta.await_args_list:
            assert "secret reasoning" not in call.args[0]

    async def test_mixed_chunk_list_yields_ordered_text_concatenation(self) -> None:
        events = [
            self._chunk(
                content=[
                    ThinkChunk(thinking=[TextChunk(text="hmm")]),
                    TextChunk(text="Hel"),
                    TextChunk(text="lo"),
                ]
            ),
            self._chunk(finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hello"]
        assert result.answer == "Hello"
        assert "hmm" not in result.answer

    async def test_rebuilds_fragmented_tool_calls(self) -> None:
        usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        fragments = [
            SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="get", arguments='{"a"')),
            SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments=": 1}")),
        ]
        events = [
            self._chunk(content="I will call a tool."),
            self._chunk(tool_calls=fragments),
            self._chunk(finish_reason="tool_calls", usage=usage),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()
        response = await provider._stream_generate_content(
            model="mistral-medium-latest",
            messages=[],
            interface=interface,
        )
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["I will call a tool."]
        assert fake_stream.closed is True
        message = response.choices[0].message
        assert message.content == "I will call a tool."
        tool_calls = cast(Any, message.tool_calls)
        assert [tc.function.name for tc in tool_calls] == ["get"]
        assert tool_calls[0].function.arguments == '{"a": 1}'

    async def test_rejecting_gateway_falls_back_to_single_non_streaming_call(self) -> None:
        stream_async = AsyncMock(side_effect=_make_mistral_sdk_error(400))
        complete_async = AsyncMock(return_value=_make_mistral_completion())
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert stream_async.await_count == 1
        assert complete_async.await_count == 1
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert result.answer == "Hello"
        assert result.usage.total_tokens == 3

    async def test_not_found_gateway_falls_back_to_non_streaming_call(self) -> None:
        stream_async = AsyncMock(side_effect=_make_mistral_sdk_error(404))
        complete_async = AsyncMock(return_value=_make_mistral_completion())
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface()

        result, _ = await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert stream_async.await_count == 1
        assert complete_async.await_count == 1
        assert result.answer == "Hello"

    async def test_no_fallback_after_first_emitted_delta(self) -> None:
        fake_stream = _FakeMistralStream([self._chunk(content="Hel"), _make_mistral_sdk_error(400)])
        stream_async = AsyncMock(return_value=fake_stream)
        complete_async = AsyncMock()
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface()

        with pytest.raises(SDKError):
            await self._call_public_path(provider, provider.__dict__["_client"], interface)

        assert complete_async.await_count == 0
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel"]
        assert interface.delta_emitted is True

    async def test_latched_rejection_is_not_retried(self) -> None:
        stream_async = AsyncMock(side_effect=_make_mistral_sdk_error(400))
        complete_async = AsyncMock()
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface(delta_emitted=True)

        with pytest.raises(SDKError):
            await provider._stream_generate_content(
                model="mistral-medium-latest",
                messages=[],
                interface=interface,
            )

        assert stream_async.await_count == 1
        assert complete_async.await_count == 0

    async def test_rejected_stream_status_is_outside_the_retry_policy(self) -> None:
        assert _is_retryable_mistral_error(_make_mistral_sdk_error(400)) is False
        assert _is_retryable_mistral_error(_make_mistral_sdk_error(404)) is False
        assert _is_retryable_mistral_error(_make_mistral_sdk_error(401)) is False
        assert _is_retryable_mistral_error(_make_mistral_sdk_error(403)) is False


@contextmanager
def _zeroed_retry_waits() -> Any:
    """Silence the 30/40/80 s tenacity waits without touching the attempt counts."""
    decorated = [MistralAI.get_chat_response, OpenAIFriendlyProvider.get_chat_response]
    saved = [(fn, fn.retry.wait) for fn in decorated]
    for fn, _ in saved:
        fn.retry.wait = wait_fixed(0)
    try:
        yield
    finally:
        for fn, wait_strategy in saved:
            fn.retry.wait = wait_strategy


class TestNonStreamingFallbackSpendLatch:
    """A failed non-streaming fallback is issued once, not once per retry.

    The fallback lives inside the tenacity-decorated ``get_chat_response``:
    a transient failure of the fallback itself re-enters the whole method,
    where the stream creation fails deterministically again. The
    spent-fallback latch stores the fallback's error, so re-entry re-raises
    it instead of issuing a second fallback, while the transient error keeps
    driving the retry attempts and the provider failover classification.
    """

    async def test_mistral_failed_fallback_is_not_reissued_on_retry(self) -> None:
        stream_async = AsyncMock(side_effect=_make_mistral_sdk_error(400))
        complete_async = AsyncMock(side_effect=_make_mistral_sdk_error(429))
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface()

        with (
            _zeroed_retry_waits(),
            patch(
                "chibi.services.providers.mistralai_native.prepare_system_prompt",
                new=AsyncMock(return_value="sp"),
            ),
        ):
            with pytest.raises(SDKError) as exc_info:
                await MistralAI.get_chat_response(
                    provider,
                    messages=[],
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    interface=interface,
                )

        assert exc_info.value.raw_response.status_code == 429
        assert stream_async.await_count == 4
        assert complete_async.await_count == 1

    async def test_openai_failed_fallback_is_not_reissued_on_retry(self) -> None:
        calls = {"stream": 0, "complete": 0}

        async def create(**kwargs: Any) -> Any:
            if kwargs.get("stream"):
                calls["stream"] += 1
                raise _make_bad_request_error()
            calls["complete"] += 1
            raise _make_rate_limit_error()

        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        provider = OpenAI(token=TEST_TOKEN)
        interface = _make_interface()

        with (
            _zeroed_retry_waits(),
            patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client),
            patch(
                "chibi.services.providers.provider.prepare_system_prompt",
                new=AsyncMock(return_value="sp"),
            ),
        ):
            with pytest.raises(ServiceRateLimitError):
                await OpenAIFriendlyProvider.get_chat_response(
                    provider,
                    messages=[],
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    interface=interface,
                )

        assert calls == {"stream": 4, "complete": 1}

    @pytest.mark.parametrize("status_code", [429, 500])
    async def test_plain_streaming_retryable_statuses_still_retry_without_fallback(self, status_code: int) -> None:
        stream_async = AsyncMock(side_effect=_make_mistral_sdk_error(status_code))
        complete_async = AsyncMock()
        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(
            chat=SimpleNamespace(stream_async=stream_async, complete_async=complete_async)
        )
        interface = _make_interface()

        with (
            _zeroed_retry_waits(),
            patch(
                "chibi.services.providers.mistralai_native.prepare_system_prompt",
                new=AsyncMock(return_value="sp"),
            ),
        ):
            with pytest.raises(SDKError) as exc_info:
                await MistralAI.get_chat_response(
                    provider,
                    messages=[],
                    user=User(id=1),
                    caller_storage_id=1,
                    caller_thread_id=0,
                    interface=interface,
                )

        assert exc_info.value.raw_response.status_code == status_code
        assert stream_async.await_count == 4
        assert complete_async.await_count == 0
        assert non_streaming_fallback_spent_error(interface) is None


class TestLatch:
    async def test_session_emit_latches_the_request(self) -> None:
        interface = _make_interface()
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("chunk-1")
        assert interface.delta_emitted is True
        assert delta_latched(interface) is True
        assert delta_streaming_allowed(interface) is False

    async def test_non_streaming_interface_never_emits(self) -> None:
        interface: Any = SimpleNamespace(streaming_enabled=False, delta_emitted=False, send_delta=AsyncMock())
        assert delta_streaming_allowed(interface) is False
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("chunk")
        interface.send_delta.assert_not_awaited()
        assert delta_latched(interface) is False

    async def test_none_interface_is_safe(self) -> None:
        assert delta_streaming_allowed(None) is False
        assert delta_latched(None) is False
        async with AckSentinelStreamSession(None) as deltas:
            await deltas.emit("chunk")

    async def test_non_streaming_fallback_latch_roundtrip(self) -> None:
        interface = _make_interface()
        assert non_streaming_fallback_spent_error(interface) is None
        error = ValueError("fallback failed")
        mark_non_streaming_fallback_spent(interface, error)
        assert non_streaming_fallback_spent_error(interface) is error

    async def test_non_streaming_fallback_latch_none_interface_is_safe(self) -> None:
        assert non_streaming_fallback_spent_error(None) is None
        mark_non_streaming_fallback_spent(None, ValueError("fallback failed"))


class TestAnthropicFamilyStreamFallback:
    """D7: the Anthropic family (incl. the MiniMax Anthropic-compatible gateway)
    falls back to the non-streaming call after the latch, never re-emitting."""

    @staticmethod
    def _final(text: str) -> AnthropicMessage:
        return AnthropicMessage(
            id="msg-1",
            content=[TextBlock(text=text, type="text")],
            model="claude-sonnet-5",
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
            usage=AnthropicUsage(input_tokens=1, output_tokens=2),
        )

    @classmethod
    async def _run_generate(cls, provider: Any, fake_client: Any, interface: Any) -> AnthropicMessage:
        with patch.object(type(provider), "client", new_callable=PropertyMock, return_value=fake_client):
            with patch.object(gpt_settings, "retries", 2):
                with patch("chibi.services.providers.provider.sleep", new=AsyncMock()):
                    return await provider._generate_content(
                        model="claude-sonnet-5",
                        system_prompt="sp",
                        messages=[],
                        interface=interface,
                    )

    async def test_empty_stream_after_delta_retries_non_streaming(self) -> None:
        """An empty streamed answer latches the request; the retry is non-streaming."""
        text_event = RawContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hel")
        )
        fake_stream = _FakeAnthropicStream([text_event], SimpleNamespace(content=[]))
        create_mock = AsyncMock(return_value=self._final("ok"))
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream, create=create_mock))
        provider = Anthropic(token=TEST_TOKEN)
        interface = _make_interface()

        result = await self._run_generate(provider, fake_client, interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel"]
        assert interface.delta_emitted is True
        assert create_mock.await_count == 1
        assert cast(TextBlock, result.content[0]).text == "ok"

    async def test_minimax_gateway_latched_retry_is_non_streaming(self) -> None:
        """The MiniMax Anthropic-compatible gateway follows the same fallback policy."""
        text_event = RawContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Mi")
        )
        fake_stream = _FakeAnthropicStream([text_event], SimpleNamespace(content=[]))
        create_mock = AsyncMock(return_value=self._final("minimax ok"))
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream, create=create_mock))
        provider = Minimax(token=TEST_TOKEN)
        interface = _make_interface()

        result = await self._run_generate(provider, fake_client, interface)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Mi"]
        assert interface.delta_emitted is True
        assert create_mock.await_count == 1
        assert cast(TextBlock, result.content[0]).text == "minimax ok"


class TestAckSentinelStreaming:
    """The ACK gate must keep a pure-ACK answer completely off the delta channel."""

    async def _run_stream(self, chunks: list[Any]) -> tuple[Any, Any, Any]:
        provider = OpenAI(token=TEST_TOKEN)
        interface = _make_interface()
        stream = _FakeOpenAIStream(chunks)
        completions = SimpleNamespace(create=AsyncMock(return_value=stream))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            response = await provider._stream_chat_completion(
                completion_kwargs={"model": "m-1", "messages": []},
                interface=interface,
            )
        return response, interface, stream

    async def test_pure_ack_emits_no_delta(self) -> None:
        """A pure-ACK answer streamed in fragments never emits a single delta."""
        chunks = [_make_chunk(text="<chibi"), _make_chunk(text=">ACK</chibi"), _make_chunk(text=">")]
        result, interface, stream = await self._run_stream(chunks)

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert stream.closed is True
        assert result.choices[0].message.content == "<chibi>ACK</chibi>"

    async def test_pure_ack_with_whitespace_emits_no_delta(self) -> None:
        """A pure-ACK answer padded with whitespace never emits a delta either."""
        chunks = [_make_chunk(text=" "), _make_chunk(text="<chibi>ACK</chibi>"), _make_chunk(text="\n")]
        _, interface, _ = await self._run_stream(chunks)

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False

    async def test_answer_starting_like_the_marker_streams_normally(self) -> None:
        """A normal answer diverging from the marker releases the buffered prefix intact."""
        chunks = [_make_chunk(text="<chibi"), _make_chunk(text="X is the answer")]
        result, interface, _ = await self._run_stream(chunks)

        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["<chibiX is the answer"]
        assert result.choices[0].message.content == "<chibiX is the answer"

    async def test_marker_inside_streamed_text_is_stripped_from_deltas(self) -> None:
        """A streamed mixed answer carries no sentinel fragment on the delta channel."""
        chunks = [_make_chunk(text="Result: "), _make_chunk(text="<chibi>ACK</chibi> done"), _make_chunk(text="!")]
        result, interface, _ = await self._run_stream(chunks)

        emitted = [call.args[0] for call in interface.send_delta.await_args_list]
        assert "".join(emitted) == "Result:  done!"
        assert all("<chibi>ACK</chibi>" not in delta for delta in emitted)
        assert result.choices[0].message.content == "Result: <chibi>ACK</chibi> done!"


class TestAckSentinelNativeAdapters:
    """The ACK gate must cover every native-adapter delta-emit path too."""

    async def test_responses_api_pure_ack_emits_no_delta(self) -> None:
        final = _fake_responses_api_response("<chibi>ACK</chibi>")
        stream = _FakeResponsesStream([_text_delta_event("<chibi"), _text_delta_event(">ACK</chibi>")], final)
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        mock_client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: stream))

        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            response = await provider._stream_response_completion(
                request_kwargs={"model": "gpt-5-mini"},
                interface=interface,
            )

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert stream.closed is True
        assert response.output_text == "<chibi>ACK</chibi>"

    async def test_responses_api_marker_inside_streamed_text_is_stripped(self) -> None:
        final = _fake_responses_api_response("Result: <chibi>ACK</chibi> done!")
        stream = _FakeResponsesStream(
            [_text_delta_event("Result: "), _text_delta_event("<chibi>ACK</chibi> done"), _text_delta_event("!")],
            final,
        )
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        mock_client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: stream))

        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            await provider._stream_response_completion(
                request_kwargs={"model": "gpt-5-mini"},
                interface=interface,
            )

        emitted = [call.args[0] for call in interface.send_delta.await_args_list]
        assert "".join(emitted) == "Result:  done!"
        assert all("<chibi>ACK</chibi>" not in delta for delta in emitted)
        assert interface.delta_emitted is True

    async def test_responses_api_fallback_discards_aborted_attempts_hold_back(self) -> None:
        """A mid-stream 400 before any emitted delta must not flush the speculative prefix.

        The stream holds back a sentinel-compatible prefix ("<chibi"), then the
        gateway rejects it. The non-streaming fallback regenerates the answer,
        so the aborted attempt's held-back text must be discarded, not emitted
        ahead of the regenerated answer.
        """
        error = _make_bad_request_error()
        interface = _make_interface()
        provider = OpenAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_fake_responses_api_response("Hello"))
        mock_client = SimpleNamespace(
            responses=SimpleNamespace(
                stream=lambda **kwargs: _AbortingResponsesStream("<chibi", error),
                create=create_mock,
            )
        )

        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            response = await provider._stream_response_completion(
                request_kwargs={"model": "gpt-5-mini"},
                interface=interface,
            )

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert create_mock.await_count == 1
        assert response.output_text == "Hello"

    async def test_anthropic_pure_ack_emits_no_delta(self) -> None:
        events: list[Any] = [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="<chibi")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text=">ACK</chibi>")
            ),
        ]
        final = SimpleNamespace(content=[TextBlock(text="<chibi>ACK</chibi>", type="text")])
        fake_stream = _FakeAnthropicStream(events, final)
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream))
        provider = AnthropicFriendlyProvider(token=TEST_TOKEN)
        interface = _make_interface()

        with patch.object(AnthropicFriendlyProvider, "client", new_callable=PropertyMock, return_value=fake_client):
            message = await provider._stream_generate_content(
                model="claude-sonnet-5",
                system=[],
                messages=[],
                interface=interface,
            )

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert fake_stream.closed is True
        assert cast(TextBlock, message.content[0]).text == "<chibi>ACK</chibi>"

    async def test_anthropic_marker_inside_streamed_text_is_stripped(self) -> None:
        events: list[Any] = [
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Result: ")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="<chibi>ACK</chibi> done")
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="!")
            ),
        ]
        final = SimpleNamespace(content=[TextBlock(text="Result: <chibi>ACK</chibi> done!", type="text")])
        fake_stream = _FakeAnthropicStream(events, final)
        fake_client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: fake_stream))
        provider = AnthropicFriendlyProvider(token=TEST_TOKEN)
        interface = _make_interface()

        with patch.object(AnthropicFriendlyProvider, "client", new_callable=PropertyMock, return_value=fake_client):
            await provider._stream_generate_content(
                model="claude-sonnet-5",
                system=[],
                messages=[],
                interface=interface,
            )

        emitted = [call.args[0] for call in interface.send_delta.await_args_list]
        assert "".join(emitted) == "Result:  done!"
        assert all("<chibi>ACK</chibi>" not in delta for delta in emitted)

    async def test_gemini_pure_ack_emits_no_delta(self) -> None:
        chunks = [
            GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text="<chibi")]))]),
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text=">ACK</chibi>")]))]
            ),
        ]
        provider = Gemini(token=TEST_TOKEN)
        interface = _make_interface()
        fake_client = _FakeGeminiClient(chunks)

        response = await provider._generate_content_streamed(
            client=cast(Any, fake_client),
            model="gemini-2.5-flash",
            contents=Content(role="user", parts=[Part(text="hi")]),
            config=GenerateContentConfig(),
            interface=interface,
        )

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert response.text == "<chibi>ACK</chibi>"

    async def test_gemini_marker_inside_streamed_text_is_stripped(self) -> None:
        chunks = [
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="Result: ")]))]
            ),
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="<chibi>ACK</chibi> done")]))]
            ),
            GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text="!")]))]),
        ]
        provider = Gemini(token=TEST_TOKEN)
        interface = _make_interface()
        fake_client = _FakeGeminiClient(chunks)

        await provider._generate_content_streamed(
            client=cast(Any, fake_client),
            model="gemini-2.5-flash",
            contents=Content(role="user", parts=[Part(text="hi")]),
            config=GenerateContentConfig(),
            interface=interface,
        )

        emitted = [call.args[0] for call in interface.send_delta.await_args_list]
        assert "".join(emitted) == "Result:  done!"
        assert all("<chibi>ACK</chibi>" not in delta for delta in emitted)

    async def test_mistral_pure_ack_emits_no_delta(self) -> None:
        events = [
            self._mistral_chunk("<chibi"),
            self._mistral_chunk(">ACK</chibi>"),
            self._mistral_chunk(
                finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
            ),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()

        response = await provider._stream_generate_content(
            model="mistral-medium-latest",
            messages=[],
            interface=interface,
        )

        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False
        assert fake_stream.closed is True
        assert response.choices[0].message.content == "<chibi>ACK</chibi>"

    async def test_mistral_marker_inside_streamed_text_is_stripped(self) -> None:
        events = [
            self._mistral_chunk("Result: "),
            self._mistral_chunk("<chibi>ACK</chibi> done"),
            self._mistral_chunk("!"),
            self._mistral_chunk(
                finish_reason="stop", usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
            ),
        ]
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

        provider = MistralAI(token=TEST_TOKEN)
        provider.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(stream_async=_stream_async))
        interface = _make_interface()

        await provider._stream_generate_content(
            model="mistral-medium-latest",
            messages=[],
            interface=interface,
        )

        emitted = [call.args[0] for call in interface.send_delta.await_args_list]
        assert "".join(emitted) == "Result:  done!"
        assert all("<chibi>ACK</chibi>" not in delta for delta in emitted)

    @staticmethod
    def _mistral_chunk(
        content: str | None = None,
        usage: Any = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(content=content, tool_calls=None)
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        return SimpleNamespace(
            data=SimpleNamespace(id="cmpl-1", model="mistral", created=7, choices=[choice], usage=usage)
        )


class TestAckSentinelStreamSession:
    """The gate lives inside the stream session: no session, no emission.

    Emission exists only as ``AckSentinelStreamSession.emit``, and the gate is
    a plain attribute of the session object, so a streaming attempt cannot
    share, inherit or leave behind gate state, and the drain at session exit
    runs on every exit path of the ``async with`` block.
    """

    async def test_pure_ack_never_leaks_through_a_session(self) -> None:
        """Even an adapter wiring nothing beyond the session cannot leak the sentinel."""
        interface = _make_interface()
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("<chibi")
            await deltas.emit(">ACK</chibi>")
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False

    async def test_emit_outside_the_with_block_fails_loudly(self) -> None:
        """There is no emission path that bypasses the session lifecycle."""
        interface = _make_interface()
        session = AckSentinelStreamSession(interface)
        with pytest.raises(RuntimeError, match="async with"):
            await session.emit("chunk")
        interface.send_delta.assert_not_awaited()

    async def test_session_releases_legitimate_text(self) -> None:
        interface = _make_interface()
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("<chibi")
            await deltas.emit("X is the answer")
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["<chibiX is the answer"]
        assert interface.delta_emitted is True

    async def test_no_state_leaks_between_attempts_even_without_clean_exit(self) -> None:
        """A fresh attempt starts on a clean gate without any reset call.

        The first attempt is an async generator abandoned mid-hold: it enters
        the session, emits '<chibi' (which the gate holds back), suspends at the
        yield and is then dropped without being closed, so its ``__aexit__``
        never runs and the held buffer is never flushed. Because the gate lives
        in the session object and not on the interface, the second session still
        starts empty and 'Hello' is emitted alone — never spliced into
        '<chibiHello'.
        """
        interface = _make_interface()

        def abandoned_stream() -> Any:
            async def stream_body() -> Any:
                async with AckSentinelStreamSession(interface) as deltas:
                    await deltas.emit("<chibi")
                    yield "suspended mid-attempt"

            return stream_body()

        abandoned = abandoned_stream()
        await abandoned.__anext__()
        del abandoned
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("Hello")
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hello"]
        assert interface.delta_emitted is True

    async def test_exit_emits_held_back_tail_of_legitimate_answer(self) -> None:
        interface = _make_interface()
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("Done <chibi")
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Done ", "<chibi"]

    async def test_exit_after_pure_ack_emits_nothing(self) -> None:
        interface = _make_interface()
        async with AckSentinelStreamSession(interface) as deltas:
            await deltas.emit("<chibi>ACK</chibi>")
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False

    async def test_early_return_still_drains_the_gate(self) -> None:
        interface = _make_interface()

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("Done <chibi")
                return

        await stream_loop()
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Done ", "<chibi"]

    async def test_exception_still_emits_released_tail_and_propagates(self) -> None:
        """A stream dying mid-answer must not lose the already-diverged tail."""
        interface = _make_interface()

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("Answer ends with <chib")
                raise RuntimeError("stream died")

        with pytest.raises(RuntimeError, match="stream died"):
            await stream_loop()
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Answer ends with ", "<chib"]

    async def test_cancellation_still_emits_released_tail_and_propagates(self) -> None:
        interface = _make_interface()

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("Answer ends with <chib")
                raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await stream_loop()
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Answer ends with ", "<chib"]

    async def test_exception_discards_unreleased_hold_back(self) -> None:
        """An abort before the answer diverged must not emit speculative text.

        The held '<chibi' never became a real answer, and a non-streaming
        fallback may regenerate the response, so emitting it would splice
        aborted-attempt text ahead of the replacement answer.
        """
        interface = _make_interface()

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("<chibi")
                raise RuntimeError("stream died")

        with pytest.raises(RuntimeError, match="stream died"):
            await stream_loop()
        interface.send_delta.assert_not_awaited()
        assert interface.delta_emitted is False

    async def test_flush_failure_during_abort_does_not_mask_the_exception(self) -> None:
        interface = _make_interface()
        interface.send_delta = AsyncMock(side_effect=[None, ConnectionError("interface gone")])

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("Answer ends with <chib")
                raise RuntimeError("stream died")

        with pytest.raises(RuntimeError, match="stream died"):
            await stream_loop()

    async def test_flush_failure_on_clean_exit_propagates(self) -> None:
        interface = _make_interface()
        interface.send_delta = AsyncMock(side_effect=[None, ConnectionError("interface gone")])

        async def stream_loop() -> None:
            async with AckSentinelStreamSession(interface) as deltas:
                await deltas.emit("Done <chib")

        with pytest.raises(ConnectionError, match="interface gone"):
            await stream_loop()

    async def test_none_interface_is_safe_for_the_session(self) -> None:
        async with AckSentinelStreamSession(None) as deltas:
            await deltas.emit("chunk")
