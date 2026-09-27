"""Unit tests for live response-delta streaming in the provider layer."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from mistralai.models import UsageInfo
from openai.types.completion_usage import CompletionUsage

from chibi.models import User
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import (
    AnthropicFriendlyProvider,
    _OpenAIChatStreamAccumulator,
)
from chibi.services.providers.streaming import (
    delta_latched,
    delta_streaming_allowed,
    emit_delta,
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


class TestOpenAIChatStreaming:
    async def test_streams_deltas_and_captures_usage(self) -> None:
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
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            response = await provider._stream_chat_completion(
                completion_kwargs={"model": "m-1", "messages": []},
                interface=interface,
            )
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["Hel", "lo"]
        assert interface.delta_emitted is True
        assert stream.closed is True
        assert response.usage is usage
        assert response.choices[0].message.content == "Hello"
        create_kwargs = completions.create.await_args.kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["stream_options"] == {"include_usage": True}

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
        final = _make_chunk(text="ok")
        completions = SimpleNamespace(create=AsyncMock(return_value=final))
        mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        provider = OpenAI(token=TEST_TOKEN)
        interface = _make_interface(delta_emitted=True)
        with patch.object(OpenAI, "client", new_callable=PropertyMock, return_value=mock_client):
            with pytest.raises(Exception):
                await provider._get_chat_completion_response(
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


class TestLatch:
    async def test_emit_delta_latches_the_request(self) -> None:
        interface = _make_interface()
        await emit_delta(interface, "chunk-1")
        assert interface.delta_emitted is True
        assert delta_latched(interface) is True
        assert delta_streaming_allowed(interface) is False

    async def test_non_streaming_interface_never_emits(self) -> None:
        interface: Any = SimpleNamespace(streaming_enabled=False, delta_emitted=False, send_delta=AsyncMock())
        assert delta_streaming_allowed(interface) is False
        await emit_delta(interface, "chunk")
        interface.send_delta.assert_not_awaited()
        assert delta_latched(interface) is False

    async def test_none_interface_is_safe(self) -> None:
        assert delta_streaming_allowed(None) is False
        assert delta_latched(None) is False
        await emit_delta(None, "chunk")


class _FakeGeminiClient:
    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        async def _agen() -> Any:
            for chunk in chunks:
                yield chunk

        async def generate_content_stream(**kwargs: Any) -> Any:
            return _agen()

        self.models = SimpleNamespace(generate_content_stream=generate_content_stream)


class TestGeminiStreaming:
    async def test_filters_thought_parts_and_signals_tool_calls(self) -> None:
        chunks = [
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="secret thought", thought=True)]))]
            ),
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="visible")]))]
            ),
            GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(
                            role="model",
                            parts=[Part(function_call=cast(Any, {"name": "f", "args": {}}))],
                        )
                    )
                ]
            ),
            GenerateContentResponse(
                candidates=[Candidate(content=Content(role="model", parts=[Part(text="more thoughts", thought=True)]))]
            ),
        ]
        provider = Gemini(token=TEST_TOKEN)
        interface = _make_interface()
        response = await provider._generate_content_streamed(
            client=cast(Any, _FakeGeminiClient(chunks)),
            model="models/m",
            contents=cast(Any, [{"role": "user", "parts": [{"text": "hi"}]}]),
            config=cast(Any, None),
            interface=interface,
        )
        assert [call.args[0] for call in interface.send_delta.await_args_list] == ["visible"]
        assert provider._get_text(response) == "visible"
        assert len(cast(Any, response.function_calls)) == 1


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


class TestAnthropicStreaming:
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
        return self._events.pop(0)


class TestMistralStreaming:
    def _chunk(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        usage: Any = None,
        finish_reason: str | None = None,
    ) -> Any:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        chunk = SimpleNamespace(id="cmpl-1", model="mistral", created=7, choices=[choice], usage=usage)
        return SimpleNamespace(data=chunk)

    async def test_rebuilds_tool_calls_and_usage(self) -> None:
        usage = UsageInfo(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        fragments = [
            SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name="get", arguments='{"a"')),
            SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments=": 1}")),
        ]
        events = [
            self._chunk(content="I will call a tool."),
            self._chunk(tool_calls=fragments),
            self._chunk(finish_reason="tool_calls", usage=usage),
        ]
        provider = MistralAI(token=TEST_TOKEN)
        fake_stream = _FakeMistralStream(events)

        async def _stream_async(**kwargs: Any) -> Any:
            return fake_stream

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
        assert response.usage.total_tokens == 7
