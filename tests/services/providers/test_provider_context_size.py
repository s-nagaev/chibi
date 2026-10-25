"""Tests for provider-to-utils context-size wiring."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import httpx
import pytest
from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from tenacity import AsyncRetrying, stop_after_attempt, wait_none

from chibi.exceptions import ContextLengthExceededError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.context_measurement import SizeProvenance
from chibi.services.interface import UserInterface
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.provider import CHAT_RETRY_PREDICATE, OpenAIFriendlyProvider
from chibi.services.providers.tools.schemas import ToolResponseSchema
from chibi.services.providers.utils import prepare_system_prompt
from chibi.services.usage_cache import UsageCacheStore


class _PrepareSystemPromptTracker:
    """Async callable that records prepare_system_prompt invocations.

    Stores a snapshot of each call's kwargs (with conversation_messages copied)
    and the returned prompt string.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> str:
        result = await prepare_system_prompt(*args, **kwargs)
        snapshot = dict(kwargs)
        if "conversation_messages" in snapshot:
            snapshot["conversation_messages"] = list(snapshot["conversation_messages"])
        self.calls.append((snapshot, result))
        return result


TEST_TOKEN = "test-token-for-context-size"
TEST_MODEL = "kimi-k2.6"


def _create_final_answer_response(answer: str) -> ChatCompletion:
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


def _create_tool_call_response(tool_name: str, arguments: str = "{}", tool_call_id: str = "call_1") -> ChatCompletion:
    """Build a ChatCompletion containing a single tool call.

    Args:
        tool_name: Name of the tool the assistant requests.
        arguments: JSON-encoded arguments for the tool call.
        tool_call_id: Identifier of the tool call (distinct ids let a test
            run several tool exchanges in one turn).

    Returns:
        A ChatCompletion object whose assistant message carries one tool call.
    """
    tool_call = ChatCompletionMessageToolCall(
        id=tool_call_id,
        type="function",
        function=Function(name=tool_name, arguments=arguments),
    )
    mock_choice = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Calling tool",
            tool_calls=[tool_call],
        ),
        finish_reason="tool_calls",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _patch_client(responses: list[ChatCompletion]) -> Any:
    """Patch the provider client property to return canned completions.

    Args:
        responses: ChatCompletion objects returned sequentially by the mocked client.

    Returns:
        A context manager patching the ``client`` property of MoonshotAI.
    """
    completions = SimpleNamespace(create=AsyncMock(side_effect=responses))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client)


@pytest.fixture
def provider() -> MoonshotAI:
    """Create a MoonshotAI provider instance with a dummy token."""
    return MoonshotAI(token=TEST_TOKEN)


@pytest.fixture
def _mocked_metrics() -> Iterator[MagicMock]:
    """Patch MetricsService to keep usage metrics out of the test run."""
    with patch("chibi.services.providers.provider.MetricsService") as metrics_mock:
        yield metrics_mock


@pytest.fixture
def interface() -> UserInterface:
    """Return a minimal UserInterface stub for context-size tests."""
    # Cast is safe: SimpleNamespace provides the attributes the production path reads from UserInterface.
    # captures_llm_thoughts must be present: send_llm_thoughts() reads it whenever
    # gpt_settings.show_llm_thoughts is False (the pydantic-settings default, i.e. CI without .env).
    return cast(
        UserInterface,
        SimpleNamespace(
            storage_id=12345,
            thread_id=0,
            uses_uploaded_file_storage=False,
            captures_llm_thoughts=False,
            send_llm_thoughts=AsyncMock(),
        ),
    )


@pytest.mark.asyncio
async def test_get_chat_response_passes_conversation_messages_to_prepare_system_prompt(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """The provider must forward the real conversation to prepare_system_prompt."""
    response = _create_final_answer_response(answer="Hello")
    user = User(id=12345)
    conversation = [Message(role="user", content="Hi")]
    conversation_before = list(conversation)

    tracker = _PrepareSystemPromptTracker()

    with (
        _patch_client([response]),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=tracker,
        ),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch.object(provider, "call_functions", new=AsyncMock()),
    ):
        chat_response, new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert isinstance(chat_response, ChatResponseSchema)
    assert chat_response.answer == "Hello"
    assert len(tracker.calls) == 1
    # The loop mirrors the final assistant message into the conversation list
    # (it is the persisted form of the turn), so compare against the
    # pre-call snapshot of that list.
    assert tracker.calls[0][0]["conversation_messages"] == conversation_before
    prompt = json.loads(tracker.calls[0][1])
    # The volatile keys no longer travel in the system payload; the context
    # size reaches the model via the note on the last converted user message.
    assert "approximate_context_size" not in prompt


@pytest.mark.asyncio
async def test_tool_call_recursion_grows_context_size(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """During a tool-call loop the context size reported to the LLM must include tool traffic."""
    tool_response = _create_tool_call_response(tool_name="test_tool")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = [Message(role="user", content="Use the tool")]

    tracker = _PrepareSystemPromptTracker()

    with (
        _patch_client([tool_response, final_response]),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=tracker,
        ),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    assert len(tracker.calls) == 2
    first_context = sum(msg.estimate_tokens for msg in tracker.calls[0][0]["conversation_messages"])
    second_context = sum(msg.estimate_tokens for msg in tracker.calls[1][0]["conversation_messages"])
    assert second_context > first_context
    assert len(tracker.calls[1][0]["conversation_messages"]) == 3


@pytest.mark.asyncio
async def test_openai_chat_retry_never_retries_context_overflow(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """An oversized request fails fast: the retry policy gives up after a single attempt."""
    overflow_error = BadRequestError(
        "context_length_exceeded",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.moonshot.ai/v1/chat/completions"),
            json={"error": {"message": "test", "type": "invalid_request_error", "code": "context_length_exceeded"}},
        ),
        body={"message": "test", "type": "invalid_request_error", "code": "context_length_exceeded"},
    )

    with (
        # The mocked client raises the raw SDK error instead of a ChatCompletion;
        # the provider error-conversion wrapper is what turns it into the typed error.
        _patch_client(cast(list[ChatCompletion], [overflow_error])),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="{}")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=User(id=12345))),
        pytest.raises(ContextLengthExceededError),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=User(id=12345),
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )


@pytest.mark.asyncio
async def test_chat_retry_predicate_still_retries_transient_failures() -> None:
    """Genuine transient failures keep being retried up to the stop condition."""
    attempts = 0

    async def fail_transient() -> None:
        nonlocal attempts
        attempts += 1
        raise ServiceResponseError(provider="openai", model=TEST_MODEL)

    retrying = AsyncRetrying(stop=stop_after_attempt(4), retry=CHAT_RETRY_PREDICATE, reraise=True)
    with pytest.raises(ServiceResponseError):
        await retrying(fail_transient)

    assert attempts == 4


@pytest.mark.asyncio
async def test_chat_retry_predicate_never_retries_context_overflow() -> None:
    """The predicate rejects a context overflow, so tenacity re-raises it after one attempt."""
    attempts = 0

    async def fail_with_overflow() -> None:
        nonlocal attempts
        attempts += 1
        raise ContextLengthExceededError(provider="openai", model=TEST_MODEL)

    retrying = AsyncRetrying(stop=stop_after_attempt(4), retry=CHAT_RETRY_PREDICATE, reraise=True)
    with pytest.raises(ContextLengthExceededError):
        await retrying(fail_with_overflow)

    assert attempts == 1


@pytest.mark.asyncio
async def test_chat_retry_predicate_never_retries_cancellation() -> None:
    """Cancellation must not be retried: it is a BaseException, outside the Exception gate."""
    attempts = 0

    async def fail_with_cancellation() -> None:
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError()

    retrying = AsyncRetrying(stop=stop_after_attempt(4), retry=CHAT_RETRY_PREDICATE, reraise=True)
    with pytest.raises(asyncio.CancelledError):
        await retrying(fail_with_cancellation)

    assert attempts == 1


def test_chat_retry_predicate_is_attached_to_openai_chat_path() -> None:
    """The production retry policy on the OpenAI-compatible chat path uses the overflow-aware predicate."""
    # tenacity attaches the AsyncRetrying object and its predicate as function attributes.
    retrying_object = getattr(OpenAIFriendlyProvider.get_chat_response, "retry")
    assert retrying_object.retry is CHAT_RETRY_PREDICATE


def _make_tool_loop_client(responses: list[ChatCompletion]) -> tuple[Any, AsyncMock]:
    """Build a mocked OpenAI-compatible client returning canned completions.

    Args:
        responses: ChatCompletion objects returned sequentially by the mocked client.

    Returns:
        A tuple of the client mock and the captured ``create`` AsyncMock.
    """
    create_mock = AsyncMock(side_effect=responses)
    mock_client: Any = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    return mock_client, create_mock


def _in_loop_conversation() -> list[Message]:
    """Build a short history the in-loop guard has droppable units in.

    The list is the thread history a turn starts from: two dialogue turns and
    a final user question. The tool loop appends its own pair on top of it.
    """
    return [
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
        Message(role="user", content="Second question"),
    ]


@pytest.mark.asyncio
async def test_in_loop_request_over_budget_drops_oldest_units(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """An over-budget in-loop request is reduced: oldest units go, head and fresh pair stay."""
    tool_response = _create_tool_call_response(tool_name="test_tool")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = _in_loop_conversation()
    mock_client, create_mock = _make_tool_loop_client([tool_response, final_response])

    measure_mock = AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED))
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=AsyncMock(return_value="{}"),
        ),
        patch("chibi.services.user.measure_request_size_with_provenance", new=measure_mock),
        patch("chibi.services.user.get_model_context_window", new=Mock(return_value=1000)),
        patch.object(provider, "_get_max_tokens_value", new=Mock(return_value=100)),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    assert measure_mock.await_count == 1
    assert create_mock.await_count == 2
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    # system + head user message + the fresh tool pair; the two droppable
    # dialogue units between them are gone.
    assert len(second_messages) == 4
    assert second_messages[0]["role"] == "system"
    assert second_messages[1]["role"] == "user"
    assert second_messages[1]["content"] == "First question"
    assert "First answer" not in json.dumps(second_messages)
    assert "Second question" not in json.dumps(second_messages)
    assistant_item = second_messages[2]
    tool_item = second_messages[3]
    assert assistant_item["role"] == "assistant" and assistant_item["tool_calls"]
    assert tool_item["role"] == "tool"
    assert tool_item["tool_call_id"] == assistant_item["tool_calls"][0]["id"]


@pytest.mark.asyncio
async def test_in_loop_request_within_budget_is_untouched(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """An in-loop request that fits keeps the whole list, tool traffic included."""
    tool_response = _create_tool_call_response(tool_name="test_tool")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = _in_loop_conversation()
    mock_client, create_mock = _make_tool_loop_client([tool_response, final_response])

    measure_mock = AsyncMock(return_value=(10, SizeProvenance.ESTIMATED))
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=AsyncMock(return_value="{}"),
        ),
        patch("chibi.services.user.measure_request_size_with_provenance", new=measure_mock),
        patch("chibi.services.user.get_model_context_window", new=Mock(return_value=1000)),
        patch.object(provider, "_get_max_tokens_value", new=Mock(return_value=100)),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    assert len(second_messages) == 6
    assert "First answer" in json.dumps(second_messages)
    assert "Second question" in json.dumps(second_messages)


@pytest.mark.asyncio
async def test_in_loop_check_skipped_when_window_unknown(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """No resolvable window means nothing is measured and nothing is dropped in-loop."""
    tool_response = _create_tool_call_response(tool_name="test_tool")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = _in_loop_conversation()
    mock_client, create_mock = _make_tool_loop_client([tool_response, final_response])

    measure_mock = AsyncMock(return_value=(999999, SizeProvenance.ESTIMATED))
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=AsyncMock(return_value="{}"),
        ),
        patch("chibi.services.user.measure_request_size_with_provenance", new=measure_mock),
        patch("chibi.services.user.get_model_context_window", new=Mock(return_value=None)),
        patch.object(provider, "_get_max_tokens_value", new=Mock(return_value=100)),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    measure_mock.assert_not_awaited()
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    assert len(second_messages) == 6


@pytest.mark.asyncio
async def test_in_loop_check_runs_before_every_in_loop_request(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """Two tool exchanges mean two in-loop requests and two budget checks."""
    tool_response = _create_tool_call_response(tool_name="test_tool")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = _in_loop_conversation()
    mock_client, create_mock = _make_tool_loop_client([tool_response, tool_response, final_response])

    async def passthrough_mask(**kwargs: Any) -> list[bool]:
        seen_lengths.append(len(kwargs["conversation_messages"]))
        return [True] * len(kwargs["conversation_messages"])

    seen_lengths: list[int] = []
    check_mock = AsyncMock(side_effect=passthrough_mask)
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=AsyncMock(return_value="{}"),
        ),
        patch("chibi.services.providers.provider.enforce_in_loop_budget", new=check_mock),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    assert check_mock.await_count == 2
    assert create_mock.await_count == 3
    # The mask lengths track the growing in-flight list (3 history + 2 + 2),
    # snapshotted at call time: both checks receive the same growing list object.
    assert seen_lengths == [5, 7]


def _assert_openai_pair_integrity(items: list[dict[str, Any]]) -> None:
    """Assert Chat Completions pair integrity: no orphan tool result, no result-less tool_calls.

    Every ``role: "tool"`` item must follow the assistant item carrying the
    matching ``tool_calls`` entry, and every assistant ``tool_calls`` entry
    must have its result somewhere after it.
    """
    pending: set[str] = set()
    for item in items:
        if item.get("role") == "assistant" and item.get("tool_calls"):
            pending.update(call["id"] for call in item["tool_calls"])
        elif item.get("role") == "tool":
            assert item.get("tool_call_id") in pending, (
                f"orphan tool result {item.get('tool_call_id')!r}: its assistant tool_calls "
                "carrier is not part of the request"
            )
            pending.discard(item["tool_call_id"])
    assert not pending, f"assistant tool_calls entries without results: {pending}"


@pytest.mark.asyncio
async def test_in_loop_second_drop_stays_aligned_with_request(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """A second in-loop drop in one turn masks the list that is actually sent.

    Regression: the keep-mask used to be computed over the never-filtered
    ``conversation_messages`` mirror and applied positionally to the request
    list, which the previous drop had already shortened, from the second
    drop onward the mask addressed the wrong items, and a request could lose
    its fresh tool pair or carry an orphan ``role: "tool"`` result. The loop
    now filters the mirror with the same mask and passes it down, so the
    mirror and the request list stay in lockstep.
    """
    tool_response_1 = _create_tool_call_response(tool_name="test_tool", tool_call_id="call_1")
    tool_response_2 = _create_tool_call_response(tool_name="test_tool", tool_call_id="call_2")
    final_response = _create_final_answer_response(answer="Done")
    user = User(id=12345)
    conversation = _in_loop_conversation()
    mock_client, create_mock = _make_tool_loop_client([tool_response_1, tool_response_2, final_response])

    measure_mock = AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED))
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch(
            "chibi.services.providers.provider.prepare_system_prompt",
            new=AsyncMock(return_value="{}"),
        ),
        patch("chibi.services.user.measure_request_size_with_provenance", new=measure_mock),
        patch("chibi.services.user.get_model_context_window", new=Mock(return_value=1000)),
        patch.object(provider, "_get_max_tokens_value", new=Mock(return_value=100)),
        patch.object(
            provider,
            "call_functions",
            new=AsyncMock(return_value=[ToolResponseSchema(tool_name="test_tool", status="ok", result={})]),
        ),
    ):
        chat_response, _new_messages = await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert chat_response.answer == "Done"
    assert create_mock.await_count == 3
    assert measure_mock.await_count == 2

    # First in-loop request (after drop 1): head + the fresh first pair.
    second_request = create_mock.await_args_list[1].kwargs["messages"]
    assert [item["role"] for item in second_request] == ["system", "user", "assistant", "tool"]
    _assert_openai_pair_integrity(second_request)
    assert second_request[2]["tool_calls"][0]["id"] == "call_1"
    assert second_request[3]["tool_call_id"] == "call_1"
    assert "First question" in json.dumps(second_request)

    # Second in-loop request (after drop 2): the mask must address THIS list,
    # not the stale mirror. The dropped first pair is gone whole, the fresh
    # pair of the last exchange is intact.
    third_request = create_mock.await_args_list[2].kwargs["messages"]
    assert [item["role"] for item in third_request] == ["system", "user", "assistant", "tool"]
    _assert_openai_pair_integrity(third_request)
    assert third_request[1]["content"] == "First question"
    assert third_request[2]["tool_calls"][0]["id"] == "call_2"
    assert third_request[3]["tool_call_id"] == "call_2"
    assert "call_1" not in json.dumps(third_request)
    assert "First answer" not in json.dumps(third_request)
    assert "Second question" not in json.dumps(third_request)


# --- volatile context note wiring (task 11: block relocation) -------------------


def _note() -> str:
    from chibi.services.context_note import format_context_note

    return format_context_note(["approximate_context_size: 42 tokens (0% of 200,000 limit)"])


def _seed_context_store(user_id: int, thread_id: int) -> None:
    """Pin a deterministic real context size so the note text is predictable."""
    store = UsageCacheStore()
    store._data.clear()
    store._data[store._make_key(user_id=user_id, thread_id=thread_id)] = 42


def _count_notes_in_payload(payload: Any) -> int:
    from chibi.services.context_note import CONTEXT_NOTE_OPEN

    return json.dumps(payload, default=str).count(CONTEXT_NOTE_OPEN)


@pytest.mark.asyncio
async def test_note_is_attached_to_last_converted_user_message(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """The volatile note travels on the last converted user message of the request.

    The Message objects the caller owns are never mutated, so the persisted
    history and the next turn stay clean.
    """
    response = _create_final_answer_response(answer="Hello")
    user = User(id=12345)
    conversation = [Message(role="user", content="Hi"), Message(role="assistant", content="Yo")]
    snapshot = [msg.content for msg in conversation]
    note = _note()
    _seed_context_store(user_id=user.id, thread_id=interface.thread_id)

    mock_client, create_mock = _make_tool_loop_client([response])
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 200000),
    ):
        await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    # The recorded list keeps growing after the call (the final assistant turn
    # is appended to the same object), so assert on the request-time items.
    request_messages = create_mock.await_args_list[0].kwargs["messages"]
    # system + converted history; the note is on the last (user) message only.
    assert request_messages[0]["role"] == "system"
    assert request_messages[1]["role"] == "user"
    assert request_messages[2]["role"] == "assistant"
    assert request_messages[1]["content"].endswith(note)
    assert _count_notes_in_payload(request_messages) == 1
    assert request_messages[0]["content"] == "sp"
    # The caller's Message objects are untouched: no note anywhere in them.
    # The merged provider loop appends the normalized final assistant turn to
    # the same list (provider.py appends Message.from_openai(final_message)),
    # so the post-call list is the snapshot plus that one appended message.
    assert [msg.content for msg in conversation] == [*snapshot, "Hello"]
    assert [msg.role for msg in conversation[len(snapshot) :]] == ["assistant"]
    assert _count_notes_in_payload(conversation) == 0


@pytest.mark.asyncio
async def test_note_stays_single_across_a_tenacity_retry(
    provider: MoonshotAI,
    _mocked_metrics: MagicMock,
    interface: UserInterface,
) -> None:
    """A retried attempt re-converts clean Message objects: exactly one note per request."""
    response = _create_final_answer_response(answer="Hello")
    user = User(id=12345)
    conversation = [Message(role="user", content="Hi")]
    note = _note()
    _seed_context_store(user_id=user.id, thread_id=interface.thread_id)

    mock_client, create_mock = _make_tool_loop_client([ValueError("transient"), response])  # type: ignore[list-item]
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        # The production wait strategy would sleep 30s+ between attempts.
        patch.object(getattr(OpenAIFriendlyProvider.get_chat_response, "retry"), "wait", wait_none()),
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 200000),
    ):
        await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert create_mock.await_count == 2
    for attempt in create_mock.await_args_list:
        request_messages = attempt.kwargs["messages"]
        assert _count_notes_in_payload(request_messages) == 1
        user_items = [item for item in request_messages if item["role"] == "user"]
        assert len(user_items) == 1
        assert user_items[0]["content"].endswith(note)


@pytest.mark.asyncio
async def test_shared_message_list_through_two_attach_sites_keeps_single_note(
    interface: UserInterface,
    _mocked_metrics: MagicMock,
) -> None:
    """The same Message list passing two attach sites yields exactly one note per payload.

    This is the double-attach case the OpenAI Responses -> Chat Completions
    fallback creates structurally: both paths convert and attach independently,
    and each conversion starts from Message objects that no path ever mutated,
    so no payload can ever carry a second note.
    """
    from chibi.services.context_note import CONTEXT_NOTE_OPEN
    from chibi.services.providers.openai import OpenAI as ChibiOpenAI

    note = _note()
    provider = ChibiOpenAI(token=TEST_TOKEN)
    user = User(id=12345)
    conversation = [Message(role="user", content="Hi")]
    _seed_context_store(user_id=user.id, thread_id=interface.thread_id)
    responses_ok = SimpleNamespace(output=[SimpleNamespace(type="message")], output_text="Done", usage=None)
    responses_mock = SimpleNamespace(create=AsyncMock(return_value=responses_ok))
    chat_client, create_mock = _make_tool_loop_client([_create_final_answer_response(answer="Hello")])
    mock_client: Any = SimpleNamespace(
        responses=responses_mock,
        chat=chat_client.chat,
    )

    with (
        patch.object(ChibiOpenAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="sp")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.config.gpt.gpt_settings.max_history_tokens", 200000),
    ):
        # Site 1: the Responses path (its own converted payload).
        await provider._get_response_completion_response(
            messages=conversation,
            model=TEST_MODEL,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            system_prompt=None,
            interface=interface,
        )
        # The Message list the fallback would re-convert is clean: the note
        # never touched the Message objects.
        assert _count_notes_in_payload(conversation) == 0

        # Site 2: the generic Chat Completions path over the same list.
        await getattr(OpenAIFriendlyProvider.get_chat_response, "__wrapped__")(
            provider,
            messages=conversation,
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    responses_input = responses_mock.create.await_args_list[0].kwargs["input"]
    assert CONTEXT_NOTE_OPEN in json.dumps(responses_input, default=str)
    fallback_request = create_mock.await_args_list[0].kwargs["messages"]
    assert _count_notes_in_payload(fallback_request) == 1
    fallback_user = [item for item in fallback_request if item["role"] == "user"]
    assert len(fallback_user) == 1
    assert fallback_user[0]["content"].endswith(note)
    assert _count_notes_in_payload(conversation) == 0
