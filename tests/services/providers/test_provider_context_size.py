"""Tests for provider-to-utils context-size wiring."""

import json
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.tools.schemas import ToolResponseSchema
from chibi.services.providers.utils import prepare_system_prompt


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


def _create_tool_call_response(tool_name: str, arguments: str = "{}") -> ChatCompletion:
    """Build a ChatCompletion containing a single tool call.

    Args:
        tool_name: Name of the tool the assistant requests.
        arguments: JSON-encoded arguments for the tool call.

    Returns:
        A ChatCompletion object whose assistant message carries one tool call.
    """
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
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
    return cast(
        UserInterface,
        SimpleNamespace(
            storage_id=12345,
            thread_id=0,
            uses_uploaded_file_storage=False,
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
    assert tracker.calls[0][0]["conversation_messages"] == conversation
    prompt = json.loads(tracker.calls[0][1])
    assert prompt["approximate_context_size"] == "n/a"


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
