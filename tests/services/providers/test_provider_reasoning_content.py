"""Unit tests for reasoning_content surfacing in OpenAIFriendlyProvider."""

from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.models import User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.providers.moonshotai import MoonshotAI

TEST_TOKEN = "test-token-for-reasoning"
TEST_MODEL = "kimi-k2.6"


def create_final_answer_response(answer: str, reasoning_content: str | None = None) -> ChatCompletion:
    """Build a ChatCompletion containing a final answer without tool calls.

    Args:
        answer: Assistant message content.
        reasoning_content: Optional reasoning text as returned by thinking models.

    Returns:
        A ChatCompletion object with a single choice and no tool calls.
    """
    message_kwargs: dict[str, Any] = {"role": "assistant", "content": answer, "tool_calls": None}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content

    mock_choice = Choice(
        index=0,
        message=ChatCompletionMessage(**message_kwargs),
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


@pytest.fixture
def provider() -> MoonshotAI:
    """Create a MoonshotAI provider instance with a dummy token."""
    return MoonshotAI(token=TEST_TOKEN)


@pytest.fixture
def _mocked_metrics() -> Iterator[MagicMock]:
    """Patch MetricsService to keep usage metrics out of the test run."""
    with patch("chibi.services.providers.provider.MetricsService") as metrics_mock:
        yield metrics_mock


def patch_client(response: ChatCompletion) -> Any:
    """Patch the provider client property so completions return a canned response.

    A plain namespace is used instead of a MagicMock because
    ``OpenAIFriendlyProvider.__getattribute__`` wraps every callable attribute,
    which would replace a callable mock client with a wrapper function.

    Args:
        response: The ChatCompletion object the mocked client must return.

    Returns:
        A context manager patching the ``client`` property of MoonshotAI.
    """
    completions = SimpleNamespace(create=AsyncMock(return_value=response))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client)


@pytest.mark.asyncio
async def test_final_answer_surfaces_reasoning_content(provider: MoonshotAI, _mocked_metrics: MagicMock) -> None:
    """Reasoning content must be sent as thoughts when the response has no tool calls."""
    response = create_final_answer_response(answer="42", reasoning_content="Let me think about the answer.")
    interface = MagicMock()

    with (
        patch_client(response),
        patch("chibi.services.providers.provider.send_llm_thoughts", new_callable=AsyncMock) as send_thoughts_mock,
    ):
        chat_response, messages = await provider._get_chat_completion_response(
            messages=[],
            model=TEST_MODEL,
            user=User(id=12345),
            system_prompt=None,
            interface=interface,
        )

    send_thoughts_mock.assert_awaited_once_with(
        thoughts="Let me think about the answer.",
        interface=interface,
    )
    assert isinstance(chat_response, ChatResponseSchema)
    assert chat_response.answer == "42"
    assert len(messages) == 1
    assert messages[0]["content"] == "42"


@pytest.mark.asyncio
async def test_final_answer_reaches_interface_thoughts(provider: MoonshotAI, _mocked_metrics: MagicMock) -> None:
    """The interface receives the reasoning text through the real send_llm_thoughts helper."""
    response = create_final_answer_response(answer="Done", reasoning_content="Step by step reasoning.")
    interface = MagicMock()
    interface.send_llm_thoughts = AsyncMock()

    with patch_client(response), patch("chibi.services.providers.utils.gpt_settings") as settings_mock:
        settings_mock.show_llm_thoughts = True
        await provider._get_chat_completion_response(
            messages=[],
            model=TEST_MODEL,
            user=User(id=12345),
            system_prompt=None,
            interface=interface,
        )

    interface.send_llm_thoughts.assert_awaited_once_with("Step by step reasoning.")


@pytest.mark.asyncio
async def test_final_answer_without_reasoning_sends_no_thoughts(
    provider: MoonshotAI, _mocked_metrics: MagicMock
) -> None:
    """No thoughts are sent when the final answer carries no reasoning content."""
    response = create_final_answer_response(answer="Plain answer")
    interface = MagicMock()

    with (
        patch_client(response),
        patch("chibi.services.providers.provider.send_llm_thoughts", new_callable=AsyncMock) as send_thoughts_mock,
    ):
        chat_response, _ = await provider._get_chat_completion_response(
            messages=[],
            model=TEST_MODEL,
            user=User(id=12345),
            system_prompt=None,
            interface=interface,
        )

    send_thoughts_mock.assert_not_awaited()
    assert chat_response.answer == "Plain answer"


@pytest.mark.asyncio
async def test_final_answer_with_empty_reasoning_sends_no_thoughts(
    provider: MoonshotAI, _mocked_metrics: MagicMock
) -> None:
    """An empty reasoning_content value must not trigger a thoughts message."""
    response = create_final_answer_response(answer="Plain answer", reasoning_content="")
    interface = MagicMock()

    with (
        patch_client(response),
        patch("chibi.services.providers.provider.send_llm_thoughts", new_callable=AsyncMock) as send_thoughts_mock,
    ):
        await provider._get_chat_completion_response(
            messages=[],
            model=TEST_MODEL,
            user=User(id=12345),
            system_prompt=None,
            interface=interface,
        )

    send_thoughts_mock.assert_not_awaited()
