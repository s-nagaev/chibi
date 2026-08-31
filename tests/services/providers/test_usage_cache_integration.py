"""Integration tests for UsageCacheStore writes at chat-completion sites."""

from types import SimpleNamespace
from typing import Iterator, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.config import application_settings as real_application_settings
from chibi.models import Message, User
from chibi.services.interface import UserInterface
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI
from chibi.services.usage_cache import UsageCacheStore

TEST_TOKEN = "test-token-for-usage-cache"
TEST_MODEL = "kimi-k2.6"


def _create_final_answer_response(answer: str, prompt_tokens: int = 50) -> ChatCompletion:
    """Build a ChatCompletion containing a final answer without tool calls.

    Args:
        answer: Assistant message content.
        prompt_tokens: Prompt token count to report in usage.

    Returns:
        A ChatCompletion object with a single choice and usage data.
    """
    mock_message = ChatCompletionMessage(role="assistant", content=answer, tool_calls=None)
    mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=5,
            total_tokens=prompt_tokens + 5,
        ),
    )


def _patch_provider_influx_not_configured():
    """Patch provider module application_settings so InfluxDB appears unconfigured."""
    mock_settings = MagicMock(wraps=real_application_settings)
    mock_settings.is_influx_configured = False
    return patch("chibi.services.providers.provider.application_settings", mock_settings)


def _patch_openai_influx_not_configured():
    """Patch openai module application_settings so InfluxDB appears unconfigured."""
    mock_settings = MagicMock(wraps=real_application_settings)
    mock_settings.is_influx_configured = False
    return patch("chibi.services.providers.openai.application_settings", mock_settings)


@pytest.fixture
def usage_cache_store() -> Iterator[UsageCacheStore]:
    """Provide the usage cache store cleared for test isolation."""
    store = UsageCacheStore()
    store._data.clear()
    yield store
    store._data.clear()


@pytest.fixture
def interface() -> UserInterface:
    """Return a minimal UserInterface stub."""
    return cast(
        UserInterface,
        SimpleNamespace(
            storage_id=12345,
            thread_id=3,
            uses_uploaded_file_storage=False,
            send_llm_thoughts=AsyncMock(),
        ),
    )


@pytest.mark.asyncio
async def test_openai_friendly_provider_writes_cache_when_influx_not_configured(
    usage_cache_store: UsageCacheStore,
    interface: UserInterface,
) -> None:
    """The OpenAI-friendly chat path must populate the store even without InfluxDB."""
    provider = MoonshotAI(token=TEST_TOKEN)
    response = _create_final_answer_response(answer="Hello", prompt_tokens=150)
    user = User(id=12345)

    completions = SimpleNamespace(create=AsyncMock(return_value=response))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
            track_prompt_size=True,
        )

    assert usage_cache_store.get(user_id=user.id, thread_id=interface.thread_id) == 150


@pytest.mark.asyncio
async def test_openai_responses_api_writes_cache_when_influx_not_configured(
    usage_cache_store: UsageCacheStore,
    interface: UserInterface,
) -> None:
    """The OpenAI Responses API chat path must populate the store even without InfluxDB."""
    provider = OpenAI(token=TEST_TOKEN)
    user = User(id=999)

    mock_response = MagicMock()
    mock_response.output = [MagicMock(type="message", content=[MagicMock(text="Hi there")])]
    mock_response.output_text = "Hi there"
    mock_response.usage = MagicMock(input_tokens=250, output_tokens=10, total_tokens=260)

    responses = MagicMock()
    responses.create = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.responses = responses

    with (
        _patch_openai_influx_not_configured(),
        patch.object(OpenAI, "get_client", return_value=mock_client),
        patch("chibi.services.providers.openai.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider._get_response_completion_response(
            messages=[Message(role="user", content="Hi")],
            model="gpt-5.2",
            user=user,
            system_prompt="base",
            interface=interface,
            track_prompt_size=True,
        )

    assert usage_cache_store.get(user_id=user.id, thread_id=interface.thread_id) == 250


@pytest.mark.asyncio
async def test_sub_agent_call_does_not_overwrite_parent_cache(
    usage_cache_store: UsageCacheStore,
    interface: UserInterface,
) -> None:
    """A sub-agent-style call must not overwrite the parent conversation's cached value.

    ``get_sub_agent_response`` (``chibi/services/providers/tools/utils.py``)
    calls ``provider.get_chat_response`` with a real ``user_id`` and no
    ``interface``, and—critically—does NOT pass ``track_prompt_size=True``.
    Because the store write is now gated on that explicit opt-in flag, the
    sub-agent path is silently excluded and the parent conversation's cached
    prompt size is preserved.

    This test simulates that exact call shape (no ``interface``, no
    ``track_prompt_size``) against the same provider used for the parent
    turn, and asserts the store is untouched afterwards.
    """
    provider = MoonshotAI(token=TEST_TOKEN)
    user = User(id=12345)

    parent_response = _create_final_answer_response(answer="Hello from parent", prompt_tokens=150)
    sub_agent_response = _create_final_answer_response(answer="sub-agent answer", prompt_tokens=5)

    completions = SimpleNamespace(create=AsyncMock(side_effect=[parent_response, sub_agent_response]))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
            track_prompt_size=True,
        )

    parent_thread = interface.thread_id
    assert usage_cache_store.get(user_id=user.id, thread_id=parent_thread) == 150

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="sub-agent prompt")],
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
        )

    assert usage_cache_store.get(user_id=user.id, thread_id=parent_thread) == 150
    assert usage_cache_store.get(user_id=user.id, thread_id=0) is None


@pytest.mark.asyncio
async def test_call_without_opt_in_does_not_write(
    usage_cache_store: UsageCacheStore,
    interface: UserInterface,
) -> None:
    """A chat call that does not opt into tracking must not populate the store.

    This guards the fail-safe default: ``track_prompt_size`` defaults to
    ``False``, so any future call path that forgets to opt in is silently
    excluded rather than silently corrupting the store.
    """
    provider = MoonshotAI(token=TEST_TOKEN)
    user = User(id=4242)
    response = _create_final_answer_response(answer="Hello", prompt_tokens=77)
    completions = SimpleNamespace(create=AsyncMock(return_value=response))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    assert usage_cache_store.get(user_id=user.id, thread_id=interface.thread_id) is None


@pytest.mark.asyncio
async def test_moderation_does_not_overwrite_chat_cache(
    usage_cache_store: UsageCacheStore,
    interface: UserInterface,
) -> None:
    """A moderation call must not overwrite the conversation's cached prompt size."""
    provider = MoonshotAI(token=TEST_TOKEN)
    user = User(id=12345)

    chat_response = _create_final_answer_response(answer="Hello", prompt_tokens=150)
    completions = SimpleNamespace(create=AsyncMock(side_effect=[chat_response]))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="system")),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
    ):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=user,
            caller_storage_id=interface.storage_id,
            caller_thread_id=interface.thread_id,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
            track_prompt_size=True,
        )

    assert usage_cache_store.get(user_id=user.id, thread_id=interface.thread_id) == 150

    moderator_response = _create_final_answer_response(
        answer='{"verdict": "accepted", "status": "ok"}',
        prompt_tokens=5,
    )
    moderator_completions = SimpleNamespace(create=AsyncMock(return_value=moderator_response))
    mock_client_moderator = SimpleNamespace(chat=SimpleNamespace(completions=moderator_completions))

    with (
        _patch_provider_influx_not_configured(),
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client_moderator),
    ):
        await provider.moderate_command(cmd="hello")

    assert usage_cache_store.get(user_id=user.id, thread_id=interface.thread_id) == 150
