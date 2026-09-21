"""Tests for the unified MITM-aware connection retry policy.

Covers:
- tenacity retry config on all four provider families (4 attempts, exponential waits);
- retry-then-success behavior for the anthropic, gemini native and mistral native families;
- the human-readable final message produced by ``handle_gpt_exceptions`` after
  connection-type failures (retries exhausted).

No real sleeps happen: tenacity's ``sleep`` hook is monkeypatched in every retry test.
"""

from typing import Any, Callable
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import APIConnectionError

from chibi.exceptions import ServiceConnectionError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.provider import (
    AnthropicFriendlyProvider,
    OpenAIFriendlyProvider,
)
from chibi.utils.app import handle_gpt_exceptions
from tests.services.test_reactive_context_recovery import _FakeInterface, _make_chat_response


class _FakeIDEInterface(_FakeInterface):
    """Fake interface that also satisfies the IDEErrorInterface protocol."""

    error_code: str | None = None
    error_message: str | None = None


def _retry_policy(provider_cls: Any) -> Any:
    """Return the tenacity policy object attached by ``@retry`` to the provider's ``get_chat_response``."""
    return provider_cls.get_chat_response.retry


def _anthropic_connection_error() -> Exception:
    return APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _httpx_read_error() -> Exception:
    return httpx.ReadError("connection reset by peer", request=httpx.Request("POST", "https://example.com"))


def _test_user() -> User:
    return User(id=42)


def _test_messages() -> list[Message]:
    return [Message(role="user", content="hi")]


@pytest.mark.parametrize(
    ("provider_cls", "flaky_error_factory"),
    [
        (Anthropic, _anthropic_connection_error),
        (Gemini, _httpx_read_error),
        (MistralAI, ConnectionResetError),
    ],
)
@pytest.mark.asyncio
async def test_connection_error_is_retried_then_succeeds(
    provider_cls: Any, flaky_error_factory: Callable[[], Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """First 3 attempts raise a connection-type error, the 4th succeeds."""
    provider = provider_cls(token="test-token")
    retry_policy = _retry_policy(provider_cls)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(retry_policy, "sleep", sleep_mock)

    attempts = {"count": 0}

    async def flaky(*args: Any, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise flaky_error_factory()
        return _make_chat_response("recovered"), []

    monkeypatch.setattr(provider, "_get_chat_completion_response", flaky)

    chat_response, updated_messages = await provider.get_chat_response(
        messages=_test_messages(), user=_test_user(), caller_storage_id=1, caller_thread_id=1
    )

    assert attempts["count"] == 4
    assert sleep_mock.await_count == 3  # no real sleeps happened
    assert chat_response.answer == "recovered"
    assert updated_messages == []


@pytest.mark.asyncio
async def test_connection_error_retries_do_not_sleep_for_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: waits are exponential 30-180s, so tests must never hit real sleep."""
    provider = Gemini(token="test-token")
    sleep_mock = AsyncMock()
    monkeypatch.setattr(_retry_policy(Gemini), "sleep", sleep_mock)

    async def always_fails(*args: Any, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(provider, "_get_chat_completion_response", always_fails)

    with pytest.raises(ConnectionResetError):  # reraise=True preserves the original exception
        await provider.get_chat_response(
            messages=_test_messages(), user=_test_user(), caller_storage_id=1, caller_thread_id=1
        )

    assert sleep_mock.await_count == 3


def test_retry_policies_are_exponential_with_four_attempts() -> None:
    """All four families share the same MITM-aware policy: 4 attempts, ~30s -> ~180s waits."""
    expected = {"multiplier": 20, "min": 30, "max": 180}
    for provider_cls in (AnthropicFriendlyProvider, Gemini, MistralAI, OpenAIFriendlyProvider):
        policy = _retry_policy(provider_cls)
        assert policy.stop.max_attempt_number == 4, provider_cls.__name__
        assert policy.wait.multiplier == expected["multiplier"], provider_cls.__name__
        assert policy.wait.min == expected["min"], provider_cls.__name__
        assert policy.wait.max == expected["max"], provider_cls.__name__


def test_retry_predicates_match_connection_shaped_errors() -> None:
    """Anthropic family also retries SDK-level APIConnectionError; native families retry httpx transport errors."""
    anthropic_exceptions = _retry_policy(AnthropicFriendlyProvider).retry.exception_types
    assert ConnectionError in anthropic_exceptions
    assert APIConnectionError in anthropic_exceptions

    for provider_cls in (Gemini, MistralAI):
        exceptions = _retry_policy(provider_cls).retry.exception_types
        assert ConnectionError in exceptions, provider_cls.__name__
        assert httpx.TransportError in exceptions, provider_cls.__name__


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: ConnectionResetError("connection reset by peer"),
        lambda: ServiceConnectionError(provider="OpenAI", model="gpt-5"),
        lambda: _httpx_read_error(),
        lambda: _anthropic_connection_error(),
    ],
)
async def test_exhausted_connection_failures_produce_human_readable_message(
    exc_factory: Callable[[], Exception],
) -> None:
    """After retries are exhausted the user gets a clear connection message, not the generic hiccup."""
    raised = exc_factory()

    @handle_gpt_exceptions
    async def handler(interface: UserInterface) -> ChatResponseSchema:
        raise raised

    interface = _FakeIDEInterface()
    result = await handler(interface=interface)

    assert result is None
    assert len(interface.sent_messages) == 1
    message = interface.sent_messages[0].lower()
    assert "connection" in message
    assert "interrupted" in message
    assert "hiccup" not in message
    assert interface.error_code == "provider_error"
    assert interface.error_message
