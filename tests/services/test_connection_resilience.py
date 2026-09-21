"""Tests for the unified MITM-aware connection retry policy.

Covers:
- tenacity retry config on all four provider families (4 attempts, exponential waits);
- retry-then-success behavior for the anthropic, gemini native and mistral native families;
- rate-limit (429) and transient 5xx retry-then-success for the same three families;
- the gemini internal 429 loop NOT being double-retried by the tenacity decorator;
- the gemini 5xx re-raise being scoped to the chat path (undecorated methods keep
  ``ServiceResponseError`` so users get the specific provider-error message);
- the human-readable final message produced by ``handle_gpt_exceptions`` after
  connection-type failures (retries exhausted).

No real sleeps happen: tenacity's ``sleep`` hook is monkeypatched in every retry test.
"""

from typing import Any, Callable
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import APIConnectionError
from anthropic import InternalServerError as AnthropicInternalServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.genai.errors import ServerError as GeminiServerError
from google.genai.types import GenerateContentConfig
from mistralai.models import SDKError

from chibi.exceptions import ServiceConnectionError, ServiceRateLimitError, ServiceResponseError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI, _is_retryable_mistral_error
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


def _anthropic_status_error(error_cls: Any, status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return error_cls(message="transient error", response=httpx.Response(status_code, request=request), body=None)


def _gemini_server_error() -> GeminiServerError:
    return GeminiServerError(500, {"error": {"message": "internal error"}})


def _mistral_sdk_error(status_code: int) -> Exception:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return SDKError(message="transient error", raw_response=httpx.Response(status_code, request=request))


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
    """Anthropic family also retries SDK-level APIConnectionError; gemini retries transport errors."""
    anthropic_exceptions = _retry_policy(AnthropicFriendlyProvider).retry.exception_types
    assert ConnectionError in anthropic_exceptions
    assert APIConnectionError in anthropic_exceptions

    gemini_exceptions = _retry_policy(Gemini).retry.exception_types
    assert ConnectionError in gemini_exceptions
    assert httpx.TransportError in gemini_exceptions
    assert GeminiServerError in gemini_exceptions


def test_mistral_retry_predicate_matches_429_and_5xx_only() -> None:
    """Mistral SDK has no distinct rate-limit/5xx classes: the custom predicate inspects status codes."""
    assert _retry_policy(MistralAI).retry.predicate is _is_retryable_mistral_error
    assert _is_retryable_mistral_error(ConnectionResetError())
    assert _is_retryable_mistral_error(_httpx_read_error())
    assert _is_retryable_mistral_error(_mistral_sdk_error(429))
    assert _is_retryable_mistral_error(_mistral_sdk_error(502))
    assert not _is_retryable_mistral_error(_mistral_sdk_error(400))
    assert not _is_retryable_mistral_error(_mistral_sdk_error(403))
    assert not _is_retryable_mistral_error(ValueError("unrelated"))


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


@pytest.mark.parametrize(
    ("provider_cls", "error_factory"),
    [
        (Anthropic, lambda: _anthropic_status_error(AnthropicRateLimitError, 429)),
        (Anthropic, lambda: _anthropic_status_error(AnthropicInternalServerError, 500)),
        (Gemini, _gemini_server_error),
        (MistralAI, lambda: _mistral_sdk_error(429)),
        (MistralAI, lambda: _mistral_sdk_error(503)),
    ],
    ids=["anthropic-429", "anthropic-500", "gemini-500", "mistral-429", "mistral-503"],
)
@pytest.mark.asyncio
async def test_rate_limit_and_5xx_are_retried_then_succeed(
    provider_cls: Any, error_factory: Callable[[], Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate limits (429) and transient 5xx errors are retried by the decorator; success on a later attempt."""
    provider = provider_cls(token="test-token")
    sleep_mock = AsyncMock()
    monkeypatch.setattr(_retry_policy(provider_cls), "sleep", sleep_mock)

    attempts = {"count": 0}

    async def flaky(*args: Any, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise error_factory()
        return _make_chat_response("recovered"), []

    monkeypatch.setattr(provider, "_get_chat_completion_response", flaky)

    chat_response, updated_messages = await provider.get_chat_response(
        messages=_test_messages(), user=_test_user(), caller_storage_id=1, caller_thread_id=1
    )

    assert attempts["count"] == 3
    assert sleep_mock.await_count == 2  # no real sleeps happened
    assert chat_response.answer == "recovered"
    assert updated_messages == []


@pytest.mark.asyncio
async def test_gemini_429_is_not_double_retried_by_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The internal 429 loop (Retry-Info delays) raises ServiceRateLimitError once its retries are
    exhausted; the tenacity decorator must NOT retry it again (no double-retry)."""
    provider = Gemini(token="test-token")
    sleep_mock = AsyncMock()
    monkeypatch.setattr(_retry_policy(Gemini), "sleep", sleep_mock)

    calls = {"count": 0}

    async def internal_loop_exhausted(*args: Any, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        # Simulates gemini's internal 429 loop having exhausted ``gpt_settings.retries``.
        calls["count"] += 1
        raise ServiceRateLimitError(provider="Gemini", model="models/gemini-3.8-flash", detail="quota exceeded")

    monkeypatch.setattr(provider, "_get_chat_completion_response", internal_loop_exhausted)

    with pytest.raises(ServiceRateLimitError):  # reraise=True preserves the original exception
        await provider.get_chat_response(
            messages=_test_messages(), user=_test_user(), caller_storage_id=1, caller_thread_id=1
        )

    assert calls["count"] == 1  # the decorator did not add extra attempts on top of the internal loop
    assert sleep_mock.await_count == 0  # and did not sleep for it


class _FakeModels:
    async def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        raise _gemini_server_error()


class _FakeAio:
    def __init__(self) -> None:
        self.models = _FakeModels()

    async def __aenter__(self) -> "_FakeAio":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeGenaiClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.aio = _FakeAio()


@pytest.mark.asyncio
async def test_gemini_5xx_reraise_is_scoped_to_the_chat_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw-5xx re-raise only fires with ``retry_server_errors=True`` (the decorated chat path).

    Undecorated callers (moderation, speech, STT, vision, OCR) keep converting 5xx into
    ``ServiceResponseError`` so users still get the specific provider-error message via
    ``handle_gpt_exceptions`` instead of the raw SDK exception.
    """
    provider = Gemini(token="test-token")
    monkeypatch.setattr("chibi.services.providers.gemini_native.Client", _FakeGenaiClient)

    # Chat path: the raw ServerError escapes so the tenacity decorator can retry it.
    with pytest.raises(GeminiServerError):
        await provider._generate_content(
            model="models/gemini-3.8-flash",
            contents="hi",
            config=GenerateContentConfig(),
            retry_server_errors=True,
        )

    # Undecorated path (default): 5xx is converted into ServiceResponseError as before.
    with pytest.raises(ServiceResponseError):
        await provider._generate_content(
            model="models/gemini-3.8-flash",
            contents="hi",
            config=GenerateContentConfig(),
        )
