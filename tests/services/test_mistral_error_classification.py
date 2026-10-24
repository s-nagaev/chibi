"""Tests for the Mistral error classification shared by failover and the adapter.

The Mistral SDK (3.3.0) exposes no per-status exception classes, so a failed
HTTP response is an ``SDKError`` carrying the raw response and transport
failures are plain ``httpx2`` exceptions. These tests pin:

- the failover classifier: 429 -> ``rate_limit``, 5xx -> ``server_error``,
  401/403 -> not a failover trigger, httpx2 timeouts -> ``timeout``,
  other httpx2 transport errors -> ``network``;
- the Mistral adapter's retry predicate, including the httpx2 transport
  coverage the project ``httpx`` classes do not provide;
- the end-to-end regression: a rate-limited Mistral call escapes the
  in-provider tenacity retries, fails over to the next provider and puts
  Mistral on a cooldown.
"""

from unittest.mock import AsyncMock, Mock

import httpx
import httpx2
import pytest
from mistralai.client.errors import SDKError

from chibi.config.gpt import gpt_settings
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema
from chibi.services.failover import (
    CooldownStore,
    FailoverEngine,
    FailoverModelRegistry,
    classify_exception,
    reset_failover_policy_cache,
)
from chibi.services.mistral_errors import classify_mistral_exception
from chibi.services.providers.mistralai_native import MistralAI, _is_retryable_mistral_error
from chibi.storage.database import _db_provider
from chibi.utils.app import SingletonMeta


def _mistral_sdk_error(status_code: int) -> Exception:
    request = httpx2.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    return SDKError(message="transient error", raw_response=httpx2.Response(status_code, request=request))


class TestFailoverClassifier:
    def test_sdk_error_429_is_rate_limit(self):
        assert classify_exception(_mistral_sdk_error(429)) == "rate_limit"

    def test_sdk_error_500_is_server_error(self):
        assert classify_exception(_mistral_sdk_error(500)) == "server_error"

    def test_sdk_error_401_is_not_a_failover_trigger(self):
        assert classify_exception(_mistral_sdk_error(401)) is None

    def test_sdk_error_403_is_not_a_failover_trigger(self):
        assert classify_exception(_mistral_sdk_error(403)) is None

    def test_httpx2_timeout_is_timeout(self):
        assert classify_exception(httpx2.ReadTimeout("timed out while reading")) == "timeout"

    def test_httpx2_connect_error_is_network(self):
        assert classify_exception(httpx2.ConnectError("connection refused")) == "network"

    def test_unrelated_exception_stays_unclassified(self):
        assert classify_exception(ValueError("unrelated")) is None


class TestMistralErrorKindModule:
    def test_classifier_matches_sdk_status_codes(self):
        assert classify_mistral_exception(_mistral_sdk_error(429)) == "rate_limit"
        assert classify_mistral_exception(_mistral_sdk_error(503)) == "server_error"
        assert classify_mistral_exception(_mistral_sdk_error(401)) is None

    def test_classifier_maps_transport_errors(self):
        assert classify_mistral_exception(httpx2.ConnectTimeout("connect timed out")) == "timeout"
        assert classify_mistral_exception(httpx2.ConnectError("connection refused")) == "network"


class TestMistralRetryPredicate:
    def test_sdk_error_429_is_retryable(self):
        assert _is_retryable_mistral_error(_mistral_sdk_error(429)) is True

    def test_sdk_error_500_is_retryable(self):
        assert _is_retryable_mistral_error(_mistral_sdk_error(500)) is True

    def test_sdk_error_401_is_not_retryable(self):
        assert _is_retryable_mistral_error(_mistral_sdk_error(401)) is False

    def test_httpx2_connect_error_is_retryable(self):
        assert _is_retryable_mistral_error(httpx2.ConnectError("connection refused")) is True

    def test_httpx2_read_timeout_is_retryable(self):
        assert _is_retryable_mistral_error(httpx2.ReadTimeout("timed out while reading")) is True

    def test_project_httpx_transport_error_is_still_retryable(self):
        error = httpx.ConnectError("connection refused", request=httpx.Request("POST", "https://api.mistral.ai"))
        assert _is_retryable_mistral_error(error) is True

    def test_unrelated_error_is_not_retryable(self):
        assert _is_retryable_mistral_error(ValueError("unrelated")) is False


@pytest.fixture(autouse=True)
def _clean_policy_cache():
    reset_failover_policy_cache()
    yield
    reset_failover_policy_cache()


@pytest.fixture(autouse=True)
def _clean_singletons():
    for cls in (CooldownStore, FailoverModelRegistry):
        SingletonMeta._instances.pop(cls, None)
    yield
    for cls in (CooldownStore, FailoverModelRegistry):
        SingletonMeta._instances.pop(cls, None)


class TestMistralRateLimitFailover:
    async def test_rate_limited_mistral_fails_over_and_marks_cooldown(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
        # No real sleeps: the tenacity policy waits 30-180s between attempts.
        sleep_mock = AsyncMock()
        monkeypatch.setattr(MistralAI.get_chat_response.retry, "sleep", sleep_mock)

        mistral = MistralAI(token="stub-token")
        rate_limit_error = _mistral_sdk_error(429)

        async def _always_rate_limited(**kwargs):
            raise rate_limit_error

        monkeypatch.setattr(mistral, "_get_chat_completion_response", _always_rate_limited)

        gemini = AsyncMock()
        gemini.name = "gemini"
        gemini.get_available_models = AsyncMock(
            return_value=[type("ModelStub", (), {"name": "mistral-medium-latest"})()]
        )
        fallback_response = ChatResponseSchema(
            answer="fallback answer", provider="gemini", model="mistral-medium-latest", usage=None
        )
        gemini.get_chat_response = AsyncMock(
            return_value=(fallback_response, [Message(role="assistant", content="fallback answer")])
        )

        user = Mock()
        user.providers = Mock()
        user.providers.chat_ready = {"gemini": type("GeminiStub", (), {"name": "gemini"})}
        user.providers.get_instance = lambda provider_class: gemini

        db = AsyncMock()
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        engine = FailoverEngine(user=user)

        async def call(provider, model):
            return await provider.get_chat_response(
                messages=[], user=user, caller_storage_id=1, caller_thread_id=0, model=model
            )

        result = await engine.run(role="master", primary_provider=mistral, model="mistral-medium-latest", call=call)

        assert result.fallback_used is True
        assert result.serving_provider == "gemini"
        assert result.serving_model == "mistral-medium-latest"
        assert result.response is not None and result.response.answer == "fallback answer"
        assert result.attempts[0].ok is False
        assert result.attempts[0].kind == "rate_limit"
        assert result.attempts[0].provider == "MistralAI"
        assert result.attempts[1].ok is True and result.attempts[1].provider == "gemini"
        gemini.get_chat_response.assert_awaited_once()
        # The in-provider retry policy gave the rate-limited call its 4 attempts...
        assert sleep_mock.await_count == 3
        # ...and the engine still recorded the 20-minute cooldown for Mistral.
        assert await engine.cooldown.is_blocked("mistralai") is True

    async def test_rate_limited_mistral_is_a_trigger_while_auth_error_is_not(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
        sleep_mock = AsyncMock()
        monkeypatch.setattr(MistralAI.get_chat_response.retry, "sleep", sleep_mock)

        mistral = MistralAI(token="stub-token")
        auth_error = _mistral_sdk_error(401)

        async def _always_unauthorized(**kwargs):
            raise auth_error

        monkeypatch.setattr(mistral, "_get_chat_completion_response", _always_unauthorized)

        user = Mock()
        user.providers = Mock()
        user.providers.chat_ready = {}
        user.providers.get_instance = lambda provider_class: None

        db = AsyncMock()
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        engine = FailoverEngine(user=user)

        async def call(provider, model):
            return await provider.get_chat_response(
                messages=[], user=user, caller_storage_id=1, caller_thread_id=0, model=model
            )

        with pytest.raises(SDKError):
            await engine.run(role="master", primary_provider=mistral, model="mistral-medium-latest", call=call)

        assert engine.cooldown._memory == {}
        # 401 is retryable=False for the tenacity policy: a single attempt, no sleeps.
        assert sleep_mock.await_count == 0
