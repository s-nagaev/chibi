"""Contract tests for the provider get_available_models None-vs-[] semantics.

``None`` means the fetch failed; ``[]`` means the fetch succeeded but the
provider exposes zero matching models. Consumers rely on this distinction to
avoid caching transient failures as valid (empty) listings.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import OpenAIFriendlyProvider


class _FakeModelsApi:
    """Non-callable stand-in for the SDK's models API (callables get wrapped
    by OpenAIFriendlyProvider.__getattribute__)."""

    def __init__(self, list_mock: AsyncMock) -> None:
        self.list = list_mock


def _openai_models_list_mock(data: list[Any]) -> AsyncMock:
    result = MagicMock()
    result.data = data
    return AsyncMock(return_value=result)


def _mock_openai_model(model_id: str) -> MagicMock:
    model = MagicMock()
    model.id = model_id
    return model


async def test_openai_friendly_provider_returns_none_on_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAI(token="test-token")
    failing_client = SimpleNamespace(models=_FakeModelsApi(AsyncMock(side_effect=RuntimeError("connection refused"))))
    monkeypatch.setattr(OpenAIFriendlyProvider, "client", property(lambda self: failing_client))

    result = await provider.get_available_models()

    assert result is None


async def test_openai_friendly_provider_returns_empty_list_on_successful_empty_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAI(token="test-token")
    client = SimpleNamespace(models=_FakeModelsApi(_openai_models_list_mock([])))
    monkeypatch.setattr(OpenAIFriendlyProvider, "client", property(lambda self: client))

    result = await provider.get_available_models()

    assert result == []


async def test_openai_friendly_provider_returns_models_on_successful_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAI(token="test-token")
    client = SimpleNamespace(models=_FakeModelsApi(_openai_models_list_mock([_mock_openai_model("gpt-5.6-terra")])))
    monkeypatch.setattr(OpenAIFriendlyProvider, "client", property(lambda self: client))

    result = await provider.get_available_models()

    assert result is not None
    assert len(result) == 1
    assert result[0].name == "gpt-5.6-terra"


async def test_mistralai_provider_returns_none_on_fetch_failure() -> None:
    provider = MistralAI(token="test-token")
    failing_client = SimpleNamespace(models=SimpleNamespace(list_async=AsyncMock(side_effect=RuntimeError("boom"))))
    provider._client = failing_client  # type: ignore[assignment]

    result = await provider.get_available_models()

    assert result is None


async def test_mistralai_provider_returns_empty_list_on_successful_empty_fetch() -> None:
    provider = MistralAI(token="test-token")
    empty_result = MagicMock()
    empty_result.data = []
    client = SimpleNamespace(models=SimpleNamespace(list_async=AsyncMock(return_value=empty_result)))
    provider._client = client  # type: ignore[assignment]

    result = await provider.get_available_models()

    assert result == []


async def test_gemini_provider_returns_none_on_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Gemini(token="test-token")

    class _FailingAio:
        async def __aenter__(self) -> "_FailingAio":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        @property
        def models(self) -> Any:
            raise RuntimeError("gemini api unreachable")

    class _FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            self.aio = _FailingAio()

    monkeypatch.setattr("chibi.services.providers.gemini_native.Client", _FailingClient)

    result = await provider.get_available_models()

    assert result is None


async def test_base_provider_api_key_is_invalid_when_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAI(token="test-token")
    failing_client = SimpleNamespace(models=_FakeModelsApi(AsyncMock(side_effect=RuntimeError("unauthorized"))))
    monkeypatch.setattr(OpenAIFriendlyProvider, "client", property(lambda self: failing_client))

    assert await provider.api_key_is_valid() is False
