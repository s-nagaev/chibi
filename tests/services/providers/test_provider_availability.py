"""RegisteredProviders availability logging and the chat-ready fallback rules."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from loguru import logger

from chibi.models import MODELS_DISCOVERY_TIMEOUT_SECONDS
from chibi.services.providers import DeepSeek, RegisteredProviders


class _LogCollector:
    """Collect loguru records as strings for the availability assertions."""

    def __init__(self) -> None:
        """Start with no collected records."""
        self.records: list[str] = []

    def __call__(self, message: Any) -> None:
        """Append one stringified log record.

        Args:
            message: The loguru message object passed to the sink.
        """
        self.records.append(str(message))


class _FakeProvider:
    """Minimal stand-in for a provider class in the availability logging tests."""

    name = "FakeProvider"


def _capture_availability_logs(monkeypatch: pytest.MonkeyPatch, available: dict[str, Any]) -> list[str]:
    """Capture the records produced by two consecutive availability log attempts.

    The registry availability mapping is replaced for the duration of the test
    and restored by the monkeypatch fixture, so the outcome no longer depends
    on the provider keys present on the machine running the suite and no other
    test observes a mutated registry.

    Args:
        monkeypatch: The active monkeypatch fixture.
        available: The availability mapping the logging must observe.

    Returns:
        The captured log record texts.
    """
    collector = _LogCollector()
    handler_id = logger.add(collector, level="INFO")
    monkeypatch.setattr(RegisteredProviders, "available", available)
    monkeypatch.setattr(RegisteredProviders, "_availability_logged", False)
    try:
        RegisteredProviders._log_availability_once()
        RegisteredProviders._log_availability_once()
    finally:
        logger.remove(handler_id)
    return collector.records


def test_availability_logged_once_with_names_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """With available providers the registry logs their names once and never again."""
    records = _capture_availability_logs(monkeypatch, {"fake_provider": _FakeProvider})

    matching = [record for record in records if "Available providers" in record]
    assert len(matching) == 1
    assert "fake_provider" in matching[0]
    assert not any("No providers are available" in record for record in records)


def test_availability_warns_once_when_no_providers_are_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no available providers the registry logs the warning branch once."""
    records = _capture_availability_logs(monkeypatch, {})

    matching = [record for record in records if "No providers are available" in record]
    assert len(matching) == 1
    assert not any("Available providers" in record for record in records)


def test_first_chat_ready_skips_empty_default_providers() -> None:
    """A chat-ready provider without a default model never wins the implicit fallback."""
    empty = SimpleNamespace(name="empty", default_model="")
    good = SimpleNamespace(name="good", default_model="m1")
    registry = SimpleNamespace(
        chat_ready={"empty": empty, "good": good},
        get_instance=lambda provider: provider,
    )

    assert RegisteredProviders.first_chat_ready.__get__(registry) is good


def test_first_chat_ready_returns_none_when_every_default_is_empty() -> None:
    """With only empty-default candidates the fallback reports nothing."""
    empty = SimpleNamespace(name="empty", default_model="   ")
    registry = SimpleNamespace(
        chat_ready={"empty": empty},
        get_instance=lambda provider: provider,
    )

    assert RegisteredProviders.first_chat_ready.__get__(registry) is None


async def test_models_list_is_called_with_an_explicit_timeout() -> None:
    """The OpenAI-compatible enumeration passes the discovery timeout to the SDK call."""
    provider = DeepSeek("test_token")

    with patch("chibi.services.providers.provider.AsyncOpenAI") as client_cls:
        # A plain namespace is not callable, so the provider's attribute
        # wrapping returns it as-is instead of wrapping it in a function.
        client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=Mock(data=[Mock(id="m1")]))))
        client_cls.return_value = client

        await provider.get_available_models()

        client.models.list.assert_awaited_once_with(timeout=MODELS_DISCOVERY_TIMEOUT_SECONDS)
