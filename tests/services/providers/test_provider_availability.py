"""RegisteredProviders availability logging and the chat-ready fallback rules."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from loguru import logger

from chibi.models import MODELS_DISCOVERY_TIMEOUT_SECONDS
from chibi.services.providers import DeepSeek, RegisteredProviders


def test_availability_logged_once_with_names_only() -> None:
    """The first registry construction logs provider names once and never logs again."""
    records: list[Any] = []
    handler_id = logger.add(lambda message: records.append(str(message)), level="INFO")
    RegisteredProviders._availability_logged = False
    try:
        RegisteredProviders._log_availability_once()
        RegisteredProviders._log_availability_once()
    finally:
        logger.remove(handler_id)
        RegisteredProviders._availability_logged = False

    expected = ", ".join(sorted(RegisteredProviders.available))
    matching = [record for record in records if "Available providers" in record]
    assert len(matching) == 1
    assert expected in matching[0]


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
