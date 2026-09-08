"""Unit tests for the Fireworks provider adapter."""

from typing import Callable, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, OpenAIError, RateLimitError

from chibi.config import gpt_settings
from chibi.exceptions import (
    ContextLengthExceededError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers import Fireworks, RegisteredProviders

TEST_TOKEN = "test-fireworks-token"


def _make_openai_error(error_cls: type[OpenAIError], status_code: int, code: str | None) -> OpenAIError:
    """Build an OpenAI SDK error with the given HTTP status and error code."""
    request = httpx.Request("POST", "https://api.fireworks.ai/inference/v1/chat/completions")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        json={"error": {"message": "test", "type": "invalid_request_error", "code": code}},
    )
    error_factory = cast("Callable[..., OpenAIError]", error_cls)
    return error_factory(
        "test",
        response=response,
        body={"message": "test", "type": "invalid_request_error", "code": code},
    )


def _make_provider() -> Fireworks:
    return Fireworks(token=TEST_TOKEN)


def _mock_client(side_effect: Exception) -> MagicMock:
    mock_client = MagicMock()
    mock_client.chat.completions.parse = AsyncMock(side_effect=side_effect)
    return mock_client


def test_fireworks_registered_in_registered_providers() -> None:
    """The class is registered under its lowercase provider name on import."""
    assert RegisteredProviders.all["fireworks"] is Fireworks


def test_availability_gated_on_api_key() -> None:
    """The provider is only marked available when a truthy api_key is configured."""
    assert Fireworks.api_key == gpt_settings.fireworks_key
    if Fireworks.api_key:
        assert "fireworks" in RegisteredProviders.available
    else:
        assert "fireworks" not in RegisteredProviders.available


def test_provider_attributes() -> None:
    """SPIKE-derived defaults: OpenAI-compatible base_url, full-id models, chat/vision/moderation ready."""
    assert Fireworks.name == "Fireworks"
    assert Fireworks.base_url == "https://api.fireworks.ai/inference/v1"
    assert Fireworks.chat_ready is True
    assert Fireworks.vision_ready is True
    assert Fireworks.moderation_ready is True
    assert Fireworks.default_model == "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    assert Fireworks.default_vision_model == "accounts/fireworks/models/glm-5p3-flash"
    assert Fireworks.default_moderation_model == "accounts/fireworks/models/glm-5p3-flash"


def test_model_filtering_chat_ready_models_sorted() -> None:
    """filter_and_return_list_of_models sorts descending by name and keeps chat-ready models."""
    provider = _make_provider()
    models = [
        ModelChangeSchema(
            provider="Fireworks",
            name="accounts/fireworks/models/kimi-k3",
            image_generation=False,
        ),
        ModelChangeSchema(
            provider="Fireworks",
            name="accounts/fireworks/models/glm-5p3-flash",
            image_generation=False,
        ),
    ]

    with patch("chibi.services.providers.provider.gpt_settings") as mock_settings:
        mock_settings.models_whitelist = []
        mock_settings.models_blacklist = []
        filtered = provider.filter_and_return_list_of_models(models)

    assert [model.name for model in filtered] == [
        "accounts/fireworks/models/kimi-k3",
        "accounts/fireworks/models/glm-5p3-flash",
    ]


def test_model_filtering_image_generation() -> None:
    """Only models with 'image' in the name survive the image_generation filter."""
    provider = _make_provider()
    models = [
        ModelChangeSchema(
            provider="Fireworks",
            name="accounts/fireworks/models/flux-1-schnell-fp8",
            image_generation=True,
        ),
        ModelChangeSchema(
            provider="Fireworks",
            name="accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
            image_generation=False,
        ),
    ]

    with patch("chibi.services.providers.provider.gpt_settings") as mock_settings:
        mock_settings.models_whitelist = []
        mock_settings.models_blacklist = []
        filtered = provider.filter_and_return_list_of_models(models, image_generation=True)

    assert [model.name for model in filtered] == ["accounts/fireworks/models/flux-1-schnell-fp8"]


@pytest.mark.asyncio
async def test_rate_limit_error_mapped_to_service_rate_limit_error() -> None:
    """OpenAI RateLimitError raised through the __getattribute__ wrapper becomes ServiceRateLimitError."""
    provider = _make_provider()
    provider.client = _mock_client(_make_openai_error(RateLimitError, 429, None))

    with pytest.raises(ServiceRateLimitError):
        await provider.vision(image=b"fake-image", mime_type="image/png")


@pytest.mark.asyncio
async def test_authentication_error_mapped_to_not_authorized_error() -> None:
    """OpenAI AuthenticationError raised through the __getattribute__ wrapper becomes NotAuthorizedError."""
    provider = _make_provider()
    provider.client = _mock_client(_make_openai_error(AuthenticationError, 401, "invalid_api_key"))

    with pytest.raises(NotAuthorizedError):
        await provider.vision(image=b"fake-image", mime_type="image/png")


@pytest.mark.asyncio
async def test_context_length_bad_request_mapped_to_context_length_exceeded_error() -> None:
    """A context_length_exceeded BadRequestError raised through the wrapper becomes ContextLengthExceededError."""
    provider = _make_provider()
    provider.client = _mock_client(_make_openai_error(BadRequestError, 400, "context_length_exceeded"))

    with pytest.raises(ContextLengthExceededError):
        await provider.vision(image=b"fake-image", mime_type="image/png")


@pytest.mark.asyncio
async def test_other_bad_request_mapped_to_service_response_error() -> None:
    """A non-context BadRequestError raised through the wrapper becomes ServiceResponseError."""
    provider = _make_provider()
    provider.client = _mock_client(_make_openai_error(BadRequestError, 400, "invalid_image"))

    with pytest.raises(ServiceResponseError):
        await provider.vision(image=b"fake-image", mime_type="image/png")
