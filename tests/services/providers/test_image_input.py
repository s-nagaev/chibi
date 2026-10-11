"""Tests for image-to-image support in providers and the user generate_image service."""

import base64
from collections.abc import Callable, Iterator
from functools import partial
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chibi.config import gpt_settings
from chibi.exceptions import ServiceResponseError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import OpenAIFriendlyProvider
from chibi.services.user import generate_image

TEST_TOKEN = "test-token-for-image-input"


def _make_client_factory(models: "_FakeGeminiModels") -> Callable[..., "_FakeGeminiClient"]:
    """Return a callable that builds fake Gemini SDK clients bound to ``models``.

    Args:
        models: Fake models namespace shared by every built client.

    Returns:
        A factory accepting (and ignoring) the SDK client constructor arguments.
    """
    return partial(_FakeGeminiClient, models)


class _FakeGeminiModels:
    """Fake ``client.aio.models`` namespace recording image-generation calls."""

    def __init__(self) -> None:
        """Initialize empty call recorders and empty canned responses."""
        self.generate_content_calls: list[dict[str, Any]] = []
        self.generate_images_calls: list[dict[str, Any]] = []
        self.generate_content_response: Any = None
        self.generate_images_response: Any = None

    async def generate_content(self, **kwargs: Any) -> Any:
        """Record a generate_content call and return the canned response.

        Args:
            kwargs: Keyword arguments of the SDK call.

        Returns:
            The canned generate_content response (may be None).
        """
        self.generate_content_calls.append(kwargs)
        return self.generate_content_response

    async def generate_images(self, **kwargs: Any) -> Any:
        """Record a generate_images call and return the canned response.

        Args:
            kwargs: Keyword arguments of the SDK call.

        Returns:
            The canned generate_images response (may be None).
        """
        self.generate_images_calls.append(kwargs)
        return self.generate_images_response


class _FakeGeminiAio:
    """Fake async namespace of the google-genai client."""

    def __init__(self, models: _FakeGeminiModels) -> None:
        """Store the fake models namespace.

        Args:
            models: Fake models namespace to expose as ``aio.models``.
        """
        self.models = models

    async def __aenter__(self) -> "_FakeGeminiAio":
        """Return self as the async context value."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Accept and ignore SDK exit arguments."""
        return None


class _FakeGeminiClient:
    """Fake ``google.genai.client.Client`` hosting the fake models namespace."""

    def __init__(self, models: _FakeGeminiModels, *args: Any, **kwargs: Any) -> None:
        """Build the fake client around the shared models namespace.

        Args:
            models: Fake models namespace to expose under ``aio.models``.
            *args: Ignored SDK constructor arguments.
            **kwargs: Ignored SDK constructor arguments.
        """
        self.aio = _FakeGeminiAio(models)


@pytest.fixture(scope="function")
def gemini_models(monkeypatch: pytest.MonkeyPatch) -> _FakeGeminiModels:
    """Patch the Gemini SDK client with a fake and expose the recorded calls."""
    models = _FakeGeminiModels()
    monkeypatch.setattr(
        "chibi.services.providers.gemini_native.Client",
        _make_client_factory(models),
    )
    return models


def make_content_response(image_bytes: bytes) -> MagicMock:
    """Build a fake GenerateContentResponse carrying one generated image.

    Args:
        image_bytes: Bytes returned by the mocked image part.

    Returns:
        A mock response whose parts yield a single image.
    """
    image = MagicMock()
    image.image_bytes = image_bytes
    part = MagicMock()
    part.as_image.return_value = image
    response = MagicMock()
    response.parts = [part]
    return response


def make_images_edit_response(image_bytes: bytes) -> MagicMock:
    """Build a fake OpenAI images.edit/images.generate response with b64 data.

    Args:
        image_bytes: Raw image bytes encoded into the b64_json field.

    Returns:
        A mock response with a single b64_json image.
    """
    image = MagicMock()
    image.b64_json = base64.b64encode(image_bytes).decode()
    response = MagicMock()
    response.data = [image]
    return response


@pytest.mark.asyncio
async def test_gemini_content_model_sends_prompt_then_image_parts(
    gemini_models: _FakeGeminiModels,
) -> None:
    """Reference images are appended as inline parts after the prompt."""
    provider = Gemini(token=TEST_TOKEN)
    gemini_models.generate_content_response = make_content_response(b"generated-bytes")
    images = [(b"ref-one", "image/png"), (b"ref-two", "image/jpeg")]

    result = await provider.get_images(prompt="a cat", model="gemini-2.5-flash-image", images=images)

    call = gemini_models.generate_content_calls[0]
    contents = call["contents"]
    assert contents[0] == "a cat"
    assert len(contents) == 3
    assert [part.inline_data.data for part in contents[1:]] == [b"ref-one", b"ref-two"]
    assert [part.inline_data.mime_type for part in contents[1:]] == ["image/png", "image/jpeg"]
    assert len(result) == 1
    assert result[0].getvalue() == b"generated-bytes"


@pytest.mark.asyncio
async def test_gemini_content_model_without_images_keeps_prompt_only_contents(
    gemini_models: _FakeGeminiModels,
) -> None:
    """Without reference images the contents list is the bare prompt."""
    provider = Gemini(token=TEST_TOKEN)
    gemini_models.generate_content_response = make_content_response(b"generated-bytes")

    result = await provider.get_images(prompt="a cat", model="gemini-2.5-flash-image")

    call = gemini_models.generate_content_calls[0]
    assert call["contents"] == ["a cat"]
    assert result[0].getvalue() == b"generated-bytes"


@pytest.mark.asyncio
async def test_gemini_imagen_model_rejects_reference_images(
    gemini_models: _FakeGeminiModels,
) -> None:
    """Reference images for an imagen model raise ServiceResponseError before any SDK call."""
    provider = Gemini(token=TEST_TOKEN)

    with pytest.raises(ServiceResponseError, match="imagen"):
        await provider.get_images(
            prompt="a cat", model="models/imagen-4.0-fast-generate-001", images=[(b"ref", "image/png")]
        )

    assert gemini_models.generate_images_calls == []
    assert gemini_models.generate_content_calls == []


@pytest.mark.asyncio
async def test_gemini_imagen_model_without_images_uses_generate_images(
    gemini_models: _FakeGeminiModels,
) -> None:
    """Without reference images the imagen path calls generate_images unchanged."""
    provider = Gemini(token=TEST_TOKEN)
    generated_image = MagicMock()
    generated_image.image_bytes = b"imagen-bytes"
    response = MagicMock()
    response.images = [generated_image]
    gemini_models.generate_images_response = response

    result = await provider.get_images(prompt="a cat", model="models/imagen-4.0-fast-generate-001")

    call = gemini_models.generate_images_calls[0]
    assert call["prompt"] == "a cat"
    assert call["model"] == "models/imagen-4.0-fast-generate-001"
    assert result[0].getvalue() == b"imagen-bytes"


@pytest.fixture(scope="function")
def openai_client() -> Iterator[SimpleNamespace]:
    """Patch the AsyncOpenAI client factory and return the fake client namespace.

    The fake is a SimpleNamespace (not callable) so the provider's
    ``__getattribute__`` wrapper leaves it untouched; ``images.edit`` and
    ``images.generate`` are AsyncMocks configured per test.
    """
    images = SimpleNamespace(edit=AsyncMock(), generate=AsyncMock())
    fake_client = SimpleNamespace(images=images)
    with patch("chibi.services.providers.provider.AsyncOpenAI", return_value=fake_client):
        yield fake_client


@pytest.mark.asyncio
async def test_openai_gpt_image_uses_images_edit(openai_client: SimpleNamespace) -> None:
    """Reference images for a gpt-image model produce a single images.edit call."""
    provider = OpenAI(token=TEST_TOKEN)
    openai_client.images.edit = AsyncMock(return_value=make_images_edit_response(b"edited-bytes"))
    images = [(b"ref-png", "image/png"), (b"ref-jpeg", "image/jpeg")]

    result = await provider.get_images(prompt="make it red", model="gpt-image-2", images=images)

    openai_client.images.edit.assert_awaited_once()
    kwargs = openai_client.images.edit.await_args.kwargs
    assert kwargs["model"] == "gpt-image-2"
    assert kwargs["prompt"] == "make it red"
    assert kwargs["image"] == [
        ("image_0.png", b"ref-png", "image/png"),
        ("image_1.jpg", b"ref-jpeg", "image/jpeg"),
    ]
    assert kwargs["n"] == provider.image_n_choices
    assert kwargs["size"] == provider.image_size
    assert kwargs["timeout"] == gpt_settings.timeout
    first = result[0]
    assert isinstance(first, BytesIO)
    assert first.getvalue() == b"edited-bytes"


@pytest.mark.asyncio
async def test_openai_dall_e_rejects_reference_images() -> None:
    """Reference images for a dall-e model raise ServiceResponseError."""
    provider = OpenAI(token=TEST_TOKEN)

    with pytest.raises(ServiceResponseError, match="gpt-image"):
        await provider.get_images(prompt="a cat", model="dall-e-3", images=[(b"ref", "image/png")])


@pytest.mark.asyncio
async def test_openai_without_images_keeps_generate_path(openai_client: SimpleNamespace) -> None:
    """Without reference images the plain generate path keeps its quality/size parameters."""
    provider = OpenAI(token=TEST_TOKEN)
    image = MagicMock()
    image.url = "https://example.com/generated.png"
    response = MagicMock()
    response.data = [image]
    openai_client.images.generate = AsyncMock(return_value=response)

    result = await provider.get_images(prompt="a cat", model="dall-e-3")

    openai_client.images.generate.assert_awaited_once()
    kwargs = openai_client.images.generate.await_args.kwargs
    assert kwargs["model"] == "dall-e-3"
    assert kwargs["prompt"] == "a cat"
    assert kwargs["n"] == provider.image_n_choices
    assert kwargs["quality"] == provider.image_quality
    assert kwargs["size"] == provider.image_size
    assert kwargs["timeout"] == provider.timeout
    assert result == ["https://example.com/generated.png"]


@pytest.mark.asyncio
async def test_openai_friendly_base_provider_rejects_reference_images() -> None:
    """The OpenAIFriendlyProvider base implementation rejects any reference images."""
    provider = OpenAI(token=TEST_TOKEN)

    with pytest.raises(ServiceResponseError, match="does not support image input"):
        await OpenAIFriendlyProvider.get_images(
            provider, prompt="a cat", model="gpt-image-2", images=[(b"ref", "image/png")]
        )


@pytest.mark.asyncio
async def test_user_generate_image_passes_images_to_provider() -> None:
    """The user service passes reference images through to provider.get_images."""
    provider = Mock()
    provider.get_images = AsyncMock(return_value=["https://example.com/result.png"])
    user = Mock()
    user.providers.get = Mock(return_value=provider)
    db = Mock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.count_image = AsyncMock()
    interface = Mock()
    interface.user_id = 1
    images = [(b"ref", "image/png")]

    result = await generate_image.__wrapped__(
        db,
        interface=interface,
        prompt="a cat",
        model="gpt-image-2",
        provider_name="OpenAI",
        images=images,
    )

    provider.get_images.assert_awaited_once_with(prompt="a cat", model="gpt-image-2", images=images)
    assert result == ["https://example.com/result.png"]


def test_capability_flag_not_exposed_in_model_listing_schema() -> None:
    """The model-listing schema has no image-to-image flag yet (documented limitation)."""
    assert "image_to_image" not in ModelChangeSchema.model_fields
