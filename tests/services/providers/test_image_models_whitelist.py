"""Tests for the IMAGE_MODELS_WHITELIST setting: parsing, listing semantics and the generation guard."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.config.gpt import GPTSettings
from chibi.exceptions import ModelNotAllowedError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.tools import media
from chibi.services.providers.tools.media import GenerateImageTool
from chibi.services.user import generate_image

TEST_TOKEN = "test-image-models-whitelist"


class TestImageModelsWhitelistParsing:
    """Tests for parsing ``IMAGE_MODELS_WHITELIST`` into a list."""

    def test_unset_returns_empty_list(self) -> None:
        """An unset setting parses to an empty list."""
        assert GPTSettings(IMAGE_MODELS_WHITELIST=None).image_models_whitelist == []

    def test_parses_and_strips_values(self) -> None:
        """Comma-separated values are split and stripped."""
        settings = GPTSettings(IMAGE_MODELS_WHITELIST="dall-e-3, gpt-image-2 ,models/imagen-4.0-generate-001")
        assert settings.image_models_whitelist == ["dall-e-3", "gpt-image-2", "models/imagen-4.0-generate-001"]

    def test_single_value(self) -> None:
        """A single value parses to a one-element list."""
        assert GPTSettings(IMAGE_MODELS_WHITELIST="gpt-image-2").image_models_whitelist == ["gpt-image-2"]


def _image_models() -> list[ModelChangeSchema]:
    """Build an image model listing with one chat-only model mixed in.

    Returns:
        Two image generation models and one chat model.
    """
    return [
        ModelChangeSchema(provider="OpenAI", name="image-allowed", image_generation=True),
        ModelChangeSchema(provider="OpenAI", name="image-blocked", image_generation=True),
        ModelChangeSchema(provider="OpenAI", name="chat-only", image_generation=False),
    ]


def _settings(**overrides: Any) -> SimpleNamespace:
    """Build a stand-in gpt_settings with every filtering knob defaulted to empty.

    Args:
        **overrides: Filtering settings to set on the stand-in.

    Returns:
        A namespace carrying the model filtering settings.
    """
    settings = SimpleNamespace(
        image_models_whitelist=[],
        models_whitelist=[],
        models_blacklist=[],
        image_generations_whitelist=[],
    )
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def _filter_image_models(settings: SimpleNamespace) -> list[str]:
    """Filter the shared image listing through the provider under the given settings.

    Args:
        settings: Stand-in gpt_settings patched into the provider module.

    Returns:
        Names of the surviving image models.
    """
    provider = OpenAI(token=TEST_TOKEN)
    with patch("chibi.services.providers.provider.gpt_settings", settings):
        filtered = provider.filter_and_return_list_of_models(models=_image_models(), image_generation=True)
    return [model.name for model in filtered]


class TestImageListingSemantics:
    """Tests for the exclusive IMAGE_MODELS_WHITELIST semantics of the model listing."""

    def test_image_whitelist_is_applied_exclusively(self) -> None:
        """A global whitelist cannot zero out an image listing when the image whitelist is set."""
        settings = _settings(image_models_whitelist=["image-allowed"], models_whitelist=["chat-only"])

        assert _filter_image_models(settings) == ["image-allowed"]

    def test_global_blacklist_is_not_applied_to_images(self) -> None:
        """A global blacklist cannot remove a whitelisted image model."""
        settings = _settings(image_models_whitelist=["image-allowed"], models_blacklist=["image-allowed"])

        assert _filter_image_models(settings) == ["image-allowed"]

    def test_models_outside_the_image_whitelist_are_dropped(self) -> None:
        """The image whitelist keeps only the models it lists."""
        settings = _settings(image_models_whitelist=["image-blocked"])

        assert _filter_image_models(settings) == ["image-blocked"]

    def test_global_whitelist_still_applies_without_image_whitelist(self) -> None:
        """With the image whitelist unset the global whitelist filters image models as before."""
        settings = _settings(models_whitelist=["chat-only"])

        assert _filter_image_models(settings) == []

    def test_global_blacklist_still_applies_without_image_whitelist(self) -> None:
        """With the image whitelist unset the global blacklist filters image models as before."""
        settings = _settings(models_blacklist=["image-blocked"])

        assert _filter_image_models(settings) == ["image-allowed"]


def _mock_provider() -> Mock:
    """Build a mock image provider accepting every generation call.

    Returns:
        A mock provider with an async image call returning one URL.
    """
    provider = Mock()
    provider.name = "OpenAI"
    provider.get_images = AsyncMock(return_value=["https://example.com/result.png"])
    return provider


def _mock_db(active_provider: Mock, active_model: str | None) -> Mock:
    """Build a mock database serving a user with the given active image selection.

    Args:
        active_provider: Provider returned by both the explicit and the active lookup.
        active_model: Model name returned as the thread's active image model.

    Returns:
        A mock database instance.
    """
    user = Mock()
    user.providers.get = Mock(return_value=active_provider)
    user.get_active_image_provider = Mock(return_value=active_provider)
    user.get_active_image_model = Mock(return_value=active_model)
    db = Mock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.count_image = AsyncMock()
    return db


def _mock_interface() -> Mock:
    """Build a mock user interface with a fixed user id.

    Returns:
        A mock user interface.
    """
    interface = Mock()
    interface.user_id = 42
    return interface


@pytest.mark.asyncio
async def test_active_model_path_rejects_model_outside_whitelist() -> None:
    """The active-model path refuses a model missing from IMAGE_MODELS_WHITELIST before calling the provider."""
    provider = _mock_provider()
    db = _mock_db(active_provider=provider, active_model="image-blocked")
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        with pytest.raises(ModelNotAllowedError):
            await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    provider.get_images.assert_not_awaited()
    db.count_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_override_path_rejects_model_outside_whitelist() -> None:
    """The explicit provider/model override path (the media tool) is guarded the same way."""
    provider = _mock_provider()
    db = _mock_db(active_provider=provider, active_model="image-allowed")
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        with pytest.raises(ModelNotAllowedError):
            await generate_image.__wrapped__(
                db, interface=interface, prompt="a cat", provider_name="OpenAI", model="image-blocked"
            )

    provider.get_images.assert_not_awaited()
    db.count_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolved_default_model_is_rejected_when_whitelist_is_set() -> None:
    """With the whitelist set a request without a resolved model is refused, since it cannot be verified."""
    provider = _mock_provider()
    db = _mock_db(active_provider=provider, active_model=None)
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        with pytest.raises(ModelNotAllowedError):
            await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    provider.get_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_whitelisted_model_is_generated() -> None:
    """A whitelisted model reaches the provider on the override path."""
    provider = _mock_provider()
    db = _mock_db(active_provider=provider, active_model="image-allowed")
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        result = await generate_image.__wrapped__(
            db, interface=interface, prompt="a cat", provider_name="OpenAI", model="image-allowed"
        )

    provider.get_images.assert_awaited_once_with(prompt="a cat", model="image-allowed", images=None)
    assert result == ["https://example.com/result.png"]


@pytest.mark.asyncio
async def test_generation_is_unrestricted_without_whitelist() -> None:
    """Without the whitelist any model (including the provider default) is generated as before."""
    provider = _mock_provider()
    db = _mock_db(active_provider=provider, active_model=None)
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings()):
        result = await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    provider.get_images.assert_awaited_once_with(prompt="a cat", model=None, images=None)
    assert result == ["https://example.com/result.png"]


@pytest.mark.asyncio
async def test_media_tool_turns_the_rejection_into_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whitelist rejection inside the media tool is reported as a tool error, not a crash."""
    interface = Mock()
    interface.user_id = 42
    interface.send_images = AsyncMock()

    async def _reject(**kwargs: Any) -> Any:
        """Raise the guard error the way the user service does.

        Args:
            **kwargs: Ignored tool call arguments.

        Raises:
            ModelNotAllowedError: Always.
        """
        raise ModelNotAllowedError(provider="OpenAI", model="image-blocked", detail="Image model is not allowed.")

    monkeypatch.setattr(media, "user_has_reached_images_generation_limit", AsyncMock(return_value=False))
    monkeypatch.setattr(media, "generate_image", _reject)

    response = await GenerateImageTool._get_tool_call_result(
        provider="OpenAI", image_model="image-blocked", prompt="a cat", interface=interface, user_id=42
    )

    assert response.status == "error"
    assert "Image model is not allowed." in str(response.result)
    interface.send_images.assert_not_awaited()
