"""Tests for the IMAGE_MODELS_WHITELIST setting: parsing, listing semantics and the generation guard."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.config.gpt import GPTSettings
from chibi.exceptions import ModelNotAllowedError
from chibi.interfaces.interface import UserInterface
from chibi.schemas.app import ModelChangeSchema
from chibi.services.bot import handle_image_generation
from chibi.services.providers.alibaba import Alibaba
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import OpenAIFriendlyProvider
from chibi.services.providers.tools import media
from chibi.services.providers.tools.media import GenerateImageTool
from chibi.services.providers.zhipuai import ZhipuAI
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


def _mock_provider(default_image_model: str | None = "image-allowed") -> Mock:
    """Build a mock image provider accepting every generation call.

    Args:
        default_image_model: Model the provider falls back to when the request
            carries no model name.

    Returns:
        A mock provider with an async image call returning one URL.
    """
    provider = Mock()
    provider.name = "OpenAI"
    provider.default_image_model = default_image_model
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
async def test_thread_without_selection_uses_whitelisted_provider_default() -> None:
    """A thread with no explicit image model falls back to the provider default, which is allowed when whitelisted."""
    provider = _mock_provider(default_image_model="image-allowed")
    db = _mock_db(active_provider=provider, active_model=None)
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        result = await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    provider.get_images.assert_awaited_once_with(prompt="a cat", model=None, images=None)
    db.count_image.assert_awaited_once_with(42)
    assert result == ["https://example.com/result.png"]


@pytest.mark.asyncio
async def test_provider_default_model_outside_whitelist_is_rejected() -> None:
    """A fallback model that is not whitelisted is refused before the provider call."""
    provider = _mock_provider(default_image_model="image-blocked")
    db = _mock_db(active_provider=provider, active_model=None)
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        with pytest.raises(ModelNotAllowedError) as error:
            await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    assert error.value.model == "image-blocked"
    provider.get_images.assert_not_awaited()
    db.count_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_resolving_to_no_model_is_rejected_when_whitelist_is_set() -> None:
    """Nothing to compare with the allowlist (no selection, no provider default) stays fail-closed."""
    provider = _mock_provider(default_image_model=None)
    db = _mock_db(active_provider=provider, active_model=None)
    interface = _mock_interface()

    with patch("chibi.services.user.gpt_settings", _settings(image_models_whitelist=["image-allowed"])):
        with pytest.raises(ModelNotAllowedError) as error:
            await generate_image.__wrapped__(db, interface=interface, prompt="a cat")

    assert error.value.model == "default"
    provider.get_images.assert_not_awaited()
    db.count_image.assert_not_awaited()


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


class _FakeImagineInterface(UserInterface):
    """Minimal interface driving the decorated ``/imagine`` handler."""

    error_code: str | None = None
    error_message: str | None = None

    def __init__(self) -> None:
        """Initialize empty capture buffers and a clear IDE error state."""
        self.sent_messages: list[str] = []
        self.sent_images: list[Any] = []

    @property
    def chat_id(self) -> int:
        """Return the fake chat identifier."""
        return 42

    @property
    def user_id(self) -> int:
        """Return the fake user identifier."""
        return 42

    @property
    def storage_id(self) -> int:
        """Return the fake storage key."""
        return 42

    @property
    def thread_id(self) -> int:
        """Return the fake thread identifier."""
        return 1

    @property
    def user_data(self) -> str:
        """Return the fake user description used in log lines."""
        return "User 42"

    @property
    def chat_data(self) -> str:
        """Return the fake chat description used in log lines."""
        return "chat 42, thread 1"

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Record an outgoing chat message.

        Args:
            message: Message text.
            reply: Ignored by the fake.
            **kwargs: Ignored by the fake.
        """
        self.sent_messages.append(message)

    async def send_images(self, images: Any, reply: bool = True, **kwargs: Any) -> None:
        """Record an outgoing image batch.

        Args:
            images: Image payloads.
            reply: Ignored by the fake.
            **kwargs: Ignored by the fake.
        """
        self.sent_images.append(images)

    async def send_action_uploading_photo(self) -> None:
        """Accept the upload indicator ping used by the handler."""


@pytest.mark.asyncio
async def test_imagine_command_reports_rejection_without_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/imagine`` turns a whitelist rejection into the specific message and an IDE error state."""
    interface = _FakeImagineInterface()
    monkeypatch.setattr("chibi.services.bot.user_has_reached_images_generation_limit", AsyncMock(return_value=False))
    monkeypatch.setattr(
        "chibi.services.bot.generate_image",
        AsyncMock(
            side_effect=ModelNotAllowedError(provider="OpenAI", model="image-blocked", detail="Image model refused.")
        ),
    )

    await handle_image_generation(prompt="a cat", interface=interface)

    assert interface.sent_images == []
    assert len(interface.sent_messages) == 1
    message = interface.sent_messages[0].lower()
    assert "not allowed by the administrator" in message
    assert "hiccup" not in message
    assert interface.error_code == "model_not_allowed"
    assert interface.error_message


class TestProviderImageListingsHonorTheWhitelist:
    """Tests for providers that build their image listing outside the shared filter."""

    @pytest.mark.asyncio
    async def test_zhipuai_image_listing_honors_the_whitelist(self) -> None:
        """The hardcoded ZhipuAI image listing is filtered by IMAGE_MODELS_WHITELIST."""
        provider = ZhipuAI(token=TEST_TOKEN)
        settings = _settings(image_models_whitelist=["glm-image"])

        with patch("chibi.services.providers.provider.gpt_settings", settings):
            models = await provider.get_available_models(image_generation=True)

        assert models is not None
        assert [model.name for model in models] == ["glm-image"]

    @pytest.mark.asyncio
    async def test_alibaba_image_listing_filters_the_appended_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Alibaba model appended after the base listing is whitelist-filtered as well."""
        provider = Alibaba(token=TEST_TOKEN)
        base_listing = [
            ModelChangeSchema(
                provider="Alibaba", name="qwen-image-plus", display_name="Qwen Image", image_generation=True
            )
        ]
        monkeypatch.setattr(OpenAIFriendlyProvider, "get_available_models", AsyncMock(return_value=list(base_listing)))
        settings = _settings(image_models_whitelist=["qwen-image-plus"])

        with patch("chibi.services.providers.provider.gpt_settings", settings):
            models = await provider.get_available_models(image_generation=True)

        assert models is not None
        assert [model.name for model in models] == ["qwen-image-plus"]

    @pytest.mark.asyncio
    async def test_alibaba_image_listing_keeps_a_whitelisted_appended_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A whitelisted hand-listed Alibaba model survives the listing filter."""
        provider = Alibaba(token=TEST_TOKEN)
        base_listing = [
            ModelChangeSchema(
                provider="Alibaba", name="qwen-image-plus", display_name="Qwen Image", image_generation=True
            )
        ]
        monkeypatch.setattr(OpenAIFriendlyProvider, "get_available_models", AsyncMock(return_value=list(base_listing)))
        settings = _settings(image_models_whitelist=["qwen-image-plus", "wan2.6-t2i"])

        with patch("chibi.services.providers.provider.gpt_settings", settings):
            models = await provider.get_available_models(image_generation=True)

        assert models is not None
        assert sorted(model.name for model in models) == ["qwen-image-plus", "wan2.6-t2i"]
