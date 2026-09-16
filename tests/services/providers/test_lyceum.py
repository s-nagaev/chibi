"""Unit tests for the Lyceum provider adapter."""

from chibi.config import gpt_settings
from chibi.services.providers import Lyceum, RegisteredProviders

TEST_TOKEN = "test-lyceum-token"


def _make_provider() -> Lyceum:
    return Lyceum(token=TEST_TOKEN)


def test_lyceum_registered_in_registered_providers() -> None:
    """The class is registered under its lowercase provider name on import."""
    assert RegisteredProviders.all["lyceum"] is Lyceum


def test_availability_gated_on_api_key() -> None:
    """The provider is only marked available when a truthy api_key is configured."""
    assert Lyceum.api_key == gpt_settings.lyceum_key
    if Lyceum.api_key:
        assert "lyceum" in RegisteredProviders.available
    else:
        assert "lyceum" not in RegisteredProviders.available


def test_provider_attributes() -> None:
    """chat/vision/moderation ready, no image/tts/stt/ocr."""
    assert Lyceum.name == "Lyceum"
    assert Lyceum.chat_ready is True
    assert Lyceum.vision_ready is True
    assert Lyceum.moderation_ready is True
    assert Lyceum.image_generation_ready is False
    assert Lyceum.tts_ready is False
    assert Lyceum.stt_ready is False
    assert Lyceum.ocr_ready is False
    assert Lyceum.temperature == 0.6


def test_token_assignment() -> None:
    """The constructor stores the provided token."""
    provider = _make_provider()
    assert provider.token == TEST_TOKEN


def test_model_filtering_excludes_non_chat_models() -> None:
    """Embedding/speech/image model IDs are filtered out of the chat model list."""
    from chibi.schemas.app import ModelChangeSchema

    provider = _make_provider()
    models = [
        ModelChangeSchema(provider="Lyceum", name="kimi-k3", image_generation=False),
        ModelChangeSchema(provider="Lyceum", name="qwen3-embedding-8b", image_generation=False),
        ModelChangeSchema(provider="Lyceum", name="whisper-large-v3", image_generation=False),
    ]
    filtered = provider.filter_and_return_list_of_models(models=models, image_generation=False)
    names = [m.name for m in filtered]
    assert "kimi-k3" in names
    assert "qwen3-embedding-8b" not in names
    assert "whisper-large-v3" not in names
