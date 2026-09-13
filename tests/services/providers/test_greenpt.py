"""Unit tests for the GreenPT provider adapter."""

from chibi.config import gpt_settings
from chibi.services.providers import GreenPT, RegisteredProviders

TEST_TOKEN = "test-greenpt-token"


def _make_provider() -> GreenPT:
    return GreenPT(token=TEST_TOKEN)


def test_greenpt_registered_in_registered_providers() -> None:
    """The class is registered under its lowercase provider name on import."""
    assert RegisteredProviders.all["greenpt"] is GreenPT


def test_availability_gated_on_api_key() -> None:
    """The provider is only marked available when a truthy api_key is configured."""
    assert GreenPT.api_key == gpt_settings.greenpt_key
    if GreenPT.api_key:
        assert "greenpt" in RegisteredProviders.available
    else:
        assert "greenpt" not in RegisteredProviders.available


def test_provider_attributes() -> None:
    """OpenAI-compatible base_url, chat/vision ready, no moderation/image/tts/stt."""
    assert GreenPT.name == "GreenPT"
    assert GreenPT.base_url == "https://api.greenpt.ai/v1"
    assert GreenPT.chat_ready is True
    assert GreenPT.vision_ready is True
    assert GreenPT.moderation_ready is False
    assert GreenPT.image_generation_ready is False
    assert GreenPT.tts_ready is False
    assert GreenPT.stt_ready is False
    assert GreenPT.ocr_ready is False
    assert GreenPT.default_model == "glm-5.2"
    assert GreenPT.default_vision_model == "kimi-k3"


def test_token_assignment() -> None:
    """The constructor stores the provided token."""
    provider = _make_provider()
    assert provider.token == TEST_TOKEN


def test_model_filtering_excludes_non_chat_models() -> None:
    """Embedding and reranker model IDs are filtered out of the chat model list."""
    from chibi.schemas.app import ModelChangeSchema

    provider = _make_provider()
    models = [
        ModelChangeSchema(provider="GreenPT", name="glm-5.2", image_generation=False),
        ModelChangeSchema(provider="GreenPT", name="green-embedding", image_generation=False),
        ModelChangeSchema(provider="GreenPT", name="green-rerank", image_generation=False),
    ]
    filtered = provider.filter_and_return_list_of_models(models=models, image_generation=False)
    names = [m.name for m in filtered]
    assert "glm-5.2" in names
    assert "green-embedding" not in names
    assert "green-rerank" not in names
