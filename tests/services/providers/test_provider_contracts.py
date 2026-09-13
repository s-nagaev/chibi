"""Invariant contract tests for the provider registry.

These tests enumerate ``RegisteredProviders.all`` (key-independent, never
``.available``) and verify class-attribute invariants only: no provider is
instantiated, no network is used and ``gpt_settings`` is never patched.
"""

import pytest

import chibi.services.providers  # noqa: F401
from chibi.config import gpt_settings
from chibi.services.providers.customopenai import CustomOpenAI
from chibi.services.providers.minimax import Minimax
from chibi.services.providers.provider import Provider, RegisteredProviders
from chibi.services.providers.suno import Suno
from chibi.services.providers.xiaomi import Xiaomi
from chibi.services.providers.zhipuai import ZhipuAI

ALL_PROVIDERS: list[type[Provider]] = list(RegisteredProviders.all.values())
BASE_URL_PROVIDERS: list[type[Provider]] = [
    provider_cls for provider_cls in ALL_PROVIDERS if getattr(provider_cls, "base_url", None) is not None
]

EXPECTED_PROVIDER_COUNT = 30

KNOWN_EMPTY_DEFAULT_MODEL = frozenset({"CustomOpenAI"})
KNOWN_MISSING_TTS_DEFAULTS = frozenset({"CustomOpenAI"})
KNOWN_MISSING_STT_DEFAULT = frozenset({"CustomOpenAI"})
KNOWN_MISSING_OCR_DEFAULT = frozenset({"Anthropic"})
KNOWN_NON_HTTPS_BASE_URL = frozenset({"CustomOpenAI"})


def test_registry_contains_all_providers() -> None:
    """The registry must hold every provider imported by the package.

    The count is exact on purpose: registering a new provider or losing a
    registration silently must both fail here and force a deliberate update.
    """
    assert len(ALL_PROVIDERS) == EXPECTED_PROVIDER_COUNT


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_registry_key_matches_lowercased_name(provider_cls: type[Provider]) -> None:
    """The registry key must be the lowercased provider name."""
    assert provider_cls.name
    assert RegisteredProviders.all[provider_cls.name.lower()] is provider_cls


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_chat_ready_providers_define_default_model(provider_cls: type[Provider]) -> None:
    """``chat_ready=True`` requires a non-empty ``default_model``.

    ``CustomOpenAI`` is the documented exception: its endpoint is fully
    user-configured and the model is selected at runtime.
    """
    if provider_cls.name in KNOWN_EMPTY_DEFAULT_MODEL:
        assert not getattr(provider_cls, "default_model", None)
        return
    if provider_cls.chat_ready:
        assert getattr(provider_cls, "default_model", None), f"{provider_cls.name} is chat_ready without default_model"


def test_empty_default_model_exceptions_match_reality() -> None:
    """The exception set must equal the real set of chat_ready providers with empty defaults."""
    actual = {
        provider_cls.name
        for provider_cls in ALL_PROVIDERS
        if provider_cls.chat_ready and not getattr(provider_cls, "default_model", None)
    }
    assert actual == KNOWN_EMPTY_DEFAULT_MODEL


def test_custom_openai_default_model_intentionally_empty() -> None:
    """CustomOpenAI ships an empty default model on purpose."""
    assert CustomOpenAI.chat_ready is True
    assert CustomOpenAI.default_model == ""


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_vision_ready_providers_define_default_vision_model(provider_cls: type[Provider]) -> None:
    """``vision_ready=True`` requires a non-empty ``default_vision_model``."""
    if provider_cls.vision_ready:
        assert getattr(provider_cls, "default_vision_model", None), (
            f"{provider_cls.name} is vision_ready without default_vision_model"
        )


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_moderation_ready_providers_define_default_moderation_model(provider_cls: type[Provider]) -> None:
    """``moderation_ready=True`` requires a non-empty ``default_moderation_model``."""
    if provider_cls.moderation_ready:
        assert getattr(provider_cls, "default_moderation_model", None), (
            f"{provider_cls.name} is moderation_ready without default_moderation_model"
        )


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_image_generation_ready_providers_define_default_image_model(provider_cls: type[Provider]) -> None:
    """``image_generation_ready=True`` requires a non-empty ``default_image_model``."""
    if provider_cls.image_generation_ready:
        assert getattr(provider_cls, "default_image_model", None), (
            f"{provider_cls.name} is image_generation_ready without default_image_model"
        )


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_tts_ready_providers_define_tts_defaults(provider_cls: type[Provider]) -> None:
    """``tts_ready=True`` requires non-empty ``default_tts_model`` and ``default_tts_voice``.

    ``CustomOpenAI`` is the documented exception: both values resolve at
    runtime through the ``Provider.tts_model``/``tts_voice`` properties, which
    fall back to ``gpt_settings``.
    """
    if provider_cls.name in KNOWN_MISSING_TTS_DEFAULTS:
        return
    if provider_cls.tts_ready:
        assert getattr(provider_cls, "default_tts_model", None), (
            f"{provider_cls.name} is tts_ready without default_tts_model"
        )
        assert getattr(provider_cls, "default_tts_voice", None), (
            f"{provider_cls.name} is tts_ready without default_tts_voice"
        )


def test_missing_tts_defaults_exceptions_match_reality() -> None:
    """The exception set must equal the real set of tts_ready providers with missing defaults."""
    actual = {
        provider_cls.name
        for provider_cls in ALL_PROVIDERS
        if provider_cls.tts_ready
        and (
            not getattr(provider_cls, "default_tts_model", None) or not getattr(provider_cls, "default_tts_voice", None)
        )
    }
    assert actual == KNOWN_MISSING_TTS_DEFAULTS


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_stt_ready_providers_define_default_stt_model(provider_cls: type[Provider]) -> None:
    """``stt_ready=True`` requires a non-empty ``default_stt_model``.

    Decision for ``CustomOpenAI``: it is kept as the documented exception
    because its STT model resolves at runtime through the ``Provider.stt_model``
    property, which falls back to ``gpt_settings.stt_model``.
    """
    if provider_cls.name in KNOWN_MISSING_STT_DEFAULT:
        return
    if provider_cls.stt_ready:
        assert getattr(provider_cls, "default_stt_model", None), (
            f"{provider_cls.name} is stt_ready without default_stt_model"
        )


def test_missing_stt_default_exceptions_match_reality() -> None:
    """The exception set must equal the real set of stt_ready providers with missing defaults."""
    actual = {
        provider_cls.name
        for provider_cls in ALL_PROVIDERS
        if provider_cls.stt_ready and not getattr(provider_cls, "default_stt_model", None)
    }
    assert actual == KNOWN_MISSING_STT_DEFAULT


@pytest.mark.parametrize("provider_cls", ALL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_ocr_ready_providers_define_default_ocr_model(provider_cls: type[Provider]) -> None:
    """``ocr_ready=True`` requires a non-empty ``default_ocr_model``.

    ``Anthropic`` is the documented exception: its ``ocr()`` method falls back
    to ``default_vision_model``, so that attribute must be non-empty instead.
    """
    if provider_cls.name in KNOWN_MISSING_OCR_DEFAULT:
        assert getattr(provider_cls, "default_vision_model", None)
        return
    if provider_cls.ocr_ready:
        assert getattr(provider_cls, "default_ocr_model", None), (
            f"{provider_cls.name} is ocr_ready without default_ocr_model"
        )


def test_missing_ocr_default_exceptions_match_reality() -> None:
    """The exception set must equal the real set of ocr_ready providers with missing defaults."""
    actual = {
        provider_cls.name
        for provider_cls in ALL_PROVIDERS
        if provider_cls.ocr_ready and not getattr(provider_cls, "default_ocr_model", None)
    }
    assert actual == KNOWN_MISSING_OCR_DEFAULT


@pytest.mark.parametrize("provider_cls", BASE_URL_PROVIDERS, ids=lambda provider_cls: provider_cls.name)
def test_base_url_uses_https(provider_cls: type[Provider]) -> None:
    """Every provider exposing a ``base_url`` must use https.

    ``CustomOpenAI`` is the documented exception: its URL is user-configurable
    and defaults to a local http endpoint (e.g. LM Studio).
    """
    base_url = getattr(provider_cls, "base_url", None)
    assert base_url
    if provider_cls.name in KNOWN_NON_HTTPS_BASE_URL:
        return
    assert base_url.startswith("https://"), f"{provider_cls.name} base_url must use https"


def test_cloudflare_base_url_interpolation() -> None:
    """Cloudflare's ``base_url`` is interpolated at class-definition time.

    The literal ``${`` placeholder must never survive interpolation. When
    ``cloudflare_account_id`` is unset, the ``$None`` placeholder is a valid
    configuration state rather than a code defect, so the ``None`` check only
    applies when an account id is configured.
    """
    base_url = getattr(RegisteredProviders.all["cloudflare"], "base_url", None)
    assert base_url
    assert base_url.startswith("https://")
    assert "${" not in base_url
    if gpt_settings.cloudflare_account_id is not None:
        assert "None" not in base_url


def test_zhipuai_dead_defaults_stay_visible() -> None:
    """ZhipuAI defines vision/image defaults while the corresponding flags are off.

    Asserting today's values keeps any silent flag flip or default change
    visible in the diff; enabling the flags is out of scope for this suite.
    """
    assert ZhipuAI.vision_ready is False
    assert ZhipuAI.default_vision_model == "GLM-4.6V-FlashX"
    assert ZhipuAI.image_generation_ready is False
    assert ZhipuAI.default_image_model == "glm-image"


def test_minimax_dead_image_default_stays_visible() -> None:
    """Minimax defines an image default while ``image_generation_ready`` is off."""
    assert Minimax.image_generation_ready is False
    assert Minimax.default_image_model == "image-01"


def test_suno_music_ready_contract() -> None:
    """Suno's custom ``music_ready`` flag maps to its default model."""
    assert Suno.music_ready is True
    assert Suno.default_model


def test_xiaomi_voice_ready_contract() -> None:
    """Xiaomi's custom ``voice_ready`` flag maps to its TTS voice default."""
    assert Xiaomi.voice_ready is True
    assert Xiaomi.default_tts_voice
