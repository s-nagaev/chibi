"""Tests for dynamically created OpenAI-compatible provider instances."""

from collections.abc import Iterator

import pytest

import chibi.services.providers  # noqa: F401
from chibi.config import gpt_settings
from chibi.services.providers.compatible import validate_instance_name
from chibi.services.providers.customopenai import CustomOpenAI
from chibi.services.providers.openai_compatible import (
    collect_openai_compatible_configs,
    create_openai_compatible,
    register_openai_compatible_instances,
)
from chibi.services.providers.provider import Provider, RegisteredProviders


@pytest.fixture()
def restore_registry() -> Iterator[None]:
    """Remove registry entries added by a test."""
    keys_before = set(RegisteredProviders.all)
    available_before = set(RegisteredProviders.available)
    yield
    for key in set(RegisteredProviders.all) - keys_before:
        RegisteredProviders.all.pop(key)
    for key in set(RegisteredProviders.available) - available_before:
        RegisteredProviders.available.pop(key)


def test_default_customopenai_instance_is_registered() -> None:
    """The back-compat default instance is registered unconditionally."""
    assert RegisteredProviders.all["customopenai"] is CustomOpenAI
    assert CustomOpenAI.name == "CustomOpenAI"
    assert CustomOpenAI.api_key == gpt_settings.customopenai_key
    assert getattr(CustomOpenAI, "base_url") == gpt_settings.customopenai_url
    assert CustomOpenAI.chat_ready is True
    assert CustomOpenAI.default_model == ""
    assert getattr(CustomOpenAI, "is_dynamic_provider") is False


def test_factory_registers_named_instance(restore_registry: None) -> None:
    """A created instance is registered under its lowercased name."""
    provider_cls = create_openai_compatible(name="MyProxy", api_key="key", base_url="https://proxy.example.com/v1")
    assert RegisteredProviders.all["myproxy"] is provider_cls
    assert provider_cls.name == "MyProxy"
    assert provider_cls.api_key == "key"
    assert getattr(provider_cls, "base_url") == "https://proxy.example.com/v1"
    assert provider_cls.chat_ready is True
    assert provider_cls.tts_ready is True
    assert provider_cls.stt_ready is True
    assert provider_cls.default_model == ""
    assert getattr(provider_cls, "is_dynamic_provider") is True
    assert isinstance(provider_cls, type) and issubclass(provider_cls, Provider)


def test_two_named_instances_coexist(restore_registry: None) -> None:
    """Two distinct named instances register side by side."""
    first = create_openai_compatible(name="Alpha", api_key="key-1", base_url="https://alpha.example.com/v1")
    second = create_openai_compatible(name="Beta", api_key="key-2", base_url="https://beta.example.com/v1")
    assert RegisteredProviders.all["alpha"] is first
    assert RegisteredProviders.all["beta"] is second
    assert first.name == "Alpha"
    assert second.name == "Beta"


def test_instance_with_key_is_registered_as_available(restore_registry: None) -> None:
    """A truthy api_key marks the instance available."""
    provider_cls = create_openai_compatible(name="Keyed", api_key="key", base_url="https://keyed.example.com/v1")
    assert RegisteredProviders.available["keyed"] is provider_cls


def test_instance_without_key_is_not_available(restore_registry: None) -> None:
    """A falsy api_key keeps the instance registered but not available."""
    create_openai_compatible(name="Keyless", api_key=None, base_url="https://keyless.example.com/v1")
    assert "keyless" in RegisteredProviders.all
    assert "keyless" not in RegisteredProviders.available


def test_factory_rejects_invalid_name_charset(restore_registry: None) -> None:
    """Names outside the allowed charset raise ValueError."""
    with pytest.raises(ValueError):
        create_openai_compatible(name="My Proxy", api_key="key", base_url="https://x.example.com/v1")
    assert "my proxy" not in RegisteredProviders.all


def test_factory_rejects_empty_name(restore_registry: None) -> None:
    """An empty name raises ValueError."""
    with pytest.raises(ValueError):
        create_openai_compatible(name="", api_key="key", base_url="https://x.example.com/v1")


def test_factory_rejects_collision_with_existing_registry_key(restore_registry: None) -> None:
    """A name colliding case-insensitively with an existing key raises ValueError."""
    existing = RegisteredProviders.all["openai"]
    with pytest.raises(ValueError):
        create_openai_compatible(name="OpenAI", api_key="key", base_url="https://x.example.com/v1")
    assert RegisteredProviders.all["openai"] is existing


def test_validate_instance_name_is_pure() -> None:
    """Validation works over an injected key mapping without touching the registry."""
    with pytest.raises(ValueError):
        validate_instance_name("OPENAI", ["openai"])
    with pytest.raises(ValueError):
        validate_instance_name("bad name", [])
    validate_instance_name("Fresh", ["other"])


def test_collect_configs_parses_pairs() -> None:
    """The adapter pairs API keys with their URLs and applies the default URL."""
    env = {
        "OPENAICOMPATIBLE_ONE_API_KEY": "key-1",
        "OPENAICOMPATIBLE_ONE_URL": "https://one.example.com/v1",
        "OPENAICOMPATIBLE_TWO_API_KEY": "key-2",
        "OPENAICOMPATIBLE_TWO_URL": "",
        "OPENAICOMPATIBLE_EMPTY_API_KEY": "",
        "UNRELATED_API_KEY": "key-3",
    }
    configs = collect_openai_compatible_configs(env)
    assert configs == {
        "ONE": {"api_key": "key-1", "base_url": "https://one.example.com/v1"},
        "TWO": {"api_key": "key-2", "base_url": "http://localhost:1234/v1"},
    }


def test_collect_configs_skips_invalid_names() -> None:
    """Entries whose name part violates the charset are skipped."""
    env = {
        "OPENAICOMPATIBLE_MY-PROXY_API_KEY": "key",
        "OPENAICOMPATIBLE_GOOD_API_KEY": "key",
    }
    configs = collect_openai_compatible_configs(env)
    assert list(configs) == ["GOOD"]


def test_register_instances_from_env(restore_registry: None) -> None:
    """Instances are registered from an injected env mapping."""
    env = {
        "OPENAICOMPATIBLE_Gamma_API_KEY": "key",
        "OPENAICOMPATIBLE_Gamma_URL": "https://gamma.example.com/v1",
    }
    register_openai_compatible_instances(env)
    registered = RegisteredProviders.all["gamma"]
    assert registered.name == "Gamma"
    assert registered.api_key == "key"
    assert getattr(registered, "base_url") == "https://gamma.example.com/v1"


def test_register_skips_collision_with_existing_registry_key(restore_registry: None) -> None:
    """A colliding env instance is skipped with a warning, not a crash."""
    existing = RegisteredProviders.all["openai"]
    env = {"OPENAICOMPATIBLE_OPENAI_API_KEY": "key"}
    register_openai_compatible_instances(env)
    assert RegisteredProviders.all["openai"] is existing


def test_register_skips_case_only_collision_between_instances(restore_registry: None) -> None:
    """Two env instances whose names differ only by case register exactly one.

    The winner is the first name in the deterministic processing order (sorted
    by registry key, then by the raw name), so ``DUO`` survives over ``duo``.
    """
    env = {
        "OPENAICOMPATIBLE_DUO_API_KEY": "key-1",
        "OPENAICOMPATIBLE_duo_API_KEY": "key-2",
    }
    register_openai_compatible_instances(env)
    assert "duo" in RegisteredProviders.all
    assert "DUO" not in RegisteredProviders.all
    assert RegisteredProviders.all["duo"].name == "DUO"
    assert RegisteredProviders.all["duo"].api_key == "key-1"


def test_case_only_collision_winner_does_not_depend_on_env_order(restore_registry: None) -> None:
    """Reversing the env iteration order does not change the surviving instance."""
    forward = {
        "OPENAICOMPATIBLE_DUO_API_KEY": "key-1",
        "OPENAICOMPATIBLE_duo_API_KEY": "key-2",
    }
    register_openai_compatible_instances(forward)
    forward_winner = RegisteredProviders.all.pop("duo")
    RegisteredProviders.available.pop("duo", None)

    register_openai_compatible_instances(dict(reversed(list(forward.items()))))
    reversed_winner = RegisteredProviders.all["duo"]

    assert forward_winner.__name__ == reversed_winner.__name__
    assert forward_winner.api_key == reversed_winner.api_key
