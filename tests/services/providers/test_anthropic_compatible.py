"""Tests for dynamically created Anthropic-compatible provider instances."""

from collections.abc import Iterator

import pytest

import chibi.services.providers  # noqa: F401
from chibi.services.providers.anthropic_compatible import (
    AnthropicCompatible,
    collect_anthropic_compatible_configs,
    create_anthropic_compatible,
    register_anthropic_compatible_instances,
)
from chibi.services.providers.compatible import validate_instance_name
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


def test_base_class_does_not_self_register() -> None:
    """The AnthropicCompatible base has no name and stays out of the registry."""
    assert not hasattr(AnthropicCompatible, "name")
    assert "anthropiccompatible" not in RegisteredProviders.all
    assert issubclass(AnthropicCompatible, Provider)


def test_factory_registers_named_instance(restore_registry: None) -> None:
    """A created instance is registered under its lowercased name."""
    provider_cls = create_anthropic_compatible(name="MyClaude", api_key="key", base_url="https://claude.example.com")
    assert RegisteredProviders.all["myclaude"] is provider_cls
    assert provider_cls.name == "MyClaude"
    assert provider_cls.api_key == "key"
    assert getattr(provider_cls, "base_url") == "https://claude.example.com"
    assert provider_cls.chat_ready is True
    assert provider_cls.default_model == ""
    assert getattr(provider_cls, "is_dynamic_provider") is True
    assert isinstance(provider_cls, type) and issubclass(provider_cls, AnthropicCompatible)


def test_two_named_instances_coexist(restore_registry: None) -> None:
    """Two distinct named instances register side by side."""
    first = create_anthropic_compatible(name="Alpha", api_key="key-1", base_url="https://alpha.example.com")
    second = create_anthropic_compatible(name="Beta", api_key="key-2", base_url="https://beta.example.com")
    assert RegisteredProviders.all["alpha"] is first
    assert RegisteredProviders.all["beta"] is second
    assert first.name == "Alpha"
    assert second.name == "Beta"


def test_instance_with_key_is_registered_as_available(restore_registry: None) -> None:
    """A truthy api_key marks the instance available."""
    provider_cls = create_anthropic_compatible(name="Keyed", api_key="key", base_url="https://keyed.example.com")
    assert RegisteredProviders.available["keyed"] is provider_cls


def test_instance_without_key_is_not_available(restore_registry: None) -> None:
    """A falsy api_key keeps the instance registered but not available."""
    create_anthropic_compatible(name="Keyless", api_key=None, base_url="https://keyless.example.com")
    assert "keyless" in RegisteredProviders.all
    assert "keyless" not in RegisteredProviders.available


def test_factory_rejects_invalid_name_charset(restore_registry: None) -> None:
    """Names outside the allowed charset raise ValueError."""
    with pytest.raises(ValueError):
        create_anthropic_compatible(name="My Claude", api_key="key", base_url="https://x.example.com")
    assert "my claude" not in RegisteredProviders.all


def test_factory_rejects_empty_name(restore_registry: None) -> None:
    """An empty name raises ValueError."""
    with pytest.raises(ValueError):
        create_anthropic_compatible(name="", api_key="key", base_url="https://x.example.com")


def test_factory_rejects_collision_with_existing_registry_key(restore_registry: None) -> None:
    """A name colliding case-insensitively with an existing key raises ValueError."""
    existing = RegisteredProviders.all["anthropic"]
    with pytest.raises(ValueError):
        create_anthropic_compatible(name="Anthropic", api_key="key", base_url="https://x.example.com")
    assert RegisteredProviders.all["anthropic"] is existing


def test_factory_rejects_collision_with_openai_compatible_instance(restore_registry: None) -> None:
    """The name namespace is shared with OpenAI-compatible instances."""
    existing = RegisteredProviders.all["customopenai"]
    with pytest.raises(ValueError):
        create_anthropic_compatible(name="CustomOpenAI", api_key="key", base_url="https://x.example.com")
    assert RegisteredProviders.all["customopenai"] is existing


def test_validate_instance_name_is_pure() -> None:
    """Validation works over an injected key mapping without touching the registry."""
    with pytest.raises(ValueError):
        validate_instance_name("ANTHROPIC", ["anthropic"])
    with pytest.raises(ValueError):
        validate_instance_name("bad name", [])
    validate_instance_name("Fresh", ["other"])


def test_collect_configs_parses_pairs() -> None:
    """The adapter pairs API keys with their URLs and applies the default URL."""
    env = {
        "ANTHROPICCOMPATIBLE_ONE_API_KEY": "key-1",
        "ANTHROPICCOMPATIBLE_ONE_URL": "https://one.example.com",
        "ANTHROPICCOMPATIBLE_TWO_API_KEY": "key-2",
        "ANTHROPICCOMPATIBLE_TWO_URL": "",
        "ANTHROPICCOMPATIBLE_EMPTY_API_KEY": "",
        "UNRELATED_API_KEY": "key-3",
    }
    configs = collect_anthropic_compatible_configs(env)
    assert configs == {
        "ONE": {"api_key": "key-1", "base_url": "https://one.example.com"},
        "TWO": {"api_key": "key-2", "base_url": "https://api.anthropic.com"},
    }


def test_collect_configs_skips_invalid_names() -> None:
    """Entries whose name part violates the charset are skipped."""
    env = {
        "ANTHROPICCOMPATIBLE_MY-PROXY_API_KEY": "key",
        "ANTHROPICCOMPATIBLE_GOOD_API_KEY": "key",
    }
    configs = collect_anthropic_compatible_configs(env)
    assert list(configs) == ["GOOD"]


def test_register_instances_from_env(restore_registry: None) -> None:
    """Instances are registered from an injected env mapping."""
    env = {
        "ANTHROPICCOMPATIBLE_Gamma_API_KEY": "key",
        "ANTHROPICCOMPATIBLE_Gamma_URL": "https://gamma.example.com",
    }
    register_anthropic_compatible_instances(env)
    registered = RegisteredProviders.all["gamma"]
    assert registered.name == "Gamma"
    assert registered.api_key == "key"
    assert getattr(registered, "base_url") == "https://gamma.example.com"


def test_register_skips_collision_with_existing_registry_key(restore_registry: None) -> None:
    """A colliding env instance is skipped with a warning, not a crash."""
    existing = RegisteredProviders.all["anthropic"]
    env = {"ANTHROPICCOMPATIBLE_ANTHROPIC_API_KEY": "key"}
    register_anthropic_compatible_instances(env)
    assert RegisteredProviders.all["anthropic"] is existing


def test_register_skips_case_only_collision_between_instances(restore_registry: None) -> None:
    """Two env instances whose names differ only by case register exactly one.

    The winner is the first name in the deterministic processing order (sorted
    by registry key, then by the raw name), so ``DUO`` survives over ``duo``.
    """
    env = {
        "ANTHROPICCOMPATIBLE_DUO_API_KEY": "key-1",
        "ANTHROPICCOMPATIBLE_duo_API_KEY": "key-2",
    }
    register_anthropic_compatible_instances(env)
    assert "duo" in RegisteredProviders.all
    assert "DUO" not in RegisteredProviders.all
    assert RegisteredProviders.all["duo"].name == "DUO"
    assert RegisteredProviders.all["duo"].api_key == "key-1"


def test_case_only_collision_winner_does_not_depend_on_env_order(restore_registry: None) -> None:
    """Reversing the env iteration order does not change the surviving instance."""
    forward = {
        "ANTHROPICCOMPATIBLE_DUO_API_KEY": "key-1",
        "ANTHROPICCOMPATIBLE_duo_API_KEY": "key-2",
    }
    register_anthropic_compatible_instances(forward)
    forward_winner = RegisteredProviders.all.pop("duo")
    RegisteredProviders.available.pop("duo", None)

    register_anthropic_compatible_instances(dict(reversed(list(forward.items()))))
    reversed_winner = RegisteredProviders.all["duo"]

    assert forward_winner.__name__ == reversed_winner.__name__
    assert forward_winner.api_key == reversed_winner.api_key
