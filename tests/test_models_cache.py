"""Tests for the failure-aware caching of User.get_available_models."""

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chibi.models import MODELS_CACHE_TTL_SECONDS, ModelsUnavailableError, User
from chibi.schemas.app import ModelChangeSchema
from chibi.services.user import set_api_key
from chibi.storage.local import LocalStorage


@pytest.fixture(autouse=True)
def _clean_models_caches():
    """Keep the class-level model caches isolated between tests."""
    User._models_cache.clear()
    User._models_cache_backup.clear()
    User._models_failure_cooldowns.clear()
    yield
    User._models_cache.clear()
    User._models_cache_backup.clear()
    User._models_failure_cooldowns.clear()


def _model(name: str) -> ModelChangeSchema:
    return ModelChangeSchema(provider="Test", name=name, display_name=name, image_generation=False)


def _provider(name: str, result: Any) -> Any:
    provider = SimpleNamespace(name=name)
    provider.get_available_models = AsyncMock(return_value=result)
    return provider


def _install_providers(monkeypatch: pytest.MonkeyPatch, providers: list[Any]) -> None:
    registry = SimpleNamespace(available_instances=providers)
    monkeypatch.setattr(User, "providers", property(lambda self: registry))


def _expire_cache(user: User, image_generation: bool = False, video_generation: bool = False) -> None:
    key = user._models_cache_key(image_generation=image_generation, video_generation=video_generation)
    expires_at, models = User._models_cache[key]
    User._models_cache[key] = (time.monotonic() - 1, models)


async def test_success_is_cached_per_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider("openai", [_model("m1")])
    _install_providers(monkeypatch, [provider])
    user = User(id=1)

    assert await user.get_available_models() == [_model("m1")]
    assert await user.get_available_models() == [_model("m1")]

    provider.get_available_models.assert_awaited_once()
    entry = User._models_cache[user._models_cache_key(image_generation=False, video_generation=False)]
    assert entry[1] == [_model("m1")]
    assert entry[0] - time.monotonic() <= MODELS_CACHE_TTL_SECONDS


async def test_all_failed_serves_stale_previous_good_and_keeps_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    good = [_model("m1"), _model("m2")]
    provider = _provider("openai", good)
    _install_providers(monkeypatch, [provider])
    user = User(id=1)

    assert await user.get_available_models() == good
    _expire_cache(user)

    provider.get_available_models = AsyncMock(return_value=None)
    result = await user.get_available_models()

    assert result == good
    key = user._models_cache_key(image_generation=False, video_generation=False)
    # The fresh entry and the backup keep the previous good value: the failed
    # aggregate was never written over them.
    assert User._models_cache[key][1] == good
    assert User._models_cache_backup[key] == good
    assert User._models_failure_cooldowns[key] > time.monotonic()


async def test_all_failed_without_previous_value_raises_and_caches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider("openai", None)
    _install_providers(monkeypatch, [provider])
    user = User(id=1)

    with pytest.raises(ModelsUnavailableError, match="every provider failed"):
        await user.get_available_models()

    assert not User._models_cache
    assert not User._models_cache_backup


async def test_partial_failure_serves_subset_without_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    failing = _provider("openai", None)
    healthy = _provider("gemini", [_model("g1")])
    _install_providers(monkeypatch, [failing, healthy])
    user = User(id=1)

    result = await user.get_available_models()

    assert result == [_model("g1")]
    assert not User._models_cache
    assert not User._models_cache_backup

    # The partial result was not cached, so the next call re-fetches.
    healthy.get_available_models = AsyncMock(return_value=[_model("g2")])
    assert await user.get_available_models() == [_model("g2")]


async def test_cooldown_suppresses_provider_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    good = [_model("m1")]
    provider = _provider("openai", good)
    _install_providers(monkeypatch, [provider])
    user = User(id=1)

    assert await user.get_available_models() == good
    _expire_cache(user)

    # Total failure records the cooldown and serves the stale good value.
    provider.get_available_models = AsyncMock(return_value=None)
    assert await user.get_available_models() == good
    provider.get_available_models.assert_awaited_once()

    # Within the cooldown the providers are not re-fetched at all.
    assert await user.get_available_models() == good
    provider.get_available_models.assert_awaited_once()

    # After the cooldown expires the providers are retried.
    User._models_failure_cooldowns[user._models_cache_key(image_generation=False, video_generation=False)] = (
        time.monotonic() - 1
    )
    provider.get_available_models = AsyncMock(return_value=[_model("fresh")])
    assert await user.get_available_models() == [_model("fresh")]


async def test_cache_key_is_stable_across_user_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider("openai", [_model("m1")])
    _install_providers(monkeypatch, [provider])
    user_a = User(id=7)
    user_b = User(id=7)

    assert await user_a.get_available_models() == [_model("m1")]
    assert await user_b.get_available_models() == [_model("m1")]

    provider.get_available_models.assert_awaited_once()


async def test_set_api_key_invalidates_models_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = LocalStorage(storage_path=str(tmp_path))
    provider = _provider("openai", [_model("m1")])
    _install_providers(monkeypatch, [provider])

    user_a = User(id=5)
    assert await user_a.get_available_models() == [_model("m1")]
    provider.get_available_models.assert_awaited_once()

    await set_api_key.__wrapped__(db, user_id=5, api_key="key", provider_name="openai")

    key = user_a._models_cache_key(image_generation=False, video_generation=False)
    assert key not in User._models_cache
    assert key not in User._models_cache_backup
    assert key not in User._models_failure_cooldowns

    provider.get_available_models = AsyncMock(return_value=[_model("m1"), _model("m2")])
    user_b = await db.get_or_create_user(user_id=5)
    assert await user_b.get_available_models() == [_model("m1"), _model("m2")]
    provider.get_available_models.assert_awaited_once()
