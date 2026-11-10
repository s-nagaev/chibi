"""Tests for FailoverModelRegistry handling of failed model enumerations."""

import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chibi.schemas.app import ModelChangeSchema
from chibi.services.failover import FailoverModelRegistry
from chibi.utils.app import SingletonMeta


def _pop_registry_singleton() -> None:
    SingletonMeta._instances.pop(FailoverModelRegistry, None)


@pytest.fixture(autouse=True)
def _clean_registry_singleton():
    _pop_registry_singleton()
    yield
    _pop_registry_singleton()


def _model(name: str) -> ModelChangeSchema:
    return ModelChangeSchema(provider="Test", name=name, display_name=name, image_generation=False)


def _provider_double(name: str = "Test", default_model: str | None = "test-default") -> Any:
    provider = AsyncMock()
    provider.name = name
    if default_model is not None:
        provider.default_model = default_model
    else:
        del provider.default_model
    return provider


async def test_none_from_provider_is_cached_as_failure_marker() -> None:
    provider = _provider_double()
    provider.get_available_models = AsyncMock(return_value=None)
    registry = FailoverModelRegistry()

    result = await registry.models_for(provider)

    assert result is None
    expires_at, cached = registry._cache["test"]
    assert cached is None
    assert 0 < expires_at - time.monotonic() <= registry._FAILURE_TTL_SECONDS


async def test_stale_previous_good_map_survives_failed_enumeration() -> None:
    provider = _provider_double()
    provider.get_available_models = AsyncMock(return_value=[_model("test-default")])
    registry = FailoverModelRegistry()
    good = await registry.models_for(provider)
    assert good == {"test-default"}

    # Expire the good entry so the next call re-enumerates and fails.
    registry._cache["test"] = (time.monotonic() - 1, {"test-default"})
    provider.get_available_models = AsyncMock(return_value=None)
    stale = await registry.models_for(provider)

    assert stale == {"test-default"}
    expires_at, cached = registry._cache["test"]
    assert cached == {"test-default"}
    assert 0 < expires_at - time.monotonic() <= registry._FAILURE_TTL_SECONDS


async def test_empty_listing_with_declared_default_model_is_treated_as_failure() -> None:
    provider = _provider_double()
    provider.get_available_models = AsyncMock(return_value=[_model("test-default")])
    registry = FailoverModelRegistry()
    assert await registry.models_for(provider) == {"test-default"}

    registry._cache["test"] = (time.monotonic() - 1, {"test-default"})
    provider.get_available_models = AsyncMock(return_value=[])
    stale = await registry.models_for(provider)

    assert stale == {"test-default"}


async def test_empty_listing_without_declared_default_model_is_a_valid_empty_answer() -> None:
    provider = _provider_double(default_model=None)
    provider.get_available_models = AsyncMock(return_value=[])
    registry = FailoverModelRegistry()

    result = await registry.models_for(provider)

    assert result == set()


async def test_successful_enumeration_updates_the_entry() -> None:
    provider = _provider_double()
    provider.get_available_models = AsyncMock(return_value=[_model("m1"), _model("m2")])
    registry = FailoverModelRegistry()

    result = await registry.models_for(provider)

    assert result == {"m1", "m2"}
    entry = registry._cache["test"]
    assert entry[1] == {"m1", "m2"}


async def test_failure_marker_expires_and_registry_retries() -> None:
    provider = _provider_double()
    provider.get_available_models = AsyncMock(return_value=None)
    registry = FailoverModelRegistry()
    assert await registry.models_for(provider) is None

    expires_at, _ = registry._cache["test"]
    registry._cache["test"] = (time.monotonic() - 1, None)
    provider.get_available_models = AsyncMock(return_value=[_model("fresh")])

    result = await registry.models_for(provider)

    assert result == {"fresh"}
