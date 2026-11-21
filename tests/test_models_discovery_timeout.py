"""Per-provider timeout isolation for ``User.get_available_models``."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chibi.models import ModelsUnavailableError, User
from chibi.schemas.app import ModelChangeSchema


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


def _hanging_provider(name: str) -> Any:
    """Build a provider stand-in whose enumeration never returns on its own."""
    provider = SimpleNamespace(name=name)

    async def hang(image_generation: bool = False, video_generation: bool = False) -> list[ModelChangeSchema]:
        await asyncio.sleep(5)
        return []

    provider.get_available_models = hang
    return provider


def _install_providers(monkeypatch: pytest.MonkeyPatch, providers: list[Any]) -> None:
    registry = SimpleNamespace(available_instances=providers)
    monkeypatch.setattr(User, "providers", property(lambda self: registry))


@pytest.mark.asyncio
async def test_slow_provider_cannot_stall_the_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider exceeding the discovery cap is dropped and the successful subset survives."""
    fast = SimpleNamespace(name="fast")
    fast.get_available_models = AsyncMock(return_value=[_model("m1")])
    _install_providers(monkeypatch, [_hanging_provider("slow"), fast])
    monkeypatch.setattr("chibi.models.MODELS_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    user = User(id=1)

    result = await user.get_available_models()

    assert result == [_model("m1")]
    # A partial failure is served but never cached.
    assert User._models_cache == {}
    assert User._models_cache_backup == {}


@pytest.mark.asyncio
async def test_every_provider_timing_out_raises_when_no_backup_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every provider timed out and no previous good listing, the failure surfaces."""
    _install_providers(monkeypatch, [_hanging_provider("p1"), _hanging_provider("p2")])
    monkeypatch.setattr("chibi.models.MODELS_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    user = User(id=1)

    with pytest.raises(ModelsUnavailableError):
        await user.get_available_models()

    assert user._models_failure_cooldowns  # the outage cooldown engaged
