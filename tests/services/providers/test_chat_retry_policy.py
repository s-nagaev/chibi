"""Retry-policy tests for the shared OpenAI-compatible chat path.

``OpenAIFriendlyProvider.get_chat_response`` must retry only genuine
transport-level failures and let application-level outcomes (the silent-turn
error among them) reach the caller after a single attempt. The tenacity wait
is zeroed in every test: the policy is exercised through attempt counts, never
by waiting through the real 30-180 s backoff.
"""

from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from tenacity import wait_fixed

from chibi.exceptions import ServiceConnectionError, SilentTurnError
from chibi.models import Message, User
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.provider import OpenAIFriendlyProvider

TEST_TOKEN = "test-token-for-retry-policy"
TEST_MODEL = "test-model"


def _zero_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real 30-180 s exponential wait of the shared chat path with a no-op.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    retryer = getattr(OpenAIFriendlyProvider.get_chat_response, "retry")
    monkeypatch.setattr(retryer, "wait", wait_fixed(0))


def _patch_chat_call(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> dict[str, Any]:
    """Make the underlying chat completion call of a ``MoonshotAI`` fail with ``error``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        error: Exception to raise on every call.

    Returns:
        A dict whose ``count`` key accumulates the number of attempts made.
    """
    calls: dict[str, Any] = {"count": 0}

    async def failing_chat_call(self: MoonshotAI, **kwargs: Any) -> Any:
        calls["count"] += 1
        raise error

    monkeypatch.setattr(MoonshotAI, "_get_chat_completion_response", failing_chat_call)
    return calls


async def _call_get_chat_response(provider: MoonshotAI) -> None:
    """Run one chat turn against the provider, ignoring the outcome.

    Args:
        provider: Provider under test.
    """
    try:
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
        )
    except Exception:
        pass


async def test_silent_turn_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent-turn outcome must cost exactly one attempt, never a retry cycle."""
    _zero_retry_wait(monkeypatch)
    error = SilentTurnError(provider=MoonshotAI.name, model=TEST_MODEL, detail="turn misuse")
    calls = _patch_chat_call(monkeypatch, error)
    provider = MoonshotAI(token=TEST_TOKEN)

    await _call_get_chat_response(provider)

    assert calls["count"] == 1, f"SilentTurnError was retried: {calls['count']} attempts"


async def test_silent_turn_error_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The silent-turn error must reach the caller as the same instance, not wrapped."""
    _zero_retry_wait(monkeypatch)
    error = SilentTurnError(provider=MoonshotAI.name, model=TEST_MODEL, detail="turn misuse")
    _patch_chat_call(monkeypatch, error)
    provider = MoonshotAI(token=TEST_TOKEN)

    with pytest.raises(SilentTurnError) as excinfo:
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
        )

    assert excinfo.value is error
    assert excinfo.value.detail == "turn misuse"


async def test_transport_error_is_retried_with_existing_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine transport failure must still be retried four times, then re-raised."""
    _zero_retry_wait(monkeypatch)
    raw_error = APIConnectionError(request=httpx.Request("POST", "https://api.test.invalid/v1"))
    calls = _patch_chat_call(monkeypatch, raw_error)
    provider = MoonshotAI(token=TEST_TOKEN)

    with pytest.raises(ServiceConnectionError):
        await provider.get_chat_response(
            messages=[Message(role="user", content="Hi")],
            user=User(id=1),
            caller_storage_id=1,
            caller_thread_id=0,
            model=TEST_MODEL,
        )

    assert calls["count"] == 4, f"transport error retry count changed: {calls['count']} attempts"
