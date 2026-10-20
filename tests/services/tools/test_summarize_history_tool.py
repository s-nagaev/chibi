"""Tests for the model-facing ``summarize_history`` tool (real compaction)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from chibi.models import Message
from chibi.services.providers.tools.memory import SummarizeHistoryTool
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import COMPACTION_SEED_SOURCE
from chibi.storage.local import LocalStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Build a real LocalStorage backend rooted in a temp directory."""
    return LocalStorage(storage_path=str(tmp_path / "storage"))


@pytest.fixture
def patched_db_provider(storage: LocalStorage):
    """Route every @inject_database call to the temp LocalStorage."""
    fake_provider = SimpleNamespace(get_database=AsyncMock(return_value=storage))
    with patch("chibi.storage.database._db_provider", fake_provider):
        yield storage


def _interface(thread_id: int = 0) -> Any:
    """Build a minimal UserInterface stand-in for the tool call."""
    return SimpleNamespace(thread_id=thread_id)


async def _append_message(storage: LocalStorage, storage_id: int, thread_id: int, message: Message) -> None:
    """Append one message the way the turn pipeline commits history."""
    user = await storage.get_or_create_user(user_id=storage_id)
    await storage.add_message(user=user, message=message, ttl=3600, thread_id=thread_id)


@pytest.mark.asyncio
async def test_tool_writes_the_two_message_seed(patched_db_provider: LocalStorage) -> None:
    """The tool replaces the whole history with the same 2-message seed the automatic path writes."""
    _reset_usage_cache()
    storage_id, thread_id = 1, 0
    await _append_message(patched_db_provider, storage_id, thread_id, Message(role="user", content="old question"))
    await _append_message(patched_db_provider, storage_id, thread_id, Message(role="assistant", content="old answer"))

    result = await SummarizeHistoryTool.function(
        summary="An exhaustive summary of the whole dialog.",
        user_id=storage_id,
        interface=_interface(thread_id),
    )

    assert result == {"status": "ok"}
    user = await patched_db_provider.get_or_create_user(user_id=storage_id)
    history = await patched_db_provider.get_conversation_messages(user=user, thread_id=thread_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "What we were talking about?"
    assert history[0].source == COMPACTION_SEED_SOURCE
    assert history[1].role == "assistant"
    assert history[1].content == "An exhaustive summary of the whole dialog."
    assert history[1].source == COMPACTION_SEED_SOURCE


@pytest.mark.asyncio
async def test_mid_turn_commit_stores_seed_first_then_the_turn(patched_db_provider: LocalStorage) -> None:
    """INTENDED BEHAVIOUR, DO NOT "FIX": the seed is stored first, the turn's own messages after.

    The tool runs MID-TURN. The turn's own messages (the user question, the
    tool call, the tool result, the answer) are committed by the turn
    pipeline AFTER the tool returns. So the stored history ends up as the
    seed followed by the current turn's messages. That is the intended
    outcome, NOT the old wipe-then-commit bug: the old bug was that the
    summary itself was never persisted; here the seed (carrying the summary
    as its assistant message) is persisted by the replacement helper, and
    the turn is appended afterwards by the normal pipeline. A history that
    ends with anything OTHER than the turn's messages after the seed, or a
    history where the seed does not come first, is a regression.
    """
    _reset_usage_cache()
    storage_id, thread_id = 2, 3
    await _append_message(patched_db_provider, storage_id, thread_id, Message(role="user", content="old question"))

    await SummarizeHistoryTool.function(
        summary="The summary of everything before this turn.",
        user_id=storage_id,
        interface=_interface(thread_id),
    )

    # The turn pipeline commits the turn's own messages after the tool returns.
    await _append_message(patched_db_provider, storage_id, thread_id, Message(role="user", content="current question"))
    await _append_message(
        patched_db_provider, storage_id, thread_id, Message(role="assistant", content="current answer")
    )

    user = await patched_db_provider.get_or_create_user(user_id=storage_id)
    history = await patched_db_provider.get_conversation_messages(user=user, thread_id=thread_id)
    assert [(msg.role, msg.content) for msg in history] == [
        ("user", "What we were talking about?"),
        ("assistant", "The summary of everything before this turn."),
        ("user", "current question"),
        ("assistant", "current answer"),
    ]
    assert all(msg.source == COMPACTION_SEED_SOURCE for msg in history[:2])
    assert all(msg.source is None for msg in history[2:])


@pytest.mark.asyncio
async def test_tool_invalidates_usage_cache(patched_db_provider: LocalStorage) -> None:
    """The invalidation happens inside the shared replacement helper, not a second mechanism."""
    _reset_usage_cache()
    storage_id, thread_id = 4, 5
    store = UsageCacheStore()
    store._data[store._make_key(user_id=storage_id, thread_id=thread_id)] = 99999

    await SummarizeHistoryTool.function(
        summary="A summary.",
        user_id=storage_id,
        interface=_interface(thread_id),
    )

    assert store.get(user_id=storage_id, thread_id=thread_id) is None


def test_summary_parameter_schema_is_unchanged() -> None:
    """The tool keeps a single required ``summary`` string parameter, nothing else.

    ``run_in_background`` is not the tool's own parameter: the ChibiTool base
    merges it into every tool's schema (``add_global_params``), so the
    model-declared surface of this tool is exactly ``summary``.
    """
    function = cast("dict[str, Any]", SummarizeHistoryTool.definition["function"])
    assert function["name"] == "summarize_history"
    parameters = cast("dict[str, Any]", function["parameters"])
    assert parameters["required"] == ["summary"]
    assert set(cast("dict[str, Any]", parameters["properties"]).keys()) == {"summary", "run_in_background"}
    assert parameters["properties"]["summary"]["type"] == "string"


@pytest.mark.asyncio
async def test_tool_raises_without_user_id() -> None:
    """Missing automatically-provided user_id is a tool error."""
    from chibi.services.providers.tools.exceptions import ToolException

    with pytest.raises(ToolException):
        await SummarizeHistoryTool.function(
            summary="A summary.",
            user_id=None,
            interface=_interface(0),
        )


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()
