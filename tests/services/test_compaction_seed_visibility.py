"""Unit tests for compaction seed visibility (forced marker, seed sections, prompt contract)."""

import re
from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# The tools package is imported before chibi.constants on purpose (same trick as
# tests/services/tools/test_say_ack_tools.py): it pulls in chibi.config first, which
# fully initializes chibi.constants. Otherwise the direct chibi.constants import
# re-enters the partially initialized chibi.config and fails on standalone collection.
import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.constants import get_llm_prompt
from chibi.interfaces.interface import UserInterface
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.context_measurement import SizeProvenance
from chibi.services.providers.tools.memory import SummarizeHistoryTool
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import (
    COMPACTION_QUESTION,
    COMPACTION_SEED_SOURCE,
    FORCED_COMPACTION_MARKER,
    build_compaction_seed,
    emergency_summarization,
)
from chibi.storage.local import LocalStorage

# Everything before the dynamic date placeholder: stable across marker instances.
MARKER_PREFIX = FORCED_COMPACTION_MARKER.split("{date}")[0]

SUMMARY_WITH_SECTIONS = (
    "## Summary\nDetailed summary of everything that was discussed.\n\n"
    "## Current work in progress\nRefactoring the parser, half done.\n\n"
    "## Background tasks\nNightly backup job at 03:00 UTC."
)


def _make_summarization_mocks(messages: list[Message], answer: str) -> tuple[MagicMock, MagicMock]:
    """Build a mock Database and a fake active provider for emergency_summarization."""
    user = MagicMock()
    user.id = 1
    user.get_active_llm_model = Mock(return_value="active-model")
    provider = MagicMock()
    provider.get_chat_response = AsyncMock(
        return_value=(
            ChatResponseSchema(answer=answer, provider="test", model="active-model", usage=UsageSchema()),
            [],
        )
    )
    user.get_active_llm_provider = Mock(return_value=provider)

    db = MagicMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=messages)
    db.add_message = AsyncMock()
    return db, provider


async def _run_emergency_summarization(db: MagicMock) -> AsyncMock:
    """Run emergency_summarization with the replacement helper mocked out.

    Returns the replacement mock, whose awaited kwargs carry the written seed.
    The measured pre figure is lifted above the seed's own count so the
    verify-before-wipe machine takes the success path and actually writes.
    """
    measured_total = 50000
    with (
        patch("chibi.services.user.replace_thread_history.__wrapped__", AsyncMock()) as replace_mock,
        patch("chibi.services.user.gpt_settings.max_tokens", 32000),
        patch("chibi.services.user.gpt_settings.max_history_tokens", 100000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(side_effect=[(measured_total, SizeProvenance.ESTIMATED)]),
        ),
    ):
        await emergency_summarization.__wrapped__(db, storage_id=1, thread_id=0, caller="reactive")
    return replace_mock


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    """Build a real LocalStorage backend rooted in a temp directory."""
    return LocalStorage(storage_path=str(tmp_path / "storage"))


@pytest.fixture
def patched_db_provider(storage: LocalStorage):
    """Route every @inject_database call to the temp LocalStorage."""
    fake_provider = SimpleNamespace(get_database=AsyncMock(return_value=storage))
    with patch("chibi.storage.database._db_provider", fake_provider):
        yield storage


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


def test_forced_seed_prepends_dated_marker() -> None:
    """forced=True prefixes the summary with the marker line carrying the current date."""
    seed = build_compaction_seed(summary="the summary body", forced=True)

    assert [message.role for message in seed] == ["user", "assistant"]
    assert all(message.source == COMPACTION_SEED_SOURCE for message in seed)
    assert seed[0].content == COMPACTION_QUESTION
    content = seed[1].content
    assert content.startswith(MARKER_PREFIX)
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", content)
    assert content.endswith("the summary body")


def test_tool_seed_has_no_marker() -> None:
    """The default (tool) seed carries no forced-compaction marker."""
    seed = build_compaction_seed(summary="agent-chosen summary")

    assert seed[1].content == "agent-chosen summary"
    assert "[SYSTEM]" not in seed[1].content


@pytest.mark.asyncio
async def test_tool_built_seed_has_no_forced_marker(patched_db_provider: LocalStorage) -> None:
    """The summarize_history tool path writes the summary verbatim, no forced marker."""
    _reset_usage_cache()
    storage_id, thread_id = 7, 0

    await SummarizeHistoryTool.function(
        summary=SUMMARY_WITH_SECTIONS,
        user_id=storage_id,
        interface=cast("UserInterface", SimpleNamespace(thread_id=thread_id)),
    )

    user = await patched_db_provider.get_or_create_user(user_id=storage_id)
    history = await patched_db_provider.get_conversation_messages(user=user, thread_id=thread_id)
    assert len(history) == 2
    assert history[1].content == SUMMARY_WITH_SECTIONS
    assert "[SYSTEM]" not in history[1].content


@pytest.mark.asyncio
async def test_emergency_seed_carries_marker_and_sections() -> None:
    """The emergency path writes a seed whose summary holds marker + detailed summary + sections."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages), answer=SUMMARY_WITH_SECTIONS)

    replace_mock = await _run_emergency_summarization(db)

    seed = replace_mock.await_args_list[0].kwargs["seed"]
    assert len(seed) == 2
    assert seed[0].content == COMPACTION_QUESTION
    assert all(message.source == COMPACTION_SEED_SOURCE for message in seed)
    content = seed[1].content
    assert content.startswith(MARKER_PREFIX)
    assert "; older turns were replaced by this summary" in content
    assert content.endswith(f"\n\n{SUMMARY_WITH_SECTIONS}")
    assert "## Summary" in content
    assert "## Current work in progress" in content
    assert "## Background tasks" in content


@pytest.mark.asyncio
async def test_summarizer_prompt_elicits_required_sections() -> None:
    """The inline summarizer system prompt demands the detailed/work/background structure."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, provider = _make_summarization_mocks(deepcopy(messages), answer=SUMMARY_WITH_SECTIONS)

    await _run_emergency_summarization(db)

    system_prompt = provider.get_chat_response.await_args.kwargs["system_prompt"]
    assert system_prompt.startswith("Summarize this conversation")
    assert "## Summary" in system_prompt
    assert "## Current work in progress" in system_prompt
    assert "## Background tasks" in system_prompt
    assert "current work" in system_prompt.lower()
    assert "background" in system_prompt.lower()


def test_base_prompt_documents_forced_compaction() -> None:
    """The context-management section of the base prompt teaches the forced-compaction contract."""
    prompt = get_llm_prompt(filesystem_access=False, allow_delegation=False)

    assert "[SYSTEM] Context was forcibly compacted on" in prompt
    assert "## Current work in progress" in prompt
    assert "## Background tasks" in prompt
    assert "continue the interrupted work coherently" in prompt
