"""Tests for the post-delivery forced-compaction trigger.

The trigger lives at the shared answer flow
(``chibi.services.user.get_llm_chat_completion_answer`` -> ``_schedule_forced_compaction``
-> ``_maybe_run_forced_compaction``) and fires after the answer has been
delivered: it re-measures the thread and, once the measured context reaches
the point where the NEXT request would be trimmed by the pre-call guard,
unconditionally runs ``emergency_summarization``. These tests pin the trigger
conditions, the throttle state (in-flight flag, back-off) and the
exception-containment contract; the compaction machine itself is mocked.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chibi.config.gpt import gpt_settings
from chibi.exceptions import ContextLengthExceededError
from chibi.models import Message, User
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.context_measurement import SizeProvenance
from chibi.services.failover import CooldownStore, FailoverModelRegistry
from chibi.services.usage_cache import CompactionStateStore, UsageCacheStore
from chibi.services.user import (
    CompactionOutcome,
    CompactionResult,
    _maybe_run_forced_compaction,
    build_compaction_seed,
    get_llm_chat_completion_answer,
)
from chibi.storage.local import LocalStorage
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

UID = 931
THREAD = 0
WINDOW = 128000
RESERVATION = 32000
# The threshold is min(max_history_tokens, WINDOW); the default
# max_history_tokens (200000) exceeds WINDOW, so the cap is WINDOW.
THRESHOLD = WINDOW
FALLBACK_THRESHOLD = 5000


@pytest.fixture(autouse=True)
def _clean_stores() -> Iterator[None]:
    """Isolate the process-lifetime stores per test."""
    SingletonMeta._instances.pop(CompactionStateStore, None)
    UsageCacheStore()._data.clear()
    yield
    SingletonMeta._instances.pop(CompactionStateStore, None)
    UsageCacheStore()._data.clear()


def _history(messages: int = 4) -> list[Message]:
    """Build a plain user/assistant history of a small heuristic size.

    Args:
        messages: Number of messages to build.

    Returns:
        The message list, oldest first.
    """
    roles = ["user", "assistant"]
    return [Message(role=roles[index % 2], content=f"turn {index} " + "x" * 200) for index in range(messages)]


def _make_db(history: list[Message]) -> tuple[MagicMock, MagicMock]:
    """Build a database and user double for one trigger invocation.

    Args:
        history: The conversation history the database returns.

    Returns:
        The (db, user) doubles.
    """
    user = MagicMock()
    user.id = UID
    db = MagicMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=list(history))
    db.add_message = AsyncMock()
    return db, user


def _make_provider() -> AsyncMock:
    """Build the provider double whose reservation path yields the standard reservation.

    Returns:
        The provider double.
    """
    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=RESERVATION)
    return provider


async def _run_trigger(
    db: MagicMock,
    user: MagicMock,
    provider: AsyncMock,
    measured_tokens: int,
    window: int | None = WINDOW,
) -> None:
    """Run one post-delivery trigger check with the measurement stubbed.

    Args:
        db: The database double.
        user: The user double.
        provider: The provider double.
        measured_tokens: The measured context size the measurement entry point reports.
        window: The model window the lookup reports (None = unresolvable).
    """
    with (
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(measured_tokens, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.get_model_context_window", Mock(return_value=window)),
    ):
        await _maybe_run_forced_compaction(
            db=db,
            storage_id=UID,
            thread_id=THREAD,
            user=user,
            provider=provider,
            model="gpt-4o",
        )


async def test_fires_at_threshold_and_compacts_once() -> None:
    """A measured context AT the threshold fires the compaction exactly once per turn."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)

    compaction.assert_awaited_once()
    assert compaction.await_args is not None
    assert compaction.await_args.kwargs["storage_id"] == UID
    assert compaction.await_args.kwargs["thread_id"] == THREAD
    store = CompactionStateStore()
    assert not store.is_in_flight(user_id=UID, thread_id=THREAD)
    assert not store.is_backed_off(user_id=UID, thread_id=THREAD)


async def test_does_not_fire_below_threshold() -> None:
    """A measured context one token below the threshold never compacts."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD - 1)

    compaction.assert_not_awaited()
    store = CompactionStateStore()
    assert not store.is_in_flight(user_id=UID, thread_id=THREAD)


async def test_fallback_threshold_when_window_unresolvable() -> None:
    """Without a resolvable window the trigger threshold falls back to max_history_tokens."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with (
        patch.object(gpt_settings, "max_history_tokens", FALLBACK_THRESHOLD),
        patch("chibi.services.user.emergency_summarization", new=compaction),
    ):
        await _run_trigger(db, user, provider, measured_tokens=FALLBACK_THRESHOLD, window=None)
        compaction.assert_awaited_once()

        compaction.reset_mock()
        await _run_trigger(db, user, provider, measured_tokens=FALLBACK_THRESHOLD - 1, window=None)
        compaction.assert_not_awaited()


async def test_skips_when_history_is_solely_the_seed() -> None:
    """A thread consisting SOLELY of compaction-seed messages is never re-compacted."""
    seed = build_compaction_seed(summary="collapsed summary")
    db, user = _make_db(seed)
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD + 1000)

    compaction.assert_not_awaited()
    assert not CompactionStateStore().is_in_flight(user_id=UID, thread_id=THREAD)


async def test_backoff_after_failed_attempt_expires_after_three_turns() -> None:
    """A failed compaction backs off; the trigger fires again on the fourth turn."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(side_effect=RuntimeError("summarizer exploded"))

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)
        store = CompactionStateStore()
        assert compaction.await_count == 1
        assert not store.is_in_flight(user_id=UID, thread_id=THREAD)
        assert store.is_backed_off(user_id=UID, thread_id=THREAD)

        # Turn 2 and turn 3 are still inside the back-off.
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)
        assert compaction.await_count == 1

        # Turn 3 turns after the failure: the back-off expired either way.
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)
        assert compaction.await_count == 2


async def test_backoff_expires_after_thirty_minutes_whichever_first() -> None:
    """The back-off expires as soon as EITHER condition passes: turns OR wall-clock time."""
    store = CompactionStateStore()
    store.record_turn(user_id=UID, thread_id=THREAD)
    store.record_failure(user_id=UID, thread_id=THREAD)
    assert store.is_backed_off(user_id=UID, thread_id=THREAD)

    # Thirty minutes and one second elapsed: the time condition expires first (no turns passed).
    key = store._make_key(user_id=UID, thread_id=THREAD)
    store._failure_monotonic[key] -= 30 * 60 + 1
    assert not store.is_backed_off(user_id=UID, thread_id=THREAD)


@pytest.mark.parametrize(
    "result",
    [
        None,
        CompactionResult(outcome=CompactionOutcome.DID_NOT_REDUCE),
        CompactionResult(outcome=CompactionOutcome.REDUCED_BUT_OVER_BUDGET),
    ],
)
async def test_unsuccessful_outcomes_record_backoff(result: CompactionResult | None) -> None:
    """Every non-compacting outcome (no attempt, no reduction, still over budget) backs off."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=result)

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)

    compaction.assert_awaited_once()
    assert CompactionStateStore().is_backed_off(user_id=UID, thread_id=THREAD)


async def test_in_flight_flag_set_before_the_call_and_cleared_after() -> None:
    """The per-thread flag is set while the summarizer runs and cleared afterwards."""
    db, user = _make_db(_history())
    provider = _make_provider()
    store = CompactionStateStore()
    seen_during_call: list[bool] = []

    async def _compaction(**kwargs: Any) -> CompactionResult:
        seen_during_call.append(store.is_in_flight(user_id=UID, thread_id=THREAD))
        return CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS)

    with patch("chibi.services.user.emergency_summarization", new=_compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)

    assert seen_during_call == [True]
    assert not store.is_in_flight(user_id=UID, thread_id=THREAD)


async def test_in_flight_flag_cleared_on_failure() -> None:
    """A crashing summarizer call also clears the in-flight flag."""
    db, user = _make_db(_history())
    provider = _make_provider()
    store = CompactionStateStore()

    async def _failing_compaction(**kwargs: Any) -> CompactionResult:
        assert store.is_in_flight(user_id=UID, thread_id=THREAD)
        raise RuntimeError("summarizer exploded")

    with patch("chibi.services.user.emergency_summarization", new=_failing_compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)

    assert not store.is_in_flight(user_id=UID, thread_id=THREAD)


async def test_exception_containment_never_breaks_the_turn() -> None:
    """A crash anywhere in the check is contained: nothing propagates to the caller."""
    db, user = _make_db(_history())
    db.get_conversation_messages = AsyncMock(side_effect=RuntimeError("storage is on fire"))
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with patch("chibi.services.user.emergency_summarization", new=compaction):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD)

    compaction.assert_not_awaited()
    assert not CompactionStateStore().is_in_flight(user_id=UID, thread_id=THREAD)


async def test_measurement_failure_is_contained() -> None:
    """A crashing measurement entry point is contained the same way."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with (
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(side_effect=RuntimeError("tokenizer exploded")),
        ),
        patch("chibi.services.user.get_model_context_window", Mock(return_value=WINDOW)),
        patch("chibi.services.user.emergency_summarization", new=compaction),
    ):
        await _maybe_run_forced_compaction(
            db=db,
            storage_id=UID,
            thread_id=THREAD,
            user=user,
            provider=provider,
            model="gpt-4o",
        )

    compaction.assert_not_awaited()


async def test_config_switch_off_disables_the_trigger() -> None:
    """With CONTEXT_FORCED_COMPACTION_ENABLED off the check is a no-op."""
    db, user = _make_db(_history())
    provider = _make_provider()
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))

    with (
        patch.object(gpt_settings, "context_forced_compaction_enabled", False),
        patch("chibi.services.user.emergency_summarization", new=compaction),
    ):
        await _run_trigger(db, user, provider, measured_tokens=THRESHOLD + 100000)

    compaction.assert_not_awaited()


@pytest.fixture
def local_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> LocalStorage:
    """Provide a real local storage backend and route every injected call to it.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture restoring the patched provider afterwards.

    Returns:
        The storage the injected helpers now see.
    """
    storage = LocalStorage(storage_path=str(tmp_path / "storage"))
    provider = SimpleNamespace(get_database=AsyncMock(return_value=storage))
    monkeypatch.setattr("chibi.storage.database._db_provider", provider)
    return storage


def _full_path_interface(storage_id: int, thread_id: int = 0) -> Any:
    """Build the minimal interface the full answer path needs.

    Args:
        storage_id: The storage key of the conversation.
        thread_id: The message thread identifier.

    Returns:
        The interface double.
    """
    interface = Mock()
    interface.storage_id = storage_id
    interface.thread_id = thread_id
    interface.user_data = {"first_name": "Tester"}
    interface.editor_context = None
    return interface


def _answer_provider(answer: str = "ok") -> AsyncMock:
    """Build the provider double the failover-guarded call resolves.

    Args:
        answer: The answer text the provider returns.

    Returns:
        The provider double.
    """
    provider = AsyncMock()
    provider.name = "openai"
    provider._get_max_tokens_value = Mock(return_value=RESERVATION)
    provider.get_chat_response = AsyncMock(
        return_value=(
            ChatResponseSchema(
                answer=answer,
                provider="openai",
                model="gpt-4o",
                usage=UsageSchema(prompt_tokens=10, completion_tokens=1, total_tokens=11),
            ),
            [Message(role="assistant", content=answer)],
        )
    )
    return provider


async def test_answer_flow_schedules_the_hook_and_never_waits_for_compaction(
    local_db: LocalStorage,
) -> None:
    """The answer returns while the compaction is still running in the background."""
    SingletonMeta._instances.pop(CooldownStore, None)
    SingletonMeta._instances.pop(FailoverModelRegistry, None)
    storage = local_db
    history = _history()
    user = await storage.get_or_create_user(user_id=UID)
    user.messages = list(history)
    await storage.save_user(user)

    provider = _answer_provider()
    release = asyncio.Event()
    background_tasks: list[asyncio.Task[None]] = []

    async def _blocking_compaction(**kwargs: Any) -> CompactionResult:
        await release.wait()
        return CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS)

    def _fake_schedule(db: Any, storage_id: int, thread_id: int, user: Any, provider: Any, model: Any) -> None:
        background_tasks.append(
            asyncio.create_task(
                _maybe_run_forced_compaction(
                    db=db,
                    storage_id=storage_id,
                    thread_id=thread_id,
                    user=user,
                    provider=provider,
                    model=model,
                )
            )
        )

    with (
        patch.object(User, "get_active_llm_provider", MagicMock(return_value=provider)),
        patch.object(User, "get_active_llm_model", MagicMock(return_value="gpt-4o")),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(THRESHOLD, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.get_model_context_window", Mock(return_value=WINDOW)),
        patch("chibi.services.user.emergency_summarization", new=_blocking_compaction),
        patch("chibi.services.user._schedule_forced_compaction", new=_fake_schedule),
    ):
        response = await asyncio.wait_for(
            get_llm_chat_completion_answer.__wrapped__(
                storage,
                storage_id=UID,
                interface=_full_path_interface(UID),
                user_text_message="hello",
            ),
            timeout=5,
        )

    # The answer was delivered while the compaction was still blocked.
    assert response.answer == "ok"
    assert not release.is_set()
    assert len(background_tasks) == 1

    release.set()
    await asyncio.gather(*background_tasks)
    SingletonMeta._instances.pop(CooldownStore, None)
    SingletonMeta._instances.pop(FailoverModelRegistry, None)


async def test_answer_flow_does_not_schedule_the_hook_on_the_recovery_path(
    local_db: LocalStorage,
) -> None:
    """A turn that goes through the reactive recovery never schedules the post-delivery check."""
    SingletonMeta._instances.pop(CooldownStore, None)
    SingletonMeta._instances.pop(FailoverModelRegistry, None)
    storage = local_db
    user = await storage.get_or_create_user(user_id=UID)
    user.messages = _history()
    await storage.save_user(user)

    provider = _answer_provider()

    async def _overflow(**kwargs: Any) -> Any:
        raise ContextLengthExceededError(provider="openai", model="gpt-4o")

    provider.get_chat_response = AsyncMock(side_effect=_overflow)
    schedule = Mock()
    recovered = ChatResponseSchema(answer="recovered", provider="openai", model="gpt-4o", usage=None)
    recovery = AsyncMock(return_value=recovered)
    # The answer flow calls the recovery through ``__wrapped__``; a bare AsyncMock
    # hides dunder attributes, so the attribute is set explicitly.
    recovery.__wrapped__ = AsyncMock(return_value=recovered)

    with (
        patch.object(User, "get_active_llm_provider", MagicMock(return_value=provider)),
        patch.object(User, "get_active_llm_model", MagicMock(return_value="gpt-4o")),
        patch("chibi.services.user.get_model_context_window", Mock(return_value=WINDOW)),
        patch("chibi.services.user._reactive_context_recovery", new=recovery),
        patch("chibi.services.user._schedule_forced_compaction", new=schedule),
    ):
        response = await get_llm_chat_completion_answer.__wrapped__(
            storage,
            storage_id=UID,
            interface=_full_path_interface(UID),
            user_text_message="hello",
        )

    assert response.answer == "recovered"
    schedule.assert_not_called()
    SingletonMeta._instances.pop(CooldownStore, None)
    SingletonMeta._instances.pop(FailoverModelRegistry, None)
