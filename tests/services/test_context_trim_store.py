"""Unit tests for the context guard trim side-channel (plan Design decision 7).

The guards shrink an oversized request transiently and silently; the
side-channel store records what they removed, and the volatile context note
discloses it. These tests pin: the store's record/consume semantics, the
write side of both guards (real last-resort shrink, no mocking of the fit),
and the read side of ``collect_volatile_context_note``, including one run
through the chat-completions provider entry point.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.models import Message, User
from chibi.services.context_measurement import SizeProvenance, estimate_messages_tokens
from chibi.services.context_note import CONTEXT_NOTE_CLOSE, CONTEXT_NOTE_OPEN, strip_context_note
from chibi.services.context_trim_store import ContextTrimStore, TrimStats
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.utils import collect_volatile_context_note
from chibi.services.user import _complete_units, _enforce_context_budget, enforce_in_loop_budget
from chibi.utils.app import SingletonMeta

TEST_MODEL = "kimi-k2.6"
TEST_TOKEN = "test-token-for-trim-store"


def _reset_singletons() -> None:
    """Drop the trim-store singleton so a test never reuses another test's state."""
    SingletonMeta._instances.pop(ContextTrimStore, None)


def _make_guard_provider(max_tokens: int) -> MagicMock:
    """Build a provider double whose output-cap resolution returns a concrete integer."""
    provider = MagicMock()
    provider._get_max_tokens_value = MagicMock(return_value=max_tokens)
    return provider


def _dialogue(chars: int, turns: int) -> list[Message]:
    """Build a plain user/assistant history of a controlled heuristic size."""
    messages: list[Message] = []
    for index in range(turns):
        messages.append(Message(role="user", content=f"turn {index} " + "u" * chars))
        messages.append(Message(role="assistant", content=f"answer {index} " + "a" * chars))
    return messages


# --- Store semantics -------------------------------------------------------------


def test_record_accumulates_and_zero_events_are_ignored() -> None:
    """Several guard events accumulate; a zero-valued event records nothing."""
    _reset_singletons()
    store = ContextTrimStore()

    store.record(storage_id=1, thread_id=0, dropped_units=2, dropped_messages=2, truncated_messages=1)
    store.record(storage_id=1, thread_id=0, dropped_units=1, dropped_messages=3, truncated_messages=0)
    # A shrink that changed nothing (its internal error paths) records nothing.
    store.record(storage_id=1, thread_id=0, dropped_units=0, dropped_messages=0, truncated_messages=0)

    stats = store.consume(storage_id=1, thread_id=0)
    assert stats == TrimStats(dropped_units=3, dropped_messages=5, truncated_messages=1)


def test_consume_is_read_once_and_absent_key_returns_none() -> None:
    """A consumed record is gone, and an unknown key reads as None."""
    _reset_singletons()
    store = ContextTrimStore()
    store.record(storage_id=1, thread_id=7, dropped_units=1, dropped_messages=1, truncated_messages=0)

    first = store.consume(storage_id=1, thread_id=7)
    assert first is not None
    assert first.dropped_units == 1
    assert store.consume(storage_id=1, thread_id=7) is None
    assert store.consume(storage_id=1, thread_id=8) is None


def test_keys_are_isolated_per_conversation() -> None:
    """Two conversations never see each other's trims."""
    _reset_singletons()
    store = ContextTrimStore()
    store.record(storage_id=1, thread_id=0, dropped_units=1, dropped_messages=1, truncated_messages=0)
    store.record(storage_id=2, thread_id=0, dropped_units=5, dropped_messages=5, truncated_messages=2)

    first = store.consume(storage_id=1, thread_id=0)
    second = store.consume(storage_id=2, thread_id=0)
    assert first == TrimStats(dropped_units=1, dropped_messages=1, truncated_messages=0)
    assert second == TrimStats(dropped_units=5, dropped_messages=5, truncated_messages=2)


def test_clear_drops_the_record_without_reading() -> None:
    """clear() removes any accumulated stats and tolerates an absent key."""
    _reset_singletons()
    store = ContextTrimStore()
    store.record(storage_id=1, thread_id=0, dropped_units=1, dropped_messages=1, truncated_messages=0)
    store.clear(storage_id=1, thread_id=0)
    store.clear(storage_id=1, thread_id=0)
    assert store.consume(storage_id=1, thread_id=0) is None


def test_trim_stats_is_empty_property() -> None:
    """is_empty tells a no-op bundle apart from a real trim."""
    assert TrimStats().is_empty is True
    assert TrimStats(dropped_units=0, dropped_messages=0, truncated_messages=1).is_empty is False


# --- Guard write side ------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_guard_records_drops_and_truncations() -> None:
    """The pre-call guard records real drop and truncation counts of its shrink."""
    _reset_singletons()
    # One droppable old unit, one unit too large to drop whole (it overshoots
    # the remaining excess, so the truncation step shortens it), and the
    # protected last message.
    old = Message(role="user", content="o" * 2000)
    huge = Message(role="assistant", content="h" * 8000)
    last = Message(role="user", content="last turn")
    messages = [old, huge, last]

    weights = {id(msg): estimate_messages_tokens([msg]) for msg in messages}
    measured_total = sum(weights.values())
    reservation = 200
    # Budget leaves room for the last message and about half of the huge one:
    # dropping the old unit alone cannot close the gap, the huge unit
    # overshoots it whole, so the fit drops the old unit and truncates the
    # huge one.
    budget = weights[id(last)] + weights[id(huge)] // 2
    window = budget + reservation

    with (
        patch("chibi.services.user.get_model_context_window", return_value=window),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(measured_total, SizeProvenance.ESTIMATED)),
        ),
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(reservation),
            model=TEST_MODEL,
            storage_id=42,
            thread_id=3,
            messages=messages,
        )

    assert old not in result
    assert huge in result and last in result
    assert len(huge.content) < 8000  # the truncation step ran

    stats = ContextTrimStore().consume(storage_id=42, thread_id=3)
    assert stats is not None
    assert stats.dropped_units == 1
    assert stats.dropped_messages == 1
    assert stats.truncated_messages == 1
    # Another conversation stays untouched.
    assert ContextTrimStore().consume(storage_id=42, thread_id=4) is None


@pytest.mark.asyncio
async def test_pre_call_guard_records_nothing_when_request_fits() -> None:
    """A request that fits is sent unchanged and no trim event is recorded."""
    _reset_singletons()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED)),
        ),
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model=TEST_MODEL,
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    assert result is messages
    assert ContextTrimStore().consume(storage_id=1, thread_id=0) is None


@pytest.mark.asyncio
async def test_pre_call_guard_records_nothing_when_window_unknown() -> None:
    """Without a resolvable window the guard guesses nothing and records nothing."""
    _reset_singletons()
    messages = [Message(role="user", content="hello")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=None),
        patch("chibi.services.user.measure_request_size_with_provenance", new=AsyncMock()) as mock_measure,
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="unknown-model",
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    assert result is messages
    mock_measure.assert_not_awaited()
    assert ContextTrimStore().consume(storage_id=1, thread_id=0) is None


@pytest.mark.asyncio
async def test_in_loop_guard_records_drop_only_trims() -> None:
    """The in-loop guard records its whole-unit drops; it never truncates."""
    _reset_singletons()
    # protected_first=True: the first and the last unit survive, the middle
    # dialogue units are dropped until the request fits.
    messages = [
        Message(role="user", content="first"),
        *_dialogue(chars=2000, turns=3),
        Message(role="assistant", content="last"),
    ]
    units = _complete_units(messages)
    assert len(units) > 3

    weights = {id(msg): estimate_messages_tokens([msg]) for msg in messages}
    measured_total = sum(weights.values())
    reservation = 200
    # Budget keeps the protected head and tail units only.
    budget = weights[id(messages[0])] + weights[id(messages[-1])] + 10
    window = budget + reservation

    with (
        patch("chibi.services.user.get_model_context_window", return_value=window),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(measured_total, SizeProvenance.ESTIMATED)),
        ),
    ):
        keep_mask = await enforce_in_loop_budget(
            provider=_make_guard_provider(reservation),
            model=TEST_MODEL,
            storage_id=7,
            thread_id=2,
            conversation_messages=messages,
        )

    kept = [msg for msg, keep in zip(messages, keep_mask) if keep]
    assert messages[0] in kept and messages[-1] in kept
    assert len(kept) < len(messages)

    stats = ContextTrimStore().consume(storage_id=7, thread_id=2)
    assert stats is not None
    assert stats.dropped_units == len(units) - 2
    assert stats.dropped_messages == len(messages) - 2
    # Drop-only mode: no content truncation is ever recorded in-loop.
    assert stats.truncated_messages == 0


@pytest.mark.asyncio
async def test_in_loop_guard_records_nothing_when_request_fits() -> None:
    """An in-loop request that fits keeps every unit and records nothing."""
    _reset_singletons()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED)),
        ),
    ):
        keep_mask = await enforce_in_loop_budget(
            provider=_make_guard_provider(200),
            model=TEST_MODEL,
            storage_id=1,
            thread_id=0,
            conversation_messages=messages,
        )

    assert keep_mask == [True, True]
    assert ContextTrimStore().consume(storage_id=1, thread_id=0) is None


# --- Collector read side ---------------------------------------------------------


def _interface(thread_id: int = 0) -> Any:
    """Build a minimal interface stub for the collector tests."""
    return cast(Any, SimpleNamespace(thread_id=thread_id, uses_uploaded_file_storage=False))


def test_collector_appends_guard_trim_line_exactly_when_trimmed() -> None:
    """A recorded trim produces the warning line, and the read-once consume keeps it off the next note."""
    _reset_singletons()
    ContextTrimStore().record(storage_id=1, thread_id=5, dropped_units=2, dropped_messages=3, truncated_messages=1)
    interface = _interface(thread_id=5)

    note = collect_volatile_context_note(user_id=1, interface=interface)

    assert (
        "guard_trim_warning: 2 oldest unit(s) were hidden from this request (3 messages in total); "
        "1 message(s) truncated" in note
    )
    # The line travels inside the delimited note block only.
    assert note.startswith(CONTEXT_NOTE_OPEN) and note.endswith(CONTEXT_NOTE_CLOSE)
    # Consumed once: the following request carries no guard-trim line.
    second_note = collect_volatile_context_note(user_id=1, interface=interface)
    assert "guard_trim_warning" not in second_note
    assert "approximate_context_size: n/a" in second_note


def test_collector_truncation_only_trim_names_no_dropped_clause() -> None:
    """A trim that only truncated content names the truncations without a drop clause."""
    _reset_singletons()
    ContextTrimStore().record(storage_id=1, thread_id=0, dropped_units=0, dropped_messages=0, truncated_messages=2)

    note = collect_volatile_context_note(user_id=1, interface=_interface())

    assert "guard_trim_warning: 2 message(s) truncated" in note
    assert "hidden from this request" not in note


def test_collector_silent_without_trim() -> None:
    """No recorded trim, no line: the note carries only the usual keys."""
    _reset_singletons()
    note = collect_volatile_context_note(user_id=1, interface=_interface(thread_id=9))
    assert "guard_trim_warning" not in note


def test_collector_without_interface_leaves_the_record_for_the_next_request() -> None:
    """Interface-less requests (sub-agents) carry no note and do not consume the record."""
    _reset_singletons()
    ContextTrimStore().record(storage_id=1, thread_id=0, dropped_units=1, dropped_messages=1, truncated_messages=0)

    assert collect_volatile_context_note(user_id=1, interface=None) == ""
    assert ContextTrimStore().consume(storage_id=1, thread_id=0) is not None


def test_guard_trim_line_is_stripped_with_the_note() -> None:
    """The guard-trim line is part of the note block, so the persistence boundary removes it."""
    _reset_singletons()
    ContextTrimStore().record(storage_id=1, thread_id=0, dropped_units=1, dropped_messages=1, truncated_messages=0)

    note = collect_volatile_context_note(user_id=1, interface=_interface())
    carried = "user text\n\n" + note

    assert "guard_trim_warning" in carried
    cleaned = strip_context_note(carried)
    assert "guard_trim_warning" not in cleaned
    assert cleaned == "user text"


# --- Provider entry point (chat completions path) --------------------------------


def _create_final_answer_response(answer: str) -> ChatCompletion:
    """Build a ChatCompletion containing a final answer without tool calls."""
    mock_choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=answer, tool_calls=None),
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_chat_completions_entry_point_discloses_the_trim_on_the_wire() -> None:
    """End to end: a guard-recorded trim reaches the last user message of the outgoing payload."""
    _reset_singletons()
    ContextTrimStore().record(storage_id=12345, thread_id=0, dropped_units=2, dropped_messages=2, truncated_messages=0)

    completions = SimpleNamespace(create=AsyncMock(side_effect=[_create_final_answer_response("Hello")]))
    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = MoonshotAI(token=TEST_TOKEN)
    user = User(id=12345)
    interface = cast(
        Any,
        SimpleNamespace(
            storage_id=12345,
            thread_id=0,
            uses_uploaded_file_storage=False,
            captures_llm_thoughts=False,
            send_llm_thoughts=AsyncMock(),
        ),
    )
    conversation = [Message(role="user", content="Hi")]

    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=mock_client),
        patch("chibi.services.providers.provider.MetricsService"),
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch.object(provider, "call_functions", new=AsyncMock()),
    ):
        await provider.get_chat_response(
            messages=conversation,
            user=user,
            caller_storage_id=12345,
            caller_thread_id=0,
            model=TEST_MODEL,
            system_prompt="base",
            interface=interface,
        )

    wire_messages = completions.create.call_args.kwargs["messages"]
    last_user = [item for item in wire_messages if item["role"] == "user"][-1]
    assert "guard_trim_warning" in last_user["content"]
    assert "2 oldest unit(s) were hidden from this request" in last_user["content"]
    # The record was consumed by the entry point, the next turn stays clean.
    assert ContextTrimStore().consume(storage_id=12345, thread_id=0) is None
