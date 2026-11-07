"""Tests for chibi.services.user module."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.models import FunctionSchema, Message, ToolSchema
from chibi.services.context_counters import ContextDecisionCounters
from chibi.services.context_measurement import SizeProvenance, estimate_messages_tokens
from chibi.services.user import (
    clone_thread_messages,
    get_cwd,
    last_resort_fit,
    set_thread_working_dir,
    set_working_dir,
)
from chibi.storage.local import LocalStorage


async def test_set_thread_working_dir_expands_tilde(tmp_path: Path) -> None:
    """Test that the thread-scoped setter normalizes '~' paths and persists them."""
    db = LocalStorage(storage_path=str(tmp_path))

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=42, new_wd="~/project")

    user = await db.get_or_create_user(user_id=1)
    expected = str(Path("~/project").expanduser())
    assert user.thread_working_dirs[42] == expected
    assert not expected.startswith("~")
    refreshed = await db.get_or_create_user(user_id=1)
    assert refreshed.get_effective_working_dir(thread_id=42) == expected


async def test_set_thread_working_dir_no_existence_check(tmp_path: Path) -> None:
    """Test that a non-existent directory can still be stored."""
    db = LocalStorage(storage_path=str(tmp_path))
    nonexistent = tmp_path / "does_not_exist"

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=7, new_wd=str(nonexistent))

    user = await db.get_or_create_user(user_id=1)
    assert user.thread_working_dirs[7] == str(nonexistent)


async def test_set_thread_working_dir_overwrites_existing_slot(tmp_path: Path) -> None:
    """Test that setting an override twice updates the slot without touching legacy fields."""
    db = LocalStorage(storage_path=str(tmp_path))
    user = await db.get_or_create_user(user_id=1)
    legacy = user.working_dir

    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=3, new_wd="/first")
    await set_thread_working_dir.__wrapped__(db, user_id=1, thread_id=3, new_wd="~/second")

    updated = await db.get_or_create_user(user_id=1)
    assert updated.thread_working_dirs[3] == str(Path("~/second").expanduser())
    assert updated.get_effective_working_dir(thread_id=4) == legacy


async def test_legacy_set_working_dir_stores_value_verbatim(tmp_path: Path) -> None:
    """Test that legacy set_working_dir/get_cwd behavior stays unchanged."""
    db = LocalStorage(storage_path=str(tmp_path))

    await set_working_dir.__wrapped__(db, user_id=1, new_wd="~/legacy")

    user = await db.get_or_create_user(user_id=1)
    assert user.working_dir == "~/legacy"
    assert await get_cwd.__wrapped__(db, user_id=1) == "~/legacy"


async def test_clone_thread_messages_persists_cloned_history_on_real_local_storage(tmp_path: Path) -> None:
    """Regression: on LocalStorage the clone's final save_user must not wipe just-cloned messages.

    LocalStorage keeps thread messages inside the user pickle and add_message persists a reloaded
    copy, so a stale save_user after the clone loop used to erase the cloned history (destination
    ended up empty while the ack reported N copies). Uses a real LocalStorage on a tmp dir — no DB mocking.
    """
    db = LocalStorage(storage_path=str(tmp_path))
    seeded = [
        Message(role="user", content="My favorite number is 42."),
        Message(role="assistant", content="OK"),
    ]
    user = await db.get_or_create_user(user_id=777)
    for message in seeded:
        await db.add_message(user=user, message=message, thread_id=101)

    copied = await clone_thread_messages.__wrapped__(
        db, storage_id=777, old_thread_id=101, new_thread_id=202, name="regression-clone"
    )

    assert copied == len(seeded)
    refreshed = await db.get_or_create_user(user_id=777)  # fresh read from disk
    assert [(m.role, m.content) for m in refreshed.thread_messages_map[202]] == [(m.role, m.content) for m in seeded]
    assert len(refreshed.thread_messages_map[202]) == copied  # returned count matches reality
    assert len(refreshed.thread_messages_map[101]) == len(seeded)  # source thread intact
    assert refreshed.thread_names[202] == "regression-clone"


def _make_last_resort_history() -> list[Message]:
    """Build a history with a tool pair, bulky dialogue turns and a protected last turn."""
    assistant = Message(
        role="assistant",
        content="Calling tool",
        tool_calls=[
            ToolSchema(id="call_1", type="function", function=FunctionSchema(name="read_file", arguments="{}"))
        ],
    )
    tool_result = Message(role="tool", content="r" * 8000, tool_call_id="call_1")
    big_user = Message(role="user", content="u" * 8000)
    recent = Message(role="assistant", content="recent turn")
    return [assistant, tool_result, big_user, recent]


def test_last_resort_fit_drops_oldest_units_while_they_do_not_overshoot() -> None:
    """Small old units are dropped whole, oldest first, until the budget is met."""
    old_user = Message(role="user", content="o" * 400)
    mid_user = Message(role="user", content="m" * 400)
    recent = Message(role="assistant", content="recent turn")
    messages = [old_user, mid_user, recent]

    total = estimate_messages_tokens(messages)
    old_weight = estimate_messages_tokens([old_user])
    mid_weight = estimate_messages_tokens([mid_user])
    # The budget is reachable by dropping exactly the two oldest units: both are small
    # relative to the excess, so both go whole and the last turn survives.
    budget = total - old_weight - mid_weight

    shrunk, fits = last_resort_fit(
        deepcopy(messages), budget=budget, measured_total=total, measured_request_counted_tokens=total
    )

    assert fits is True
    assert [msg.content for msg in shrunk] == [recent.content]


def test_last_resort_fit_truncates_tool_results_before_other_messages() -> None:
    """When a unit would overshoot the excess it is truncated, tool results first.

    The tool result and the big user turn are both larger than the gap to close, so neither is
    dropped; the truncation order must consume the tool result first and leave the user turn's
    content intact.
    """
    messages = _make_last_resort_history()
    snapshot = deepcopy(messages)

    total = estimate_messages_tokens(messages)
    # The excess is smaller than any single droppable unit, so the drop step leaves everything
    # and the truncation step closes the gap, tool result first.
    budget = total - estimate_messages_tokens([messages[1]]) // 2

    shrunk, fits = last_resort_fit(messages, budget=budget, measured_total=total, measured_request_counted_tokens=total)

    assert fits is True
    assert len(shrunk) == len(messages)
    tool_result = next(msg for msg in shrunk if msg.role == "tool")
    big_user = next(msg for msg in shrunk if msg.role == "user")
    assert len(tool_result.content) < len(snapshot[1].content)
    assert big_user.content == snapshot[2].content


def test_last_resort_fit_keeps_tool_pairs_intact_and_ids_stable() -> None:
    """Truncation never breaks a tool pair and never renumbers, recreates or re-keys messages."""
    messages = _make_last_resort_history()
    ids_before = [msg.id for msg in messages]

    total = estimate_messages_tokens(messages)
    budget = total - estimate_messages_tokens([messages[1]]) // 2

    shrunk, fits = last_resort_fit(messages, budget=budget, measured_total=total, measured_request_counted_tokens=total)

    assert fits is True
    assert [msg.id for msg in shrunk] == ids_before
    assistant = next(msg for msg in shrunk if msg.tool_calls)
    tool_result = next(msg for msg in shrunk if msg.role == "tool")
    assert assistant.tool_calls is not None
    assert [call.id for call in assistant.tool_calls] == ["call_1"]
    assert tool_result.tool_call_id == "call_1"
    # No orphan: the only tool result still belongs to a present assistant call.
    present_call_ids = {call.id for msg in shrunk if msg.tool_calls is not None for call in msg.tool_calls}
    assert all(msg.tool_call_id in present_call_ids for msg in shrunk if msg.role == "tool")


def test_last_resort_fit_never_drops_or_truncates_last_message() -> None:
    """Even with an unreachable budget the last message keeps its content and its id."""
    messages = _make_last_resort_history()
    last = messages[-1]

    shrunk, fits = last_resort_fit(
        deepcopy(messages),
        budget=1,
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert fits is False
    assert shrunk[-1].id == last.id
    assert shrunk[-1].content == last.content


def test_last_resort_fit_is_deterministic() -> None:
    """Same input list, same output: ids and contents match across independent runs."""
    base = _make_last_resort_history()
    total = estimate_messages_tokens(base)
    budget = total - estimate_messages_tokens([base[1]]) // 2

    shrunk_first, fits_first = last_resort_fit(
        deepcopy(base), budget=budget, measured_total=total, measured_request_counted_tokens=total
    )
    shrunk_second, fits_second = last_resort_fit(
        deepcopy(base), budget=budget, measured_total=total, measured_request_counted_tokens=total
    )

    assert fits_first == fits_second
    # deepcopy preserves the ids, so the full identity comparison must hold.
    assert [(msg.id, msg.content) for msg in shrunk_first] == [(msg.id, msg.content) for msg in shrunk_second]


def test_last_resort_fit_noop_when_already_within_budget() -> None:
    """A measured total within the budget is a no-op: the messages come back untouched."""
    messages = _make_last_resort_history()
    snapshot = deepcopy(messages)

    shrunk, fits = last_resort_fit(
        deepcopy(messages),
        budget=estimate_messages_tokens(messages),
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert fits is True
    assert [(msg.id, msg.content) for msg in shrunk] == [(msg.id, msg.content) for msg in snapshot]


def test_last_resort_fit_reports_failure_when_nothing_can_fit() -> None:
    """When only the protected last message remains and it alone exceeds the budget, it reports failure."""
    huge = Message(role="user", content="h" * 8000)
    recent = Message(role="assistant", content="recent turn")
    messages = [huge, recent]

    shrunk, fits = last_resort_fit(
        deepcopy(messages),
        budget=1,
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert fits is False
    assert [msg.content for msg in shrunk] == [recent.content]


def test_last_resort_fit_log_counts_distinct_truncated_messages() -> None:
    """The shrink log counts distinct truncated messages, not halving iterations.

    One big message halved over several iterations used to be reported as
    "N message(s) content-truncated" with N equal to the iteration count, which
    the observability work item cannot consume as a message count. The fixture
    forces at least two halving iterations of a single message and pins the
    logged count at 1.
    """
    from loguru import logger

    big_user = Message(role="user", content="u" * 8000)
    recent = Message(role="assistant", content="recent turn")
    messages = [big_user, recent]
    total = estimate_messages_tokens(messages)
    weight = estimate_messages_tokens([big_user])
    # The excess exceeds half the message's weight, so one halving cannot close
    # the gap: the loop must halve the same message more than once.
    budget = total - int(weight * 0.7)

    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG")
    try:
        shrunk, fits = last_resort_fit(
            deepcopy(messages), budget=budget, measured_total=total, measured_request_counted_tokens=total
        )
    finally:
        logger.remove(sink_id)

    assert fits is True
    assert shrunk[0].id == big_user.id
    assert len(shrunk[0].content) < len(big_user.content)
    line = next(record for record in records if "Last-resort shrink" in record)
    assert "1 message(s) content-truncated" in line


def test_last_resort_fit_drops_non_truncatable_unit_when_truncation_cannot_fit() -> None:
    """Regression (review reproduction): a non-truncatable unit that blocks the fit is dropped whole.

    A bulky assistant ``tool_calls`` carrier overshoots the excess (so the overshoot guard
    skips it in step one) and is excluded from truncation (its protocol role lives in the
    untouched structure). Dropping that unit whole, the loss this fallback exists to make,
    closes the gap: total - unit_weight = 104 <= budget. The function must end with
    fits=True instead of the old fits=False while the request still does not fit.
    """
    assistant = Message(
        role="assistant",
        content="Calling tool",
        tool_calls=[
            ToolSchema(
                id="call_1",
                type="function",
                function=FunctionSchema(name="read_file", arguments='{"payload": "' + "x" * 40000 + '"}'),
            )
        ],
    )
    tool_result = Message(role="tool", content="r" * 400, tool_call_id="call_1")
    recent = Message(role="assistant", content="recent turn")
    messages = [assistant, tool_result, recent]

    total = estimate_messages_tokens(messages)
    unit_weight = estimate_messages_tokens([assistant, tool_result])
    budget = total - 200
    # The premise of the gap: the unit overshoots the excess and cannot be truncated,
    # while dropping it whole would fit.
    assert unit_weight > total - budget
    assert total - unit_weight <= budget

    shrunk, fits = last_resort_fit(
        deepcopy(messages), budget=budget, measured_total=total, measured_request_counted_tokens=total
    )

    assert fits is True
    assert [msg.content for msg in shrunk] == [recent.content]
    # No orphan tool result: the dropped pair is gone as a whole, the last turn survives.
    assert all(msg.tool_call_id is None for msg in shrunk)
    assert shrunk[-1].id == recent.id
    assert shrunk[-1].content == recent.content


def _make_guard_provider(max_tokens: int) -> Mock:
    """Build a provider double whose output-cap resolution returns a concrete integer."""
    provider = Mock()
    provider._get_max_tokens_value = Mock(return_value=max_tokens)
    return provider


@pytest.mark.asyncio
async def test_enforce_context_budget_passes_untouched_when_within_window() -> None:
    """A request that fits the window together with the output reservation is sent unchanged."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit") as mock_shrink,
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    assert result is messages
    mock_shrink.assert_not_called()


@pytest.mark.asyncio
async def test_enforce_context_budget_truncates_when_over_window() -> None:
    """An oversized request is refused and shrunk through the last-resort path."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="old"), Message(role="assistant", content="new")]
    shrunk = [messages[-1]]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(12000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=(shrunk, True)) as mock_shrink,
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    mock_shrink.assert_called_once_with(
        messages,
        budget=9800,
        measured_total=12000,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )
    assert result is shrunk


@pytest.mark.asyncio
async def test_enforce_context_budget_accounts_for_the_output_reservation() -> None:
    """An input that fits the window but leaves no room for the requested output is oversized."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="hello")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(9900, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=([messages[-1]], True)) as mock_shrink,
    ):
        await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    # 9900 + 200 > 10000: the reservation turns a fitting input into an
    # oversized request.
    mock_shrink.assert_called_once()
    assert mock_shrink.call_args.kwargs["budget"] == 9800


@pytest.mark.asyncio
async def test_enforce_context_budget_falls_through_when_window_unknown() -> None:
    """Without a resolvable window the guard guesses nothing and falls through."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="hello")]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=None),
        patch("chibi.services.user.measure_request_size_with_provenance", new=AsyncMock()) as mock_measure,
        patch("chibi.services.user.last_resort_fit") as mock_shrink,
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
    mock_shrink.assert_not_called()


@pytest.mark.asyncio
async def test_enforce_context_budget_sends_best_effort_when_truncation_cannot_fit() -> None:
    """When even the last-resort path cannot fit the request, the shrunk list is sent anyway."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="huge")]
    shrunk = [messages[-1]]

    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(12000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=(shrunk, False)) as mock_shrink,
    ):
        result = await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=messages,
        )

    mock_shrink.assert_called_once()
    assert result is shrunk


@pytest.mark.asyncio
async def test_enforce_context_budget_uses_the_shared_reservation_expression() -> None:
    """The reservation comes from the single shared expression, not a second copy."""
    from chibi.services.user import _enforce_context_budget, _summarizer_max_tokens

    provider = _make_guard_provider(777)

    with (
        patch("chibi.services.user.get_model_context_window", return_value=1000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(900, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=([Message(role="user", content="x")], True)) as m,
    ):
        await _enforce_context_budget(
            provider=provider,
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=[Message(role="user", content="hello")],
        )

    assert m.call_args.kwargs["budget"] == 1000 - _summarizer_max_tokens(provider, "gpt-4")
    assert m.call_args.kwargs["budget"] == 223


@pytest.mark.asyncio
async def test_enforce_context_budget_counts_the_noop_and_logs_provenance() -> None:
    """A request that fits counts as a guard no-op, and the decision line names size, provenance, threshold."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    ContextDecisionCounters().reset()

    records: list[str] = []
    sink_id = None
    from loguru import logger

    sink_id = logger.add(records.append, level="DEBUG")
    try:
        with (
            patch("chibi.services.user.get_model_context_window", return_value=10000),
            patch(
                "chibi.services.user.measure_request_size_with_provenance",
                new=AsyncMock(return_value=(5000, SizeProvenance.ESTIMATED)),
            ),
            patch("chibi.services.user.last_resort_fit") as mock_shrink,
        ):
            result = await _enforce_context_budget(
                provider=_make_guard_provider(200),
                model="gpt-4",
                storage_id=1,
                thread_id=0,
                messages=messages,
            )
    finally:
        logger.remove(sink_id)

    assert result is messages
    mock_shrink.assert_not_called()
    assert ContextDecisionCounters().snapshot().get("context_guard_noop") == 1
    line = next(record for record in records if "Context guard:" in record)
    assert "5000 tokens" in line
    assert "provenance=estimated" in line
    assert "10000-token window" in line
    assert "200-token output reservation" in line


@pytest.mark.asyncio
async def test_enforce_context_budget_counts_truncation_outcomes() -> None:
    """The guard counts a fitting shrink and a still-over shrink as different decisions."""
    from chibi.services.user import _enforce_context_budget

    messages = [Message(role="user", content="old"), Message(role="assistant", content="new")]

    ContextDecisionCounters().reset()
    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(12000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=([messages[-1]], True)),
    ):
        await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=deepcopy(messages),
        )
    assert ContextDecisionCounters().snapshot().get("context_guard_truncated") == 1

    ContextDecisionCounters().reset()
    with (
        patch("chibi.services.user.get_model_context_window", return_value=10000),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(return_value=(12000, SizeProvenance.ESTIMATED)),
        ),
        patch("chibi.services.user.last_resort_fit", return_value=([messages[-1]], False)),
    ):
        await _enforce_context_budget(
            provider=_make_guard_provider(200),
            model="gpt-4",
            storage_id=1,
            thread_id=0,
            messages=deepcopy(messages),
        )
    assert ContextDecisionCounters().snapshot().get("context_guard_truncated_still_over") == 1


def test_last_resort_fit_log_lists_tier_identities() -> None:
    """The shrink log names each tier with the dropped/truncated ids and roles, never content."""
    from loguru import logger

    old = Message(role="user", content="o" * 8000)
    mid = Message(role="assistant", content="a" * 8000)
    recent = Message(role="assistant", content="recent")
    messages = [old, mid, recent]
    total = estimate_messages_tokens(messages)
    # The excess admits the oldest unit whole (no overshoot skip) but leaves a
    # remainder for the truncation step on the middle message.
    budget = total - 3000

    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG")
    try:
        shrunk, fits = last_resort_fit(
            deepcopy(messages), budget=budget, measured_total=total, measured_request_counted_tokens=total
        )
    finally:
        logger.remove(sink_id)

    assert fits is True
    assert [msg.id for msg in shrunk] == [mid.id, recent.id]
    line = next(record for record in records if "Last-resort shrink" in record)
    assert f"oldest_units=1 [{old.id}:user]" in line
    assert f"truncated=1 [{mid.id}:assistant]" in line
    assert "sweep_units=0 []" in line
    assert all("o" * 100 not in record for record in records)
    assert all("a" * 100 not in record for record in records)
