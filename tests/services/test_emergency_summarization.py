"""Unit tests for emergency summarization (measured fit budget, two-tier drop, pair integrity, empty guards)."""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chibi.models import FunctionSchema, Message, ToolSchema
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.context_measurement import SizeProvenance, estimate_messages_tokens
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import (
    COMPACTION_SEED_SOURCE,
    EMERGENCY_SUMMARIZATION_INPUT_BUDGET,
    CompactionOutcome,
    CompactionResult,
    TokenProvenance,
    _complete_units,
    _fit_messages_to_budget,
    _is_tool_response_message,
    _summarizer_fit_budget,
    _summarizer_max_tokens,
    emergency_summarization,
    verify_compaction,
)


def _make_tool_response_blob(tool_name: str, filler: str) -> Message:
    """Build a shape-(ii) background tool-response turn (role="user" JSON blob).

    Args:
        tool_name: Name stored in the blob payload.
        filler: Payload content used to control the message size.

    Returns:
        The constructed message.
    """
    payload = {"type": "tool response", "tool_name": tool_name, "result": filler}
    return Message(role="user", content=json.dumps(payload))


def _make_shape_i_pair(tool_result_content: str) -> list[Message]:
    """Build a shape-(i) in-loop tool-call pair (assistant tool_calls + role="tool" result).

    Args:
        tool_result_content: Content for the role="tool" message.

    Returns:
        The assistant and tool messages forming the pair.
    """
    assistant = Message(
        role="assistant",
        content="Calling tool",
        tool_calls=[
            ToolSchema(id="call_1", type="function", function=FunctionSchema(name="read_file", arguments="{}"))
        ],
    )
    tool_result = Message(role="tool", content=tool_result_content, tool_call_id="call_1")
    return [assistant, tool_result]


def _make_summarization_mocks(
    messages: list[Message], answer: str = "summary", model: str = "active-model"
) -> tuple[MagicMock, MagicMock]:
    """Build a mock Database and a fake active provider for emergency_summarization.

    Args:
        messages: Conversation history returned by the mocked storage.
        answer: Answer text the fake provider returns.
        model: Active model name the fake user reports (``active-model`` is
            unresolvable in the context-window map, exercising the fallback).

    Returns:
        A tuple of the database mock and the fake provider whose get_chat_response is an AsyncMock.
    """
    user = MagicMock()
    user.id = 1
    user.get_active_llm_model = Mock(return_value=model)
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


async def _run_emergency_summarization(
    db: MagicMock, patch_replace: bool = True, result_holder: dict[str, Any] | None = None, **patches: Any
) -> AsyncMock:
    """Run emergency_summarization with the history replacement stubbed out.

    By default the single replacement helper is mocked (the tests assert on
    the seed it receives). With ``patch_replace=False`` only
    ``reset_chat_history`` is mocked instead, so the real helper, and its
    usage-cache invalidation, runs against the mocked database.

    The single measurement entry point is patched (it is an injected wrapper
    that would otherwise touch real storage); by default it reports the
    tokenizer count of the mocked history for the pre-compaction measure and
    one token less for the post-compaction verify (so a completed compaction
    is predicted to fit unless the test overrides a figure), and
    tests exercise the real fit arithmetic against a consistent total unless
    they override the figures.

    Args:
        db: The mocked database to inject.
        patch_replace: When True, mock out ``replace_thread_history``;
            otherwise mock out ``reset_chat_history`` and keep the helper real.
        result_holder: Optional dict; when given, the ``CompactionResult``
            returned by ``emergency_summarization`` is stored under the key
            ``"result"`` so a test can branch on it.
        **patches: Keyword arguments forwarded to ``patch`` over user-module gpt_settings attributes;
            ``measured_total`` overrides the pre-compaction measured request size reported to the fit;
            ``post_measured_total`` overrides the post-compaction size the verify phase reports.

    Returns:
        The AsyncMock standing in for the history replacement (the helper by
        default, ``reset_chat_history`` when ``patch_replace=False``).
    """
    messages: list[Message] = db.get_conversation_messages.return_value
    measured_total = patches.pop("measured_total", estimate_messages_tokens(messages))
    post_measured_total = patches.pop("post_measured_total", max(measured_total - 1, 0))
    if patch_replace:
        replace_mock = AsyncMock()
        with (
            patch("chibi.services.user.replace_thread_history.__wrapped__", replace_mock),
            patch("chibi.services.user.gpt_settings.max_tokens", 32000),
            patch("chibi.services.user.gpt_settings.max_history_tokens", patches.get("max_history_tokens", 100000)),
            patch(
                "chibi.services.user.measure_request_size_with_provenance",
                new=AsyncMock(
                    side_effect=[
                        (measured_total, SizeProvenance.ESTIMATED),
                        (post_measured_total, SizeProvenance.ESTIMATED),
                    ]
                ),
            ),
        ):
            result = await emergency_summarization.__wrapped__(db, storage_id=1, thread_id=0)
        if result_holder is not None:
            result_holder["result"] = result
        return replace_mock
    reset_mock = AsyncMock()
    with (
        patch("chibi.services.user.reset_chat_history.__wrapped__", reset_mock),
        patch("chibi.services.user.gpt_settings.max_tokens", 32000),
        patch("chibi.services.user.gpt_settings.max_history_tokens", patches.get("max_history_tokens", 100000)),
        patch(
            "chibi.services.user.measure_request_size_with_provenance",
            new=AsyncMock(
                side_effect=[
                    (measured_total, SizeProvenance.ESTIMATED),
                    (post_measured_total, SizeProvenance.ESTIMATED),
                ]
            ),
        ),
    ):
        result = await emergency_summarization.__wrapped__(db, storage_id=1, thread_id=0)
    if result_holder is not None:
        result_holder["result"] = result
    return reset_mock


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


@pytest.mark.asyncio
async def test_emergency_summarization_passes_active_model_and_unboosted_max_tokens() -> None:
    """The provider call must receive the active model and the derived, unboosted max_tokens.

    The summarizer may never reserve more output than the request that just
    failed: the ~30% boost is gone, and the requested cap is exactly the
    single shared reservation expression the fit budget subtracts. The fake
    provider resolves no concrete cap, so the documented configured default
    applies.
    """
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, provider = _make_summarization_mocks(list(messages))

    await _run_emergency_summarization(db, max_history_tokens=100000)

    kwargs = provider.get_chat_response.await_args.kwargs
    assert kwargs["model"] == "active-model"
    assert kwargs["max_tokens"] == 32000 == _summarizer_max_tokens(provider, "active-model")
    assert kwargs["system_prompt"].startswith("Summarize this conversation")


@pytest.mark.asyncio
async def test_emergency_summarization_marks_the_seed_messages() -> None:
    """Both seed messages written by the replacement carry the compaction source marker."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(list(messages))

    await _run_emergency_summarization(db, patch_replace=False, max_history_tokens=100000)

    written = [call.kwargs["message"] for call in db.add_message.await_args_list]
    assert len(written) == 2
    assert all(message.source == COMPACTION_SEED_SOURCE for message in written)
    assert [message.role for message in written] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_emergency_summarization_invalidates_usage_cache() -> None:
    """After a successful summarization the cached pre-reset prompt size must be dropped."""
    _reset_usage_cache()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(list(messages))

    store = UsageCacheStore()
    store._data[store._make_key(user_id=1, thread_id=0)] = 99999

    await _run_emergency_summarization(db, patch_replace=False, max_history_tokens=100000)

    assert store.get(user_id=1, thread_id=0) is None


@pytest.mark.asyncio
async def test_shape_i_pairs_excluded_from_input_and_history_never_mutated() -> None:
    """Shape-(i) tool-call pairs never reach the summarizer input and stored history stays intact."""
    pair = _make_shape_i_pair(tool_result_content='{"status": "ok", "result": "SECRET_SHAPE_I_PAYLOAD"}')
    messages = [
        Message(role="user", content="real user turn"),
        *pair,
        Message(role="assistant", content="final answer"),
    ]
    snapshot = deepcopy(messages)
    db, provider = _make_summarization_mocks(messages)

    await _run_emergency_summarization(db, max_history_tokens=100000)

    sent_messages = provider.get_chat_response.await_args.kwargs["messages"]
    assert len(sent_messages) == 1
    sent_text = sent_messages[0].content
    assert "SECRET_SHAPE_I_PAYLOAD" not in sent_text
    assert "Calling tool" not in sent_text
    assert "real user turn" in sent_text
    assert "final answer" in sent_text
    assert messages == snapshot


@pytest.mark.asyncio
async def test_fit_uses_measured_scale_over_budget_when_counted_is_small() -> None:
    """The budget decision follows the measured real-scale figure, not the counted history size.

    The counted history is a couple hundred tokens while the measured request size is 150k against
    a 120k budget: the old character-heuristic fit would have dropped nothing, so a drop here proves
    the decision runs on the measured scale.
    """
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 400)
    messages = [
        Message(role="user", content="a" * 80),
        blob,
        Message(role="assistant", content="recent user turn"),
    ]
    db, provider = _make_summarization_mocks(deepcopy(messages))

    await _run_emergency_summarization(db, max_history_tokens=200000, measured_total=150000)

    counted = estimate_messages_tokens(messages)
    assert counted < int(200000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" not in sent_text
    assert "recent user turn" in sent_text


@pytest.mark.asyncio
async def test_fit_noop_when_measured_total_within_budget_despite_large_counted_history() -> None:
    """A measured size within the budget is a clean no-op even when the raw history is bulky."""
    messages = [
        Message(role="user", content="u" * 2000),
        Message(role="assistant", content="a" * 2000),
    ]
    db, provider = _make_summarization_mocks(deepcopy(messages))

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=1000, measured_total=100)

    counted = estimate_messages_tokens(messages)
    assert counted > int(1000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    # Whole history went to the summarizer: nothing was dropped.
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "u" * 100 in sent_text
    assert "a" * 100 in sent_text
    # The compaction completed normally.
    assert replace_mock.await_count == 1


@pytest.mark.asyncio
async def test_shape_ii_blobs_dropped_under_budget_and_recent_turn_kept() -> None:
    """Over-budget shape-(ii) blobs are dropped from the summarizer input, the latest turn is kept."""
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 400)
    messages = [
        Message(role="user", content="a" * 80),
        blob,
        Message(role="assistant", content="recent user turn"),
    ]
    snapshot = deepcopy(messages)
    db, provider = _make_summarization_mocks(messages)

    await _run_emergency_summarization(db, max_history_tokens=100)

    budget = int(100 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    total = estimate_messages_tokens(messages)
    blob_weight = estimate_messages_tokens([blob])
    assert budget < total
    assert budget >= total - blob_weight
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" not in sent_text
    assert "a" * 80 in sent_text
    assert "recent user turn" in sent_text
    assert messages == snapshot


def test_fit_messages_to_budget_drops_largest_blob_first() -> None:
    """With a budget reachable by dropping only the largest blob, the smaller blob must survive.

    Dropping smallest-first would have to remove both blobs to fit; keeping the small blob proves
    the largest-first ordering.
    """
    small_blob = _make_tool_response_blob("small_tool", "s" * 400)
    big_blob = _make_tool_response_blob("big_tool", "b" * 4000)
    old_turn = Message(role="user", content="old turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [old_turn, big_blob, small_blob, recent]

    measured_total = estimate_messages_tokens(messages)
    budget = measured_total - estimate_messages_tokens([big_blob])

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=budget,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert [msg.content for msg in kept] == [old_turn.content, small_blob.content, recent.content]


def test_fit_messages_to_budget_drops_oldest_turns_after_blobs() -> None:
    """When blob-dropping is not enough, the oldest remaining turns go next."""
    blob = _make_tool_response_blob("tool", "x" * 400)
    first_turn = Message(role="user", content="first")
    second_turn = Message(role="assistant", content="second")
    recent = Message(role="assistant", content="most recent turn")
    messages = [first_turn, blob, second_turn, recent]

    measured_total = estimate_messages_tokens(messages)
    budget = measured_total - estimate_messages_tokens([blob]) - estimate_messages_tokens([first_turn])

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=budget,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert [msg.content for msg in kept] == [second_turn.content, recent.content]


def test_fit_messages_to_budget_never_drops_most_recent_message() -> None:
    """Even with a tiny budget, the most recent message must survive."""
    first = Message(role="user", content="x" * 400)
    recent = Message(role="assistant", content="most recent turn")
    messages = [first, recent]

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=1,
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert [msg.content for msg in kept] == [recent.content]


def test_fit_messages_to_budget_protects_most_recent_tool_response_blob() -> None:
    """The last-message protection covers tool-response blobs too: the current turn must survive."""
    old = Message(role="user", content="old turn")
    blob_last = _make_tool_response_blob("web_search", "x" * 400)
    messages = [old, blob_last]

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=1,
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert [msg.content for msg in kept] == [blob_last.content]


def test_fit_messages_to_budget_noop_when_measured_total_fits() -> None:
    """A measured total within the budget is a no-op: the candidates are returned unchanged."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=estimate_messages_tokens(messages),
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert kept == messages


def test_fit_messages_to_budget_returns_unchanged_when_nothing_can_be_dropped() -> None:
    """Over budget but with nothing droppable (only the protected last message), the fit degrades cleanly."""
    messages = [Message(role="user", content="only turn")]

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=1,
        measured_total=estimate_messages_tokens(messages),
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert kept == messages


def test_complete_units_keep_tool_pairs_together() -> None:
    """An assistant tool_calls message and its role=tool results form one indivisible unit."""
    assistant, tool_result = _make_shape_i_pair(tool_result_content="result payload")
    second_pair_assistant = Message(
        role="assistant",
        content="Calling two tools",
        tool_calls=[
            ToolSchema(id="call_a", type="function", function=FunctionSchema(name="t1", arguments="{}")),
            ToolSchema(id="call_b", type="function", function=FunctionSchema(name="t2", arguments="{}")),
        ],
    )
    result_a = Message(role="tool", content="result a", tool_call_id="call_a")
    result_b = Message(role="tool", content="result b", tool_call_id="call_b")
    plain = Message(role="user", content="plain turn")
    messages = [assistant, tool_result, plain, second_pair_assistant, result_a, result_b]

    units = _complete_units(messages)

    assert units == [[assistant, tool_result], [plain], [second_pair_assistant, result_a, result_b]]


def test_fit_messages_to_budget_drops_complete_tool_pairs_only() -> None:
    """The second tier never separates an assistant tool_calls message from its role=tool results.

    The budget is set so the drop boundary falls exactly at the pair: the pair goes whole, and the
    dialogue after it survives. A boundary one token past the pair also takes the next complete
    unit (the following user turn), never the assistant alone.
    """
    assistant, tool_result = _make_shape_i_pair(tool_result_content="r" * 400)
    user_mid = Message(role="user", content="mid dialogue turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [assistant, tool_result, user_mid, recent]

    pair_weight = estimate_messages_tokens([assistant]) + estimate_messages_tokens([tool_result])
    measured_total = estimate_messages_tokens(messages)

    # Boundary exactly at the pair: only the pair goes.
    kept_exact = _fit_messages_to_budget(
        deepcopy(messages),
        budget=measured_total - pair_weight,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )
    assert [msg.content for msg in kept_exact] == [user_mid.content, recent.content]

    # Boundary one token before the pair sufficing: the pair and the next complete unit go
    # together; the assistant tool_calls message is never orphaned from its result.
    kept_over = _fit_messages_to_budget(
        deepcopy(messages),
        budget=measured_total - pair_weight - 1,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )
    assert [msg.content for msg in kept_over] == [recent.content]


def test_fit_messages_to_budget_drops_orphan_tool_result_whole() -> None:
    """An orphan role=tool result is removed as its own unit instead of being left behind."""
    orphan = Message(role="tool", content="o" * 400, tool_call_id="call_missing")
    user_turn = Message(role="user", content="user turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [orphan, user_turn, recent]

    orphan_weight = estimate_messages_tokens([orphan])
    measured_total = estimate_messages_tokens(messages)

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=measured_total - orphan_weight,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    assert [msg.content for msg in kept] == [user_turn.content, recent.content]


@pytest.mark.asyncio
async def test_emergency_summarization_skips_when_fitted_input_is_empty() -> None:
    """A history with no summarizable content must not produce an empty seed: the thread stays untouched."""
    pair = _make_shape_i_pair(tool_result_content='{"status": "ok"}')
    db, provider = _make_summarization_mocks(list(pair))

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=100)

    provider.get_chat_response.assert_not_awaited()
    replace_mock.assert_not_awaited()
    db.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_emergency_summarization_skips_on_blank_content_only() -> None:
    """Kept messages whose content is only whitespace are treated as an empty input."""
    messages = [Message(role="user", content="   \n\t ")]
    db, provider = _make_summarization_mocks(messages)

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=100000)

    provider.get_chat_response.assert_not_awaited()
    replace_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_emergency_summarization_aborts_on_empty_answer() -> None:
    """An empty summarizer answer must not be written as the seed: the thread stays untouched."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(messages, answer="")

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=100000)

    replace_mock.assert_not_awaited()
    db.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_emergency_summarization_aborts_on_whitespace_answer() -> None:
    """A whitespace-only summarizer answer is near-empty and must not become the seed."""
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(messages, answer="   \n")

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=100000)

    replace_mock.assert_not_awaited()
    db.add_message.assert_not_awaited()


@pytest.mark.parametrize(
    ("content", "role", "expected"),
    [
        (json.dumps({"type": "tool response", "tool_name": "t", "result": "r"}), "user", True),
        (json.dumps({"type": "something else"}), "user", False),
        ("not json at all", "user", False),
        (json.dumps({"type": "tool response"}), "tool", False),
        ("", "user", False),
    ],
)
def test_is_tool_response_message(content: str, role: str, expected: bool) -> None:
    """Only role="user" JSON blobs with type "tool response" are background tool responses."""
    message = Message(role=role, content=content)

    assert _is_tool_response_message(message) is expected


def test_summarizer_fit_budget_derived_from_window_minus_output_reservation() -> None:
    """A resolvable window is the budget source: window minus the shared output reservation.

    The reservation is the single shared expression ``_summarizer_max_tokens``:
    the same value the summarizer request itself sends as ``max_tokens``.
    """
    provider: Any = _FakeReservationProvider(reservation=32000)
    with patch("chibi.services.user.gpt_settings.max_tokens", 32000):
        reservation = _summarizer_max_tokens(provider, "gpt-4o")
        assert reservation == 32000

        assert _summarizer_fit_budget(provider, "gpt-4o") == 128000 - reservation
        assert _summarizer_fit_budget(provider, "claude-sonnet-5") == 1000000 - reservation


def test_summarizer_fit_budget_fallback_when_window_unresolvable() -> None:
    """An unresolvable window falls back to the documented share of max_history_tokens."""
    provider: Any = _FakeReservationProvider(reservation=32000)
    with patch("chibi.services.user.gpt_settings.max_history_tokens", 200000):
        assert _summarizer_fit_budget(provider, "active-model") == int(200000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
        assert _summarizer_fit_budget(provider, None) == int(200000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)


class _FakeReservationProvider:
    """Provider double exposing the reservation resolution of the failed request.

    Args:
        reservation: The value the failed request's resolution path yields,
            a concrete integer cap, or a non-int sentinel (``NOT_GIVEN``)
            when the failed request carried no output cap.
    """

    def __init__(self, reservation: Any) -> None:
        self._reservation = reservation

    def _get_max_tokens_value(self, model_name: str) -> Any:
        return self._reservation


def test_summarizer_reservation_derived_from_the_failed_requests_cap() -> None:
    """A concrete cap on the failed request bounds the summarizer reservation from above.

    Load-bearing: if ``_summarizer_max_tokens`` went back to reading an
    independent configured constant, the 16384 cap here could never surface
    and this test would fail. The fit budget subtracts the same derived
    reservation, not the constant.
    """
    provider: Any = _FakeReservationProvider(reservation=16384)

    assert _summarizer_max_tokens(provider, "gpt-4o") == 16384
    with (
        patch("chibi.services.user.gpt_settings.max_tokens", 32000),
        patch("chibi.services.user.gpt_settings.max_history_tokens", 200000),
    ):
        assert _summarizer_max_tokens(provider, "gpt-4o") != 32000
        # gpt-4o window minus the FAILED request's cap, not minus the default.
        assert _summarizer_fit_budget(provider, "gpt-4o") == 128000 - 16384


def test_summarizer_reservation_falls_back_to_default_when_failed_request_had_no_cap() -> None:
    """A failed request without an output cap resolves to the configured default.

    The OpenAI family sends ``max_output_tokens: NOT_GIVEN`` (the SDK omits
    it), so the failed request's resolution path yields the ``NOT_GIVEN``
    sentinel rather than a cap. There is nothing to compare against, and the
    code documents this: it resolves to ``gpt_settings.max_tokens`` and says
    so in its docstring.
    """
    from openai import NOT_GIVEN

    provider: Any = _FakeReservationProvider(reservation=NOT_GIVEN)

    with patch("chibi.services.user.gpt_settings.max_tokens", 32000):
        assert _summarizer_max_tokens(provider, "gpt-5.6-terra") == 32000
    # The no-cap case is documented in the code itself.
    assert _summarizer_max_tokens.__doc__ is not None
    assert "no output cap" in _summarizer_max_tokens.__doc__.lower()


@pytest.mark.asyncio
async def test_emergency_request_max_tokens_equals_the_failed_requests_cap() -> None:
    """The summarizer request sends exactly the cap the failed request resolution yields.

    End-to-end through ``emergency_summarization``: the failed request's
    resolution path reports 16384, so the summarizer call must ask for 16384,
    not the configured 32000 default.
    """
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, provider = _make_summarization_mocks(list(messages))
    provider._get_max_tokens_value = Mock(return_value=16384)

    await _run_emergency_summarization(db, max_history_tokens=200000)

    kwargs = provider.get_chat_response.await_args.kwargs
    assert kwargs["max_tokens"] == 16384
    assert kwargs["max_tokens"] == _summarizer_max_tokens(provider, "active-model")


@pytest.mark.asyncio
async def test_emergency_request_max_tokens_defaults_when_failed_request_had_no_cap() -> None:
    """Without a cap on the failed request, the summarizer sends the documented default.

    The OpenAI-family shape: the failed request's resolution yields the
    ``NOT_GIVEN`` sentinel, so the summarizer falls back to the configured
    ``max_tokens`` default (32000 in this suite).
    """
    from openai import NOT_GIVEN

    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, provider = _make_summarization_mocks(list(messages))
    provider._get_max_tokens_value = Mock(return_value=NOT_GIVEN)

    await _run_emergency_summarization(db, max_history_tokens=200000)

    kwargs = provider.get_chat_response.await_args.kwargs
    assert kwargs["max_tokens"] == 32000


@pytest.mark.asyncio
async def test_full_collapse_on_large_window_history_reaches_summarizer_intact() -> None:
    """With a large-window model a history past the trigger reaches the summarizer WHOLE.

    The measured request (250k) is past the 200k trigger, but the 1M window of gpt-4.1 minus the
    output reservation leaves the fit over-admitted: nothing is dropped and the entire history
    goes to the summarizer as one input. This pins the owner's full-collapse decision.
    """
    messages: list[Message] = []
    for i in range(50):
        messages.append(Message(role="user", content=f"turn {i} " + "u" * 300))
        messages.append(Message(role="assistant", content=f"answer {i} " + "a" * 150))
    db, provider = _make_summarization_mocks(deepcopy(messages), model="gpt-4.1")

    replace_mock = await _run_emergency_summarization(db, max_history_tokens=200000, measured_total=250000)

    budget = 1000000 - 32000
    assert 250000 <= budget
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    for msg in messages:
        assert msg.content in sent_text
    assert replace_mock.await_count == 1


@pytest.mark.asyncio
async def test_fallback_budget_is_the_drop_boundary_when_window_unresolvable() -> None:
    """With an unresolvable window the fallback share of max_history_tokens decides the fit."""
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 400)
    messages = [
        Message(role="user", content="a" * 80),
        blob,
        Message(role="assistant", content="recent user turn"),
    ]

    # Exactly at the fallback budget: clean no-op, everything reaches the summarizer.
    db, provider = _make_summarization_mocks(deepcopy(messages))
    await _run_emergency_summarization(
        db, max_history_tokens=200000, measured_total=int(200000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    )
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" in sent_text

    # One token over: the blob goes.
    db, provider = _make_summarization_mocks(deepcopy(messages))
    await _run_emergency_summarization(
        db, max_history_tokens=200000, measured_total=int(200000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET) + 1
    )
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" not in sent_text
    assert "recent user turn" in sent_text


@pytest.mark.asyncio
async def test_small_window_model_drops_history_to_fit_summarizer_request() -> None:
    """A small window genuinely too small for the history still drops, tier 1 first, and fits."""
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 400)
    messages = [
        Message(role="user", content="a" * 80),
        blob,
        Message(role="assistant", content="recent user turn"),
    ]
    db, provider = _make_summarization_mocks(deepcopy(messages), model="gpt-4o")

    # gpt-4o: 128000 window minus the 32000 reservation -> 96000 budget; the measured request is
    # 150k, so admission control must drop until the credited total is within the budget.
    await _run_emergency_summarization(db, max_history_tokens=200000, measured_total=150000)

    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    assert "BLOB_MARKER_PAYLOAD" not in sent_text
    assert "recent user turn" in sent_text


def _capture_fit_logs() -> tuple[list[str], Any]:
    """Return a loguru sink list and a cleanup callable for capturing fit log lines."""
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG")
    return records, lambda: logger.remove(sink_id)


def test_fit_drop_credit_matches_independently_computed_expectation() -> None:
    """The per-drop credit is anchored to the measured list, not the candidates-only count.

    Regression for the over-credit defect: with scale != 1 the credited post-drop figure must equal
    measured_total minus the dropped weight times measured_total / counted(whole measured list),
    computed here independently of the code's own arithmetic. The old implementation divided by the
    candidates-only count, which over-credited every drop and reported a post-drop total far below
    the real remaining size.
    """
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 36000)
    pair = _make_shape_i_pair(tool_result_content="r" * 200)
    user_turn = Message(role="user", content="a" * 80)
    recent = Message(role="assistant", content="most recent turn")
    chat_history = [user_turn, blob, *pair, recent]
    candidates = [msg for msg in chat_history if not msg.tool_calls and not msg.tool_call_id]

    measured_total = 200000
    budget = 120000
    counted_whole = estimate_messages_tokens(chat_history)
    blob_weight = estimate_messages_tokens([blob])
    # The blob alone must suffice to reach the budget, so the drop set is deterministic.
    assert measured_total - blob_weight * (measured_total / counted_whole) <= budget

    records, cleanup = _capture_fit_logs()
    try:
        kept = _fit_messages_to_budget(
            candidates,
            budget=budget,
            measured_total=measured_total,
            measured_request_counted_tokens=counted_whole,
        )
    finally:
        cleanup()

    assert [msg.content for msg in kept] == [user_turn.content, recent.content]
    info_lines = [record for record in records if "estimated tokens against budget" in record]
    assert len(info_lines) == 1
    reported_total = int(info_lines[0].split("-> ")[1].split(" estimated")[0])
    expected_total = measured_total - blob_weight * (measured_total / counted_whole)
    assert reported_total == int(expected_total)
    # The candidates-only denominator of the old implementation would have credited the same drop
    # with measured_total / counted(candidates) and reported a visibly different figure.
    counted_candidates = estimate_messages_tokens(candidates)
    assert counted_candidates < counted_whole


def test_fit_scale_attribution_through_tier_two_with_units() -> None:
    """With scale != 1 the second tier still drops complete units and the credited total fits."""
    assistant, tool_result = _make_shape_i_pair(tool_result_content="r" * 400)
    user_mid = Message(role="user", content="mid dialogue turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [assistant, tool_result, user_mid, recent]

    counted = estimate_messages_tokens(messages)
    pair_weight = estimate_messages_tokens([assistant]) + estimate_messages_tokens([tool_result])
    measured_total = counted * 10
    budget = measured_total - pair_weight * 10

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=budget,
        measured_total=measured_total,
        measured_request_counted_tokens=counted,
    )

    # Exactly the pair unit goes; the credited total lands on the budget.
    assert [msg.content for msg in kept] == [user_mid.content, recent.content]


def test_complete_units_absorbs_non_contiguous_tool_results() -> None:
    """A tool result separated from its assistant call by dialogue turns joins the assistant's unit."""
    assistant, tool_result = _make_shape_i_pair(tool_result_content="result payload")
    user_mid = Message(role="user", content="mid turn")
    messages = [assistant, user_mid, tool_result]

    units = _complete_units(messages)

    assert units[0] == [assistant, tool_result]
    assert units[1] == [user_mid]


def test_fit_never_orphans_non_contiguous_tool_result() -> None:
    """Dropping the assistant unit also removes a non-contiguous result: no orphan is ever left."""
    assistant, tool_result = _make_shape_i_pair(tool_result_content="r" * 400)
    user_mid = Message(role="user", content="mid dialogue turn")
    recent = Message(role="assistant", content="most recent turn")
    messages = [assistant, user_mid, tool_result, recent]

    pair_weight = estimate_messages_tokens([assistant]) + estimate_messages_tokens([tool_result])
    measured_total = estimate_messages_tokens(messages)

    kept = _fit_messages_to_budget(
        deepcopy(messages),
        budget=measured_total - pair_weight,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )

    # The assistant and its (non-contiguous) result go together; the dialogue between them survives
    # and no orphan tool result remains.
    assert [msg.content for msg in kept] == [user_mid.content, recent.content]


def test_verify_compaction_decides_success_from_provider_tokens_not_estimate() -> None:
    """The authoritative mode consumes the provider-reported figure, not the local estimate.

    The len//4 heuristic for the same request would claim a comfortable fit (the request
    would look ~4.8x smaller than reality), so an implementation that sized the post-compaction
    request locally could not record an over-budget failure here. The provider-reported 150000
    against the 128000 window with a 32000 reservation is over budget, and the check must say so.
    """
    outcome = verify_compaction(
        post_tokens=150000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )

    assert outcome is CompactionOutcome.REDUCED_BUT_OVER_BUDGET
    assert outcome is not CompactionOutcome.REDUCED_AND_FITS


def test_verify_compaction_rejects_compaction_that_did_not_reduce() -> None:
    """A compaction that did not reduce the size is a failure even when the result would fit.

    Load-bearing guard: a no-op summarizer call must never be recorded as a successful
    compaction, so post_tokens == previous_tokens yields DID_NOT_REDUCE regardless of the
    fit arithmetic. The failure shapes are provenance-independent: acting on a predicted
    no-reduction is safe (conservative), so both modes reject it.
    """
    fitting = verify_compaction(
        post_tokens=50000,
        previous_tokens=50000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )
    fitting_predicted = verify_compaction(
        post_tokens=50000,
        previous_tokens=50000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PREDICTED,
    )
    grown = verify_compaction(
        post_tokens=60000,
        previous_tokens=50000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )

    assert fitting is CompactionOutcome.DID_NOT_REDUCE
    assert fitting_predicted is CompactionOutcome.DID_NOT_REDUCE
    assert grown is CompactionOutcome.DID_NOT_REDUCE


def test_predicted_post_size_can_never_yield_verified_success() -> None:
    """A predicted post-compaction figure can never come back as REDUCED_AND_FITS.

    The exact same numbers under the two provenances must split: the provider-reported
    figure that fits is a verified success, the predicted one is only PREDICTED_FITS,
    an advisory result awaiting the provider's confirmation. The default (a caller that
    omits the provenance) is the safe predicted mode, never a claimed verification.
    """
    reported = verify_compaction(
        post_tokens=90000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )
    predicted = verify_compaction(
        post_tokens=90000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PREDICTED,
    )
    defaulted = verify_compaction(post_tokens=90000, previous_tokens=160000, requested_max_tokens=32000, budget=128000)

    assert reported is CompactionOutcome.REDUCED_AND_FITS
    assert predicted is CompactionOutcome.PREDICTED_FITS
    assert defaulted is CompactionOutcome.PREDICTED_FITS
    assert predicted is not CompactionOutcome.REDUCED_AND_FITS


def test_provider_reported_post_size_yields_verified_success_only_when_it_fits() -> None:
    """PROVIDER_REPORTED is the authoritative mode: REDUCED_AND_FITS when it fits, rejection otherwise."""
    fits = verify_compaction(
        post_tokens=90000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )
    over_budget = verify_compaction(
        post_tokens=120000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )

    assert fits is CompactionOutcome.REDUCED_AND_FITS
    assert over_budget is CompactionOutcome.REDUCED_BUT_OVER_BUDGET


def test_verify_compaction_outcomes_distinguishable() -> None:
    """Reduced-and-fits, reduced-but-over-budget, did-not-reduce and predicted-fits are distinct."""
    reduced_and_fits = verify_compaction(
        post_tokens=90000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )
    predicted_fits = verify_compaction(
        post_tokens=90000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PREDICTED,
    )
    reduced_but_over = verify_compaction(
        post_tokens=150000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )
    did_not_reduce = verify_compaction(
        post_tokens=160000,
        previous_tokens=160000,
        requested_max_tokens=32000,
        budget=128000,
        post_provenance=TokenProvenance.PROVIDER_REPORTED,
    )

    assert len({reduced_and_fits, predicted_fits, reduced_but_over, did_not_reduce}) == 4
    assert reduced_and_fits is CompactionOutcome.REDUCED_AND_FITS
    assert predicted_fits is CompactionOutcome.PREDICTED_FITS
    assert reduced_but_over is CompactionOutcome.REDUCED_BUT_OVER_BUDGET
    assert did_not_reduce is CompactionOutcome.DID_NOT_REDUCE


@pytest.mark.asyncio
async def test_predicted_fit_records_prediction_not_verification() -> None:
    """The state machine's post-compaction check is a PREDICTION and its log must say so.

    After the replacement no provider figure exists for the compacted history (the durable
    record is dropped by design), so the machine feeds the predicted post-size into the
    advisory mode: the best it may record is PREDICTED_FITS. The log line says "predicted",
    never claims verification, and anchors the reduction check on the pre-replacement
    measured figure (50000) rather than the raw counted estimate of the same history (40001),
    against which the identical post figure (45000) would read as did_not_reduce.
    """
    messages = [Message(role="user", content="u" * 160000)]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    records, cleanup = _capture_fit_logs()
    holder: dict[str, Any] = {}
    try:
        replace_mock = await _run_emergency_summarization(
            db,
            max_history_tokens=200000,
            measured_total=50000,
            post_measured_total=45000,
            result_holder=holder,
        )
    finally:
        cleanup()

    assert replace_mock.await_count == 1
    assert estimate_messages_tokens(messages) == 40001
    assert any("Compaction predicted to fit" in record for record in records)
    assert any("50000 -> 45000" in record for record in records)
    assert not any("Compaction verified" in record for record in records)
    result = holder["result"]
    assert isinstance(result, CompactionResult)
    assert result.outcome is CompactionOutcome.PREDICTED_FITS
    assert result.fallback_ran is False
    assert result.fallback_fitted is None


@pytest.mark.asyncio
async def test_no_reduction_compaction_is_rejected_and_last_resort_runs() -> None:
    """A compaction whose result is not smaller is rejected and the last-resort path runs.

    The predicted post-compaction measurement equals the pre-compaction one (a no-op
    summarizer call), and the resulting request is over the budget (max_history_tokens
    fallback stands in for the window), so the machine must log the did_not_reduce
    rejection and shrink the seed through the replacement helper instead of treating
    the compaction as a success. The equality is fed through the patched measurement
    entry point: in the live flow the guard is defensive, not reachable (the pre side
    is the oversized history, the post side the two-message seed). This test proves
    the guard's behaviour, not its production reachability.
    """
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    records, cleanup = _capture_fit_logs()
    holder: dict[str, Any] = {}
    try:
        replace_mock = await _run_emergency_summarization(
            db,
            max_history_tokens=100000,
            measured_total=90000,
            post_measured_total=90000,
            result_holder=holder,
        )
    finally:
        cleanup()

    assert replace_mock.await_count == 2
    assert any("did_not_reduce" in record for record in records)
    assert any("post side predicted" in record for record in records)
    assert not any("Compaction verified" in record for record in records)
    result = holder["result"]
    assert isinstance(result, CompactionResult)
    assert result.outcome is CompactionOutcome.DID_NOT_REDUCE
    assert result.fallback_ran is True
    assert result.fallback_fitted is True
    shrunk_seed = replace_mock.await_args_list[1].kwargs["seed"]
    assert len(shrunk_seed) == 2


@pytest.mark.asyncio
async def test_no_reduction_compaction_rejected_even_when_result_fits() -> None:
    """DID_NOT_REDUCE is a rejection on its own: no success is recorded even though it fits.

    The fallback still runs (and reports that the request fits), but its no-op shrink
    changes nothing, so the seed is not rewritten: exactly one replacement, one recorded
    compaction outcome, and a result value that says fallback_ran without a second write.
    """
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    records, cleanup = _capture_fit_logs()
    holder: dict[str, Any] = {}
    try:
        replace_mock = await _run_emergency_summarization(
            db,
            max_history_tokens=200000,
            measured_total=90000,
            post_measured_total=90000,
            result_holder=holder,
        )
    finally:
        cleanup()

    assert replace_mock.await_count == 1
    assert any("did_not_reduce" in record for record in records)
    assert not any("Compaction verified" in record for record in records)
    result = holder["result"]
    assert result.outcome is CompactionOutcome.DID_NOT_REDUCE
    assert result.fallback_ran is True
    assert result.fallback_fitted is True


@pytest.mark.asyncio
async def test_reduced_but_over_budget_runs_last_resort_and_persists_shrunk_seed() -> None:
    """Reduced but still over the window: the shrunk seed is persisted with ids intact.

    gpt-4o resolves a 128000 window; with the 32000 reservation the post-compaction request
    (149999 mocked) is over budget, so the last-resort path runs and the second replacement
    writes the shrunk seed, the same Message objects, ids and source marker unchanged.
    """
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages), model="gpt-4o")
    holder: dict[str, Any] = {}

    replace_mock = await _run_emergency_summarization(
        db,
        max_history_tokens=200000,
        measured_total=150000,
        post_measured_total=149999,
        result_holder=holder,
    )

    result = holder["result"]
    assert result.outcome is CompactionOutcome.REDUCED_BUT_OVER_BUDGET
    # The structured distinction the observability work item branches on: the compaction
    # itself was rejected, and the fallback ran and produced a fitting request.
    assert result.fallback_ran is True
    assert result.fallback_fitted is True
    assert replace_mock.await_count == 2
    original_seed = replace_mock.await_args_list[0].kwargs["seed"]
    shrunk_seed = replace_mock.await_args_list[1].kwargs["seed"]
    # The shrink mutates the seed messages in place, so both recorded calls reference the same
    # objects: the proof of id preservation is that both calls carry the identical id list.
    assert [message.id for message in shrunk_seed] == [message.id for message in original_seed]
    assert all(message.source == COMPACTION_SEED_SOURCE for message in shrunk_seed)
    # The seed question was truncated (the unit overshoots the excess, so it is truncated, not
    # dropped), and the last message, the summary, the current turn, survived untouched.
    assert len(shrunk_seed[0].content) < len("What we were talking about?")
    assert shrunk_seed[1].content == "summary"


def test_fit_log_reports_tier_identity_and_never_content() -> None:
    """The fit's drop log carries tier counts and id/role identity, never message content."""
    blob = _make_tool_response_blob("web_search", "BLOB_MARKER_PAYLOAD" + "x" * 36000)
    user_turn = Message(role="user", content="a" * 80)
    recent = Message(role="assistant", content="most recent turn")
    candidates = [user_turn, blob, recent]

    measured_total = 200000
    budget = 120000
    counted_whole = estimate_messages_tokens([user_turn, blob, recent])

    records, cleanup = _capture_fit_logs()
    try:
        kept = _fit_messages_to_budget(
            candidates,
            budget=budget,
            measured_total=measured_total,
            measured_request_counted_tokens=counted_whole,
        )
    finally:
        cleanup()

    assert [msg.content for msg in kept] == [user_turn.content, recent.content]
    info_lines = [record for record in records if "estimated tokens against budget" in record]
    assert len(info_lines) == 1
    line = info_lines[0]
    assert f"tool_response_blobs=1 [{blob.id}:user]" in line
    assert "oldest_units=0 []" in line
    # Message content never reaches the log: neither the blob payload nor the filler.
    assert all("BLOB_MARKER_PAYLOAD" not in record for record in records)
    assert all("most recent turn" not in record for record in records)


@pytest.mark.asyncio
async def test_compaction_decision_log_carries_size_provenance_and_budget() -> None:
    """The compaction decision line states the observed size, its provenance and the fit budget."""
    messages = [Message(role="user", content="u" * 160000)]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    records, cleanup = _capture_fit_logs()
    holder: dict[str, Any] = {}
    try:
        await _run_emergency_summarization(
            db,
            max_history_tokens=200000,
            measured_total=50000,
            post_measured_total=45000,
            result_holder=holder,
        )
    finally:
        cleanup()

    assert any(
        "Emergency summarization for user 1 thread 0: measured request size 50000 tokens (provenance=estimated)"
        in record
        and "summarizer fit budget" in record
        for record in records
    )
    assert any(
        "Compaction predicted to fit" in record and "against the 200000-token budget" in record for record in records
    )


@pytest.mark.asyncio
async def test_counters_distinguish_compaction_success_from_fallback() -> None:
    """A predicted-fit compaction counts as compaction_succeeded and no fallback counter moves."""
    from chibi.services.context_counters import ContextDecisionCounters

    ContextDecisionCounters().reset()
    messages = [Message(role="user", content="u" * 160000)]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    holder: dict[str, Any] = {}
    await _run_emergency_summarization(
        db,
        max_history_tokens=200000,
        measured_total=50000,
        post_measured_total=45000,
        result_holder=holder,
    )

    result = holder["result"]
    assert result.outcome is CompactionOutcome.PREDICTED_FITS
    assert result.fallback_ran is False
    counters = ContextDecisionCounters().snapshot()
    assert counters.get("compaction_succeeded") == 1
    assert "compaction_fallback_succeeded" not in counters
    assert "compaction_fallback_failed" not in counters
    assert "compaction_skipped" not in counters


@pytest.mark.asyncio
async def test_counters_distinguish_fallback_success_from_failure() -> None:
    """A rejected compaction counts the rejection and whether the fallback fitted."""
    from chibi.services.context_counters import ContextDecisionCounters

    # Rejected as did_not_reduce; the fallback shrink makes the request fit.
    ContextDecisionCounters().reset()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    holder: dict[str, Any] = {}
    await _run_emergency_summarization(
        db,
        max_history_tokens=100000,
        measured_total=90000,
        post_measured_total=90000,
        result_holder=holder,
    )
    assert holder["result"].fallback_fitted is True
    counters = ContextDecisionCounters().snapshot()
    assert counters.get("compaction_rejected_did_not_reduce") == 1
    assert counters.get("compaction_fallback_succeeded") == 1
    assert "compaction_fallback_failed" not in counters

    # Rejected again; this time even the fallback cannot make the request fit.
    ContextDecisionCounters().reset()
    messages = [Message(role="user", content="hello"), Message(role="assistant", content="hi")]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    holder = {}
    await _run_emergency_summarization(
        db,
        max_history_tokens=100000,
        measured_total=90000,
        post_measured_total=10000000,
        result_holder=holder,
    )
    assert holder["result"].fallback_fitted is False
    counters = ContextDecisionCounters().snapshot()
    assert counters.get("compaction_rejected_did_not_reduce") == 1
    assert counters.get("compaction_fallback_failed") == 1
    assert "compaction_fallback_succeeded" not in counters


@pytest.mark.asyncio
async def test_skipped_compaction_is_countable() -> None:
    """A machine run that attempts no compaction (empty input) counts as compaction_skipped."""
    from chibi.services.context_counters import ContextDecisionCounters

    ContextDecisionCounters().reset()
    messages = [Message(role="user", content="")]
    db, _provider = _make_summarization_mocks(deepcopy(messages))
    records, cleanup = _capture_fit_logs()
    holder: dict[str, Any] = {}
    try:
        replace_mock = await _run_emergency_summarization(
            db,
            max_history_tokens=100000,
            result_holder=holder,
        )
    finally:
        cleanup()

    replace_mock.assert_not_awaited()
    assert holder["result"] is None
    assert any("Emergency summarization skipped" in record for record in records)
    counters = ContextDecisionCounters().snapshot()
    assert counters.get("compaction_skipped") == 1
    assert "compaction_succeeded" not in counters
