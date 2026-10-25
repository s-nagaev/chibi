import asyncio
import datetime
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import timezone
from enum import Enum
from functools import partial
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from chibi.config import gpt_settings
from chibi.constants import get_model_context_window
from chibi.exceptions import ContextLengthExceededError, NoProviderSelectedError
from chibi.models import Message, SelectedModel, TelegramFileMeta, User
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema, UsageSchema, VideoResult, VisionResultSchema
from chibi.services.context_counters import ContextDecisionCounters
from chibi.services.context_measurement import estimate_messages_tokens, measure_request_size_with_provenance
from chibi.services.failover import (
    FailoverPair,
    FailoverTrigger,
    failover_failure_message,
    failover_warning_message,
    run_with_failover,
)
from chibi.services.interface import EditorContextProvider, UserInterface
from chibi.services.lock_manager import LockManager
from chibi.services.usage_cache import UsageCacheStore
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database
from chibi.utils.ack_sentinel import is_pure_ack, strip_ack_sentinel

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider
    from chibi.services.providers.tools import ToolResponseSchema


@inject_database
async def get_chibi_user(db: Database, user_id: int) -> User:
    return await db.get_or_create_user(user_id=user_id)


@inject_database
async def set_active_model(db: Database, interface: UserInterface, model: ModelChangeSchema) -> None:
    user = await db.get_or_create_user(user_id=interface.storage_id)
    thread_id = interface.thread_id
    if model.video_generation:
        user.thread_selected_video_model[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    elif model.image_generation:
        user.thread_selected_image_model[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    else:
        user.thread_selected_llm[thread_id] = SelectedModel(name=model.name, provider_name=model.provider)
    await db.save_user(user)


@inject_database
async def reset_chat_history(db: Database, storage_id: int, thread_id: int) -> None:
    user = await db.get_or_create_user(user_id=storage_id)
    await db.drop_messages(user=user, thread_id=thread_id)
    if thread_id in user.thread_usage:
        user = await db.get_or_create_user(user_id=storage_id)
        user.thread_usage.pop(thread_id, None)
        await db.save_user(user)


@inject_database
async def replace_thread_history(db: Database, storage_id: int, thread_id: int, seed: list[Message]) -> None:
    """Replace the entire stored history of a thread with the given seed messages.

    The single implementation of whole-thread replacement: it reuses the
    wipe-then-write shape of the existing reset path. ``reset_chat_history``
    wipes the whole thread (``drop_messages`` deletes everything on every
    backend, there is no range delete and no per-message delete), then the
    seed is written back message by message. On the local backend the write
    half is atomic (the user pickle is written to a temp file and moved into
    place with ``os.replace``); on Redis and DynamoDB it is the whole-thread
    delete followed by the seed write, exactly as the reset path does.
    Seed marking (``Message.source``) is the caller's job: the helper is a
    pure replacement primitive shared by the emergency summarization, the
    tool-side compaction and the history cleanup.

    Crash safety between the wipe and the seed write is an explicitly
    accepted risk (owner decision, 2026-10-20): the window is milliseconds,
    and the worst case is one thread reading as empty, not a corrupted
    state.

    Args:
        db: Database instance (injected).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        thread_id: The message thread identifier (0 for the main thread).
        seed: The messages the thread history becomes, in stored order.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    # The db is passed through explicitly (via __wrapped__) so the whole
    # replacement chain runs against the same injected backend; a wrapped
    # call would re-resolve the singleton instead.
    await reset_chat_history.__wrapped__(db, storage_id=storage_id, thread_id=thread_id)
    for message in seed:
        await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
    # The cached prompt size reflects the pre-replacement history; without
    # this the next turn would read a stale context size.
    UsageCacheStore().invalidate(user_id=storage_id, thread_id=thread_id)


# Fallback share of gpt_settings.max_history_tokens for the token budget of
# the emergency summarization request input, used ONLY when the context window
# of the summarizer model cannot be resolved via
# :func:`chibi.constants.get_model_context_window`. The primary budget source
# is the resolvable window minus the output reservation (see
# :func:`_summarizer_fit_budget`); this share remains as the documented
# fallback so an unknown model still gets a working, bounded budget instead of
# an unbounded request.
EMERGENCY_SUMMARIZATION_INPUT_BUDGET = 0.6

# Marker written into ``Message.source`` for every message a compaction
# writes as the new thread history (the emergency seed and the tool-side
# seed alike). It is the only idempotency the seed needs: a repeated
# compaction recognises its own seed and REPLACES the thread through
# :func:`replace_thread_history` instead of appending to it.
COMPACTION_SEED_SOURCE = "compaction_seed"

COMPACTION_QUESTION = "What we were talking about?"


def build_compaction_seed(summary: str) -> list[Message]:
    """Build the 2-message compaction seed (question + summary).

    The single construction site for the seed literal: the automatic
    compaction path and the ``summarize_history`` tool both build their seed
    here and both write it through the single replacement helper
    :func:`replace_thread_history`.

    Args:
        summary: The summary text the seed's assistant message carries.

    Returns:
        The marked seed messages, in stored order.
    """
    return [
        Message(role="user", content=COMPACTION_QUESTION, source=COMPACTION_SEED_SOURCE),
        Message(role="assistant", content=summary, source=COMPACTION_SEED_SOURCE),
    ]


def _summarizer_max_tokens(provider: "Provider", model: str | None) -> int:
    """Output reservation for the summarizer request, derived from the failed request.

    The reservation is resolved through the SAME path the failed turn's
    request used: when a request passes no explicit ``max_tokens``, both
    request builders fall back to ``Provider._get_max_tokens_value`` (the
    generic path at ``chibi/services/providers/provider.py:935`` and the
    OpenAI Responses path at ``chibi/services/providers/openai.py:371``), so
    the summarizer resolves its own cap through that same method. When the
    resolution yields a concrete integer cap, the summarizer reserves
    exactly that much: it may never reserve more output than the request
    that just failed, because a larger reservation shrinks the input space
    of a request that failed precisely on input size.

    When the failed request carried NO output cap, there is nothing to
    compare against and the reservation resolves to the configured
    ``gpt_settings.max_tokens`` default. This is a real case, not a
    degenerate one: the OpenAI family sets ``max_tokens = NOT_GIVEN``
    (``chibi/services/providers/openai.py:51``), so its failed requests send
    ``max_output_tokens: NOT_GIVEN``, which the SDK omits, and the model's
    own provider-side default governs the output size. That default is not
    reachable from here, so the documented limit is: for such providers the
    invariant "never more output than the failed request" holds while the
    provider-side default is at least the configured default (the shipped
    ``32000``), and cannot be guaranteed against a smaller one.

    This is also the single shared expression for the output reservation:
    the fit budget (:func:`_summarizer_fit_budget`) subtracts exactly what
    the summarizer call requests, the pre-call budget guard
    (:func:`_enforce_context_budget`), the in-loop budget check
    (:func:`enforce_in_loop_budget`) and the reactive recovery
    (:func:`_reactive_context_recovery`) resolve the very same value, so
    the sides can never drift apart.

    Args:
        provider: The provider instance the summarizer call will use, the
            same provider the failed request used.
        model: The model identifier the summarizer call will use.

    Returns:
        The max_tokens value for the summarizer request.
    """
    # The declared return type of _get_max_tokens_value is ``int``, but the
    # OpenAI family returns the ``NOT_GIVEN`` sentinel in practice (its class
    # attribute ``max_tokens = NOT_GIVEN`` wins the ``getattr``), so the value
    # is narrowed by hand: only a concrete integer cap is a reservation the
    # failed request actually sent. ``model or ""`` mirrors the failed path:
    # the request builders pass a concrete model string here (the outer call
    # resolves None to the provider default first), and the resolution ignores
    # the model name anyway, the class attribute decides.
    failed_request_reservation: Any = provider._get_max_tokens_value(model_name=model or "")
    if isinstance(failed_request_reservation, int):
        return failed_request_reservation
    # No cap was sent by the failed request (``NOT_GIVEN`` or ``None``):
    # nothing to compare against, resolve to the configured default.
    return gpt_settings.max_tokens


def _summarizer_fit_budget(provider: "Provider", model: str | None) -> int:
    """Token budget the summarizer request input has to fit into.

    The budget is the space the summarizer call actually has: the context
    window of the model that will make the summarizer call (the thread's
    active model), minus the output reservation
    (:func:`_summarizer_max_tokens`, exactly the ``max_tokens`` the call
    itself requests, derived from the failed request's own reservation).
    Full collapse is the intended normal case, so the budget exists only as
    admission control for that single call and is as large as the window
    allows.

    When the window cannot be resolved (unknown or unset model), the
    historical share of ``max_history_tokens``
    (:data:`EMERGENCY_SUMMARIZATION_INPUT_BUDGET`) is used as a documented
    fallback: on that path the fit can trim a history the real window would
    have admitted, and full collapse is not guaranteed.

    Args:
        provider: The provider instance the summarizer call will use.
        model: The model identifier the summarizer call will use.

    Returns:
        The fit budget in tokens; zero or negative when the window does not
        exceed the output reservation, in which case the fit degrades to
        dropping everything droppable while always keeping the most recent
        message.
    """
    window = get_model_context_window(model)
    if window is None:
        return int(gpt_settings.max_history_tokens * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    return window - _summarizer_max_tokens(provider, model)


def _is_tool_response_message(message: Message) -> bool:
    """Detect background tool-response turns stored as role="user" JSON blobs.

    Args:
        message: The message to inspect.

    Returns:
        True when the message content is a JSON object with "type": "tool response".
    """
    if message.role != "user" or not message.content:
        return False
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("type") == "tool response"


def _complete_units(messages: list[Message]) -> list[list[Message]]:
    """Group messages into drop-safe units along tool-call/tool-result pairing.

    An assistant message carrying ``tool_calls`` forms one unit together with
    EVERY ``role="tool"`` result whose ``tool_call_id`` matches one of its
    calls, regardless of position in the list. The guarantee is
    unconditional: dropping such a unit can never separate a tool call from
    its results, and a result is never left behind without its request, even
    when dialogue turns sit between the assistant message and its result (the
    grouping therefore does not preserve list order inside a unit; every
    message still belongs to exactly one unit). An orphan ``role="tool"``
    result (no assistant carrying its ``tool_call_id`` is part of the list) is
    a unit of its own: removing it whole is safe and is the fix for an
    orphan, keeping it is not. Every other message is a unit of its own.

    Args:
        messages: The messages to group, in stored order.

    Returns:
        The drop-safe units, anchored in stored order (a unit starts at its
        assistant message, or at the message itself for singletons).
    """
    owner_by_call_id: dict[str, Message] = {}
    for message in messages:
        if message.tool_calls:
            for tool_call in message.tool_calls:
                owner_by_call_id.setdefault(tool_call.id, message)

    results_by_owner: dict[int, list[Message]] = {}
    consumed: set[int] = set()
    for message in messages:
        if message.role == "tool" and message.tool_call_id and message.tool_call_id in owner_by_call_id:
            owner = owner_by_call_id[message.tool_call_id]
            results_by_owner.setdefault(id(owner), []).append(message)
            consumed.add(id(message))

    units: list[list[Message]] = []
    for message in messages:
        if id(message) in consumed:
            continue
        if message.tool_calls:
            units.append([message, *results_by_owner.get(id(message), [])])
            continue
        units.append([message])
    return units


def _message_identities(messages: list[Message]) -> str:
    """Render the identity of messages for a log line: id and role only.

    Identity is enough to reconstruct which messages a decision dropped or truncated;
    message content is user data and must never reach the logs.

    Args:
        messages: The messages to describe, in the order they should be listed.

    Returns:
        A comma-separated ``<id>:<role>`` list, empty (between brackets) for no messages.
    """
    return ",".join(f"{msg.id}:{msg.role}" for msg in messages)


def _fit_messages_to_budget(
    messages: list[Message],
    budget: int,
    measured_total: int,
    measured_request_counted_tokens: int,
) -> list[Message]:
    """Build a transient summarizer input under the given token budget.

    The input is a separate construction: the stored conversation history is
    never mutated. Shape-(i) tool-call pairs (assistant ``tool_calls`` +
    role="tool" results) are already excluded from the summarizer input by the
    caller and never appear here; the pair-aware unit grouping below is
    therefore a structural no-op on this candidate set today, but it is the
    rule the second tier must follow wherever it drops from the oldest end
    (the truncation work item reuses it over lists that do contain pairs).

    Size accounting: the decision "does it fit" is made on the real-scale
    request size ``measured_total``, the prediction of the single measurement
    entry point (:func:`chibi.services.context_measurement.measure_request_size`),
    a durable provider-measured base plus a counted appended part, never on a
    per-message character heuristic. The credited reduction of every dropped
    message must itself be a real-scale quantity, so the attribution ratio is
    computed against ``measured_request_counted_tokens``, the counted total of
    the SAME list that was measured, and each dropped message reduces the
    running total by its counted weight times that ratio. The measured total
    exceeds the counted sum of the measured list by the request overhead
    (system prompt, tool schemas, JSON envelopes, tokenizer calibration), and
    this proportional attribution models dropping a fraction of the measured
    content as removing the same fraction of the measured request. Computing
    the ratio against the candidates-only count instead would mix scales: the
    numerator covers the whole conversation request while the denominator
    covers only the summarizable subset, inflating every credit by the share
    of non-droppable overhead in the measured request and making the fit stop
    while the request it claims to admit is still far above the budget.

    Dropping order when over budget:

    1. Background tool-response turns (role="user" JSON blobs with
       ``"type": "tool response"``), LARGEST first — they are the bulk of the
       volume and the least valuable as summarization context.
    2. The oldest remaining complete units (see :func:`_complete_units`).

    The most recent message is never dropped by either tier so the summarizer
    always keeps the latest context anchor.

    Outcomes are distinguishable in the code path for observability: a fit
    (nothing dropped, measured total already within the budget) logs at debug
    level; a drop logs what was removed; an over-budget request that could not
    be reduced further (nothing left to drop) logs a warning.

    Args:
        messages: Candidate messages (already filtered from tool-result turns).
        budget: Maximum total tokens for the kept messages.
        measured_total: Predicted request size in tokens from the single
            measurement function, for the request these candidates feed.
        measured_request_counted_tokens: Tokenizer count of the same message
            list ``measured_total`` was measured against (the full stored
            conversation, not just the candidates), from the same measurement
            module. Anchors the per-drop credit to the measured scale.

    Returns:
        The kept messages, in original order, fitting the budget as closely
        as the droppable material allows.
    """
    weights = {id(msg): estimate_messages_tokens([msg]) for msg in messages}
    if measured_total <= budget:
        logger.debug(
            f"Summarizer fit: measured request size {measured_total} tokens fits the "
            f"budget {budget}; no messages dropped."
        )
        return list(messages)

    counted_total = sum(weights.values())
    if counted_total <= 0 or measured_request_counted_tokens <= 0:
        logger.warning(
            f"Summarizer fit: measured request size {measured_total} tokens exceeds the budget {budget} "
            "but the candidates have no countable weight; returning the candidates unchanged."
        )
        return list(messages)
    # One proportional scale for the whole fit: measured tokens per counted
    # token of the SAME list the measured total was taken over.
    scale = measured_total / measured_request_counted_tokens

    dropped: set[int] = set()
    tier_blob_drops: list[Message] = []
    tier_unit_drops: list[Message] = []
    total: float = measured_total
    last_id = id(messages[-1]) if messages else None

    # (1) Background tool-response JSON blobs, largest first. The most recent
    # message is protected here too: even when it is a blob, dropping it would
    # leave the summarizer without the current turn.
    tool_responses = sorted(
        (msg for msg in messages if _is_tool_response_message(msg) and id(msg) != last_id),
        key=lambda msg: weights[id(msg)],
        reverse=True,
    )
    for msg in tool_responses:
        if total <= budget:
            break
        dropped.add(id(msg))
        tier_blob_drops.append(msg)
        total -= weights[id(msg)] * scale

    # (2) Oldest remaining complete units; never the unit holding the most
    # recent message.
    if total > budget:
        remaining = [msg for msg in messages if id(msg) not in dropped]
        for unit in _complete_units(remaining):
            if total <= budget:
                break
            if last_id is not None and any(id(msg) == last_id for msg in unit):
                continue
            for msg in unit:
                dropped.add(id(msg))
                tier_unit_drops.append(msg)
                total -= weights[id(msg)] * scale

    kept = [msg for msg in messages if id(msg) not in dropped]
    if not dropped:
        logger.warning(
            f"Summarizer fit: measured request size {measured_total} tokens exceeds the budget {budget} "
            "but there is nothing left to drop; returning the candidates unchanged."
        )
    else:
        logger.info(
            f"Summarizer fit: dropped {len(dropped)} message(s) "
            f"({measured_total} -> {int(total)} estimated tokens against budget {budget}). "
            f"tiers: tool_response_blobs={len(tier_blob_drops)} [{_message_identities(tier_blob_drops)}] "
            f"oldest_units={len(tier_unit_drops)} [{_message_identities(tier_unit_drops)}]."
        )
    return kept


class TokenProvenance(str, Enum):
    """Whether a token figure was reported by the provider or predicted locally.

    A caller that feeds a number into :func:`verify_compaction` must state where
    the number came from: a provider-reported ``prompt_tokens`` is authoritative,
    a locally computed prediction is not. The distinction is enforced here, at
    the only place a success can be decided, so a predicted figure can never be
    passed off as a verified one.
    """

    PROVIDER_REPORTED = "provider_reported"
    PREDICTED = "predicted"


class CompactionOutcome(str, Enum):
    """The outcomes of a compaction verification, kept distinguishable.

    A caller (and the observability work item) must be able to tell apart a
    compaction that reduced the size and fits, one that reduced the size but
    still does not fit, and one that did not reduce the size at all. The
    outcomes are never collapsed into a boolean: the last-resort path runs
    only for the two failure shapes, and a no-op summarizer call can never be
    recorded as a successful compaction.

    :attr:`REDUCED_AND_FITS` is reachable ONLY with a provider-reported
    post-compaction figure (``post_provenance=TokenProvenance.PROVIDER_REPORTED``).
    A predicted post-compaction size that appears to fit yields
    :attr:`PREDICTED_FITS` instead: an advisory result that explicitly waits
    for the provider to confirm. This makes it impossible to silently upgrade
    a predicted number into a verified success.
    """

    REDUCED_AND_FITS = "reduced_and_fits"
    REDUCED_BUT_OVER_BUDGET = "reduced_but_over_budget"
    DID_NOT_REDUCE = "did_not_reduce"
    PREDICTED_FITS = "predicted_fits"


@dataclass
class CompactionResult:
    """Structured result of a compaction attempt, for callers to branch on.

    ``emergency_summarization`` returns one so "the compaction succeeded" and
    "the fallback ran and succeeded" are distinguishable by a value, not only
    by a log line (the observability work item counts them separately).

    Attributes:
        outcome: The verification outcome of the compaction itself.
        fallback_ran: Whether the lossy last-resort path ran.
        fallback_fitted: Whether the fallback produced a request that fits;
            ``None`` when the fallback did not run.
    """

    outcome: CompactionOutcome
    fallback_ran: bool = False
    fallback_fitted: bool | None = None


def verify_compaction(
    post_tokens: int,
    previous_tokens: int,
    requested_max_tokens: int,
    budget: int,
    post_provenance: TokenProvenance = TokenProvenance.PREDICTED,
) -> CompactionOutcome:
    """Decide whether a compaction succeeded, with explicit number provenance.

    The exact success predicate is::

        post_tokens + requested_max_tokens <= budget

    Provenance contract: a figure marked
    :attr:`TokenProvenance.PROVIDER_REPORTED` must be a provider-reported
    ``prompt_tokens``, the durable ``ThreadUsageRecord.prompt_tokens`` behind
    the measurement entry point, or the usage the provider reported for the
    request just evaluated. This is the authoritative mode: only it can
    return :attr:`CompactionOutcome.REDUCED_AND_FITS`. A figure marked
    :attr:`TokenProvenance.PREDICTED` is a local prediction (the default, so
    an omitted provenance can never fake authority): its best case is
    :attr:`CompactionOutcome.PREDICTED_FITS`, an advisory result awaiting the
    provider's confirmation, never a verified success.

    The two failure shapes are independent of provenance: rejecting a
    compaction that did not reduce the size, or one that still does not fit,
    is actionable (and safe) on a prediction too.

    A compaction that did not reduce the size is a FAILURE, not a success:
    when ``post_tokens >= previous_tokens`` the outcome is
    :attr:`CompactionOutcome.DID_NOT_REDUCE` even when the resulting request
    would fit. This is the guard against a no-op summarizer call being
    recorded as a successful compaction.

    Args:
        post_tokens: Token size of the resulting request: provider-reported
            in the authoritative mode, predicted (via the single measurement
            entry point) in the advisory mode.
        previous_tokens: Token size of the request the compaction replaced,
            measured through the same entry point as ``post_tokens``. In the
            authoritative mode both figures are provider-reported (one scale).
            In the predicted mode the two sides may sit on different scales:
            the pre side is provider-anchored when a durable record existed,
            while the post side is the counted estimate, which UNDER-reports
            a real request. The reduction check is therefore not conservative
            in direction, the estimate biases toward "reduced"/"fits", and
            its soundness rests on the order-of-magnitude margin between an
            oversized history and the bounded two-message seed, not on the
            estimator's pessimism.
        requested_max_tokens: The output reservation the next request will
            carry.
        budget: The token budget the resulting request must fit, the model's
            context window, since the reservation is added by the predicate.
        post_provenance: Whether ``post_tokens`` was provider-reported or
            predicted locally. Defaults to the safe value: a caller that
            forgets to declare provenance gets the advisory mode, never a
            claimed verification.

    Returns:
        One of the four distinguishable outcomes; ``REDUCED_AND_FITS`` is
        reachable only from a provider-reported ``post_tokens``.
    """
    reduced = post_tokens < previous_tokens
    fits = post_tokens + requested_max_tokens <= budget
    if not reduced:
        return CompactionOutcome.DID_NOT_REDUCE
    if not fits:
        return CompactionOutcome.REDUCED_BUT_OVER_BUDGET
    if post_provenance is TokenProvenance.PROVIDER_REPORTED:
        return CompactionOutcome.REDUCED_AND_FITS
    return CompactionOutcome.PREDICTED_FITS


def _verification_budget(model: str | None) -> int:
    """Token budget the verify predicate compares against: the model's window.

    The predicate (:func:`verify_compaction`) adds the output reservation
    itself, so the budget is the FULL context window, not the input-space
    figure the summarizer fit uses. When the window does not resolve, the
    enforced threshold ``gpt_settings.max_history_tokens`` is the documented
    fallback: an unknown model is verified against the same bound the
    trigger uses, never against a guessed window.

    Args:
        model: The model identifier the next request will use.

    Returns:
        The verification budget in tokens.
    """
    window = get_model_context_window(model)
    if window is None:
        return gpt_settings.max_history_tokens
    return window


def last_resort_fit(
    messages: list[Message],
    budget: int,
    measured_total: int,
    measured_request_counted_tokens: int,
    truncate: bool = True,
    protected_first: bool = False,
) -> tuple[list[Message], bool]:
    """Synchronous last-resort shrink: drop oldest units, then truncate.

        This path loses information by design and is visibly distinct from a
        successful compaction: it runs only when a compaction did not produce a
        request that fits (or when no compaction ran and a send must be forced
        under the budget), and its logs never describe it as a compaction.

        Step one: drop the oldest complete units, reusing the shared pair
    rule (:func:`_complete_units`) rather than reimplementing it, an
        assistant ``tool_calls`` message is never dropped without its
        ``role="tool"`` results, and an orphan tool result is removed as its own
        unit. The unit holding the most recent message is never dropped. A unit
        is dropped whole only when its credit does not overshoot the remaining
        excess: dropping a message far larger than the gap it has to close would
        destroy more context than the budget requires, so such units are left
        for the truncation step, which removes most of their content while
        keeping the message, its id and its pair in place. This is the policy
        difference from the summarizer fit's second tier (which drops
        unconditionally, because overshoot is harmless there); the shared,
        non-divergent part is the pair-integrity grouping itself.


        Step two (skipped entirely in drop-only mode, ``truncate=False``):
        truncate the content of the largest remaining messages.
        Truncation rule (stated, and deterministic): **tool results first, then
        other messages, largest first within each class** (ties broken by stored
        position); only ``Message.content`` shrinks, message ids are never
        renumbered, recreated or re-keyed, and ``tool_calls`` / ``tool_call_id``
        are never touched, so a tool-call/tool-result pair stays complete (the
        pair rule applies to truncation too: an assistant message carrying
        ``tool_calls`` is not a truncation candidate, because its protocol role
        lives in the untouched structure). The last message is never dropped or
        truncated, the current turn survives. Same input list, same output.

        Step three (the backstop of the backstop): if truncation is exhausted
        and the request still does not fit, a whole unit whose overshoot-skip
        left it in place in step one is dropped after all, oldest first, with
        the overshoot guard disabled, until the request fits. Without this pass
        a large non-truncatable unit (an assistant ``tool_calls`` carrier with a
        bulky argument payload) could block the fit even though dropping that
        unit whole, exactly the loss this fallback exists to make, would have
        fitted. The pass is bounded (one deterministic sweep, oldest first) and
        pair-safe (whole units via :func:`_complete_units` only); it is a real
        last resort, so the extra loss is accepted by design.

        Size accounting matches :func:`_fit_messages_to_budget`: the decision is
        per counted token, never on the raw character heuristic.

        The truncation mutates the given ``Message`` objects in place (content
        only). Callers that persist the result do so through
        :func:`replace_thread_history`, which keeps the ids intact; callers that
        shrink a transient request list simply send the mutated objects.

        Args:
            messages: The message list to shrink, in stored order.
            budget: Maximum request-scale token total the kept messages may
                account for; the caller subtracts the output reservation from
                the window itself. Zero or negative means no message list can
                fit, and the shrink refuses to destroy the thread for nothing.
            measured_total: Predicted request size of the request these messages
                feed, from the single measurement entry point.
            measured_request_counted_tokens: Tokenizer count of the same message
                list ``measured_total`` was measured against.
            truncate: When True (the default) the content-truncation step runs
                after the unit drops. When False the shrink is drop-only: whole
                complete units are removed and no ``Message.content`` is ever
                modified. The in-loop budget check uses this mode, because the
                provider loop's request lists hold independently serialized
                copies of the same messages and an in-place truncation could not
                reach the actual request.
            protected_first: When True, the unit holding the FIRST message is
                never dropped (steps one and three), exactly like the unit
                holding the last message. The in-loop budget check uses this
                mode: many provider protocols require the conversation to open
                with its original first message (Anthropic requires a
                user-role first message), and losing it would trade a budget
                problem for an invalid request.

        Returns:
            A tuple of the shrunk messages (in original order, same objects with
            intact ids) and whether the credited total now fits the budget.
    """
    weights = {id(msg): estimate_messages_tokens([msg]) for msg in messages}
    if measured_total <= budget:
        return list(messages), True
    if budget <= 0:
        logger.error(
            f"Last-resort shrink: the output reservation alone exceeds the budget {budget}; "
            "no message list can fit, the thread is left untouched."
        )
        return list(messages), False
    counted_total = sum(weights.values())
    if counted_total <= 0 or measured_request_counted_tokens <= 0:
        logger.warning(
            f"Last-resort shrink: measured request size {measured_total} tokens exceeds the budget "
            f"{budget} but the messages have no countable weight; returning them unchanged."
        )
        return list(messages), False
    scale = measured_total / measured_request_counted_tokens

    dropped: set[int] = set()
    tier_oldest_unit_drops: list[Message] = []
    tier_sweep_drops: list[Message] = []
    truncated_messages: list[Message] = []
    total: float = float(measured_total)
    last_id = id(messages[-1]) if messages else None
    protected_ids: set[int] = set()
    if protected_first and messages:
        for unit in _complete_units(messages):
            if any(id(msg) == id(messages[0]) for msg in unit):
                protected_ids = {id(msg) for msg in unit}
                break

    # Step one: drop the oldest complete units; never the unit holding the
    # most recent message, never a unit whose drop would overshoot the
    # remaining excess (those are left for the truncation step), and, in
    # protected-first mode, never the unit holding the first message.
    for unit in _complete_units(messages):
        if total <= budget:
            break
        if last_id is not None and any(id(msg) == last_id for msg in unit):
            continue
        if any(id(msg) in protected_ids for msg in unit):
            continue
        unit_weight = sum(weights[id(msg)] for msg in unit) * scale
        if unit_weight > total - budget:
            continue
        for msg in unit:
            dropped.add(id(msg))
            tier_oldest_unit_drops.append(msg)
            total -= weights[id(msg)] * scale

    kept = [msg for msg in messages if id(msg) not in dropped]

    # Step two: truncate content. The rule is stated in the docstring and
    # holds here: tool results first, then other messages, largest first
    # within each class, ties by stored position; only content shrinks; the
    # last message is protected; assistant tool_calls carriers are excluded
    # so no half-truncated pair can be produced. Skipped entirely in
    # drop-only mode (truncate=False): the in-loop budget check uses this
    # mode, because the provider loop's request lists hold independently
    # serialized copies of the messages and an in-place content truncation
    # could not reach the actual request.
    truncated_ids: set[int] = set()
    if truncate:
        position = {id(msg): index for index, msg in enumerate(messages)}
        while total > budget:
            candidates = [msg for msg in kept if id(msg) != last_id and not msg.tool_calls and msg.content]
            if not candidates:
                break
            target = min(
                candidates,
                key=lambda msg: (0 if msg.role == "tool" else 1, -weights[id(msg)], position[id(msg)]),
            )
            target.content = target.content[: len(target.content) // 2]
            new_weight = estimate_messages_tokens([target])
            total -= (weights[id(target)] - new_weight) * scale
            weights[id(target)] = new_weight
            truncated_ids.add(id(target))
            truncated_messages.append(target)

    # Step three: if truncation is exhausted and the request still does not
    # fit, a unit the overshoot guard protected in step one is dropped whole
    # after all, oldest first, guard disabled, until it fits. This is what
    # keeps the "guarantees the next request fits" promise when the blocker
    # is a non-truncatable unit (an assistant tool_calls carrier): dropping
    # that unit whole is exactly the loss this fallback exists to make, so
    # refusing to make it would leave the request over the window. One
    # bounded, deterministic sweep; whole units only, so the pair rule holds;
    # the unit holding the most recent message is still never dropped, and
    # neither is the protected first unit.
    for unit in _complete_units(kept):
        if total <= budget:
            break
        if last_id is not None and any(id(msg) == last_id for msg in unit):
            continue
        if any(id(msg) in protected_ids for msg in unit):
            continue
        for msg in unit:
            dropped.add(id(msg))
            tier_sweep_drops.append(msg)
            total -= weights[id(msg)] * scale
    kept = [msg for msg in messages if id(msg) not in dropped]

    if total > budget:
        logger.error(
            f"Last-resort shrink: even dropping and truncating cannot bring the request under the "
            f"budget {budget} ({measured_total} -> {int(total)} tokens); the next request may still "
            f"be over the window. tiers: oldest_units={len(tier_oldest_unit_drops)} "
            f"[{_message_identities(tier_oldest_unit_drops)}] "
            f"sweep_units={len(tier_sweep_drops)} [{_message_identities(tier_sweep_drops)}] "
            f"truncated={len(truncated_messages)} [{_message_identities(truncated_messages)}]."
        )
    elif truncate:
        logger.warning(
            f"Last-resort shrink: {len(dropped)} message(s) dropped and {len(truncated_ids)} message(s) "
            f"content-truncated ({measured_total} -> {int(total)} tokens against budget {budget}); "
            f"this is a lossy last resort, not a compaction. tiers: "
            f"oldest_units={len(tier_oldest_unit_drops)} [{_message_identities(tier_oldest_unit_drops)}] "
            f"sweep_units={len(tier_sweep_drops)} [{_message_identities(tier_sweep_drops)}] "
            f"truncated={len(truncated_messages)} [{_message_identities(truncated_messages)}]."
        )
    else:
        logger.warning(
            f"Last-resort shrink: {len(dropped)} message(s) dropped whole, no content truncation "
            f"({measured_total} -> {int(total)} tokens against budget {budget}); "
            f"this is a lossy last resort, not a compaction. tiers: "
            f"oldest_units={len(tier_oldest_unit_drops)} [{_message_identities(tier_oldest_unit_drops)}] "
            f"sweep_units={len(tier_sweep_drops)} [{_message_identities(tier_sweep_drops)}]."
        )
    return kept, total <= budget


async def _enforce_context_budget(
    provider: "Provider",
    model: str | None,
    storage_id: int,
    thread_id: int,
    messages: list[Message],
) -> list[Message]:
    """Refuse to send an oversized request: truncate to the model window when it resolves.

    The single enforcement point of the context budget. It is called once,
    under the per-thread lock, on the one call site every client (Telegram,
    IDE/stdio, terminal), every provider path and streaming share
    (``get_llm_chat_completion_answer``, immediately before
    ``run_with_failover``), after the message list is fully assembled.

    Decision table:

    - The model window resolves
      (:func:`chibi.constants.get_model_context_window`) and the measured
      request plus the output reservation fits it: the request is sent
      unchanged.
    - The window resolves and the request does not fit: the request is NOT
      sent as-is. The synchronous last-resort path (:func:`last_resort_fit`)
      drops the oldest complete units and truncates the largest messages
      until the request fits the input space (window minus the output
      reservation), and the shrunk list is sent instead. The truncation is
      transient, the stored history is never touched, the next turn
      re-reads the full history, and the overflow corner case (the provider
      rejecting the request) is not reached on this path.
    - The window does not resolve (unknown model): the guard does not guess
      a window out of thin air and does not block on a wrong assumption. It
      falls through and lets the reactive backstop
      (:func:`_reactive_context_recovery`) carry the case. The division of
      responsibility is deliberate: the guard refuses only sends it can
      prove oversized; the backstop reacts to the provider's own verdict.

    The arithmetic accounts for the requested output reservation, not only
    the input: the reservation is the single shared expression
    (:func:`_summarizer_max_tokens`) resolved through the same provider
    path the request itself resolves ``max_tokens`` with, so an input that
    fits the window but leaves no room for the requested output is still an
    oversized request. The request size comes only from the single
    measurement entry point
    (:func:`chibi.services.context_measurement.measure_request_size`); no
    ad-hoc size calculation exists here. The guard never changes when
    compaction triggers and never changes the compaction threshold: it only
    prevents a send that would fail.

    When even the last-resort path cannot make the request fit (the
    protected most recent message alone exceeds the input space), the
    best-effort shrunk list is sent anyway and the outcome is logged as an
    error: the provider's verdict is the ground truth of last resort, and
    the reactive backstop remains behind it.

    Args:
        provider: The provider instance the request will use.
        model: The model identifier the request will use (None = provider
            default).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        thread_id: The message thread identifier (0 for the main thread).
        messages: The fully assembled request messages.

    Returns:
        The message list to send: the input unchanged when it fits (or the
        window does not resolve), the transiently shrunk list otherwise.
    """
    window = get_model_context_window(model)
    counters = ContextDecisionCounters()
    if window is None:
        counters.increment("context_guard_unknown_window")
        logger.debug(
            f"Context guard: no context window known for model {model or '<default>'}; the request is "
            "sent unmeasured and the reactive recovery remains the backstop for an oversized request."
        )
        return messages
    reservation = _summarizer_max_tokens(provider, model)
    measured_total, measured_provenance = await measure_request_size_with_provenance(
        user_id=storage_id,
        thread_id=thread_id,
        model=model or "",
        messages=messages,
    )
    if measured_total + reservation <= window:
        counters.increment("context_guard_noop")
        logger.debug(
            f"Context guard: measured request size {measured_total} tokens "
            f"(provenance={measured_provenance.value}) plus the {reservation}-token output reservation "
            f"fits the {window}-token window of {model or '<default>'}; the request is sent unchanged."
        )
        return messages
    logger.warning(
        f"Context guard: measured request size {measured_total} tokens (provenance={measured_provenance.value}) "
        f"plus the {reservation}-token output reservation exceeds the {window}-token window of "
        f"{model or '<default>'}; the oversized request is not sent, the last-resort shrink runs first."
    )
    shrunk, fits = last_resort_fit(
        messages,
        budget=window - reservation,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(messages),
    )
    if not fits:
        counters.increment("context_guard_truncated_still_over")
        logger.error(
            f"Context guard: the last-resort shrink could not bring the request under the "
            f"{window - reservation}-token input space; the best-effort shrunk list is sent and the "
            "provider's verdict remains the ground truth."
        )
    else:
        counters.increment("context_guard_truncated")
    return shrunk


async def enforce_in_loop_budget(
    provider: "Provider",
    model: str | None,
    storage_id: int,
    thread_id: int,
    conversation_messages: list[Message],
) -> list[bool]:
    """Check the running budget before an in-loop (tool-loop) provider request.

    The pre-call guard (:func:`_enforce_context_budget`) covers the request
    that starts a turn. While the model calls tools, every iteration of the
    provider tool loop grows the message list with the executed tool traffic,
    and a turn that started comfortably inside the budget can exceed it by a
    later in-loop request. Every provider tool loop calls this helper
    immediately before its recursive in-loop provider request, over the
    ``conversation_messages`` mirror of its request list, and applies the
    returned keep-mask to BOTH lists: the provider-format request list and
    the mirror itself. The filtered mirror is what the loop passes into the
    recursion, so mirror and request list stay in lockstep, same length,
    same order, on every later check of the turn, and each mask stays
    index-aligned with the list that is actually sent even after several
    drops in one turn.

    The running size comes only from the single measurement entry point
    (:func:`chibi.services.context_measurement.measure_request_size`), re-read
    over the WHOLE current list on every in-loop request, the durable
    provider-measured base plus the counted appends. The choice is deliberate:
    an incrementally maintained figure would drift from the list it describes
    (every drop, every concurrent history mutation and every re-anchoring
    provider report would have to be replayed into it), while re-measuring
    costs one storage read plus a tokenizer count per in-loop request, cheap
    next to the provider call it guards. When an in-loop response reports
    usage, the durable record is re-anchored with the message count of the
    list just sent, so the next re-measurement resumes from the provider's own
    figure; after a drop the list is shorter than the last record's
    ``message_count`` and the measurement degrades to the full tokenizer
    estimate, which UNDER-reports a real request, the accepted direction of
    the measurement work item. Because the loop filters the mirror with the
    same mask it applies to the request list, the mirror it measures IS the
    list about to be sent: no drop can leave stale, already-dropped traffic
    in the measured list.

    The budget arithmetic is the pre-call guard's: the model window minus the
    single shared output reservation (:func:`_summarizer_max_tokens`). Where
    the window does not resolve, nothing is measured and nothing is dropped,
    the reactive backstop (:func:`_reactive_context_recovery`) remains the
    backstop for an oversized request, exactly as at the pre-call site.

    When the running size exceeds the budget, the oldest complete units are
    dropped through the shared last-resort path (:func:`last_resort_fit`) in
    its drop-only, protected-first mode: the pair-integrity rule, the drop
    ordering, the overshoot policy and the accounting are the shared
    implementation, not a divergent copy. Content truncation is deliberately
    not used in-loop: it shrinks ``Message.content`` in place, while the
    request lists of the provider loops hold independently serialized copies
    of the same messages, so a truncation could not reach the actual request,
    a shrink that claims to reduce a request that still carries the full
    content would be a lie. The drop-only mode still reduces the list to the
    protected units (the unguarded whole-unit sweep), and when even the
    protected head and tail units alone exceed the input space, the
    best-effort list is sent and the provider's verdict, with the reactive
    recovery behind it, decides, exactly as at the pre-call site.

    Two units are always protected: the unit holding the LAST message (the
    in-flight exchange the model is waiting on) and the unit holding the FIRST
    message (losing the head would trade a budget problem for a
    protocol-invalid request, e.g. Anthropic requires a user-role first
    message). Everything between them is droppable, oldest first.

    Args:
        provider: The provider instance the in-loop request will use.
        model: The concrete model identifier the in-loop request will use.
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        thread_id: The message thread identifier (0 for the main thread).
        conversation_messages: The Message-object mirror of the provider
            loop's request list (the same messages, appended in lockstep with
            the provider-format list, on the same measured base). When the
            mask drops messages, the loop filters this list with the same
            mask it applies to the request list and passes the filtered
            mirror into the recursion, so the lockstep invariant is
            maintained for every later check of the turn. The caller's own
            list object is never mutated.

    Returns:
        A keep-mask aligned index-for-index with ``conversation_messages``:
        True keeps the message at that index, False drops it. The provider
        loop applies the same mask to its own provider-format request list
        AND to the mirror itself, and passes the filtered mirror down, so
        the alignment holds for the second and every later drop of a turn,
        not only for the first one.
    """
    window = get_model_context_window(model)
    counters = ContextDecisionCounters()
    if window is None:
        counters.increment("in_loop_guard_unknown_window")
        logger.debug(
            f"In-loop context guard: no context window known for model {model or '<default>'}; "
            "the in-loop request is sent unmeasured and the reactive recovery remains the backstop."
        )
        return [True] * len(conversation_messages)
    reservation = _summarizer_max_tokens(provider, model)
    try:
        measured_total, measured_provenance = await measure_request_size_with_provenance(
            user_id=storage_id,
            thread_id=thread_id,
            model=model or "",
            messages=conversation_messages,
        )
    except Exception as e:
        counters.increment("in_loop_guard_measurement_failed")
        # The measurement entry point degrades to the tokenizer estimate on any
        # storage problem it can see; a failure it cannot see (the database
        # itself being unresolvable) must not break the in-flight turn either.
        # The request is sent unmeasured and the reactive recovery remains the
        # backstop, exactly as on the unknown-window path.
        logger.warning(
            f"In-loop context guard: the request size for user {storage_id} thread {thread_id} could not "
            f"be measured ({e}); the in-loop request is sent unmeasured and the reactive recovery remains "
            "the backstop for an oversized request."
        )
        return [True] * len(conversation_messages)
    if measured_total + reservation <= window:
        counters.increment("in_loop_guard_noop")
        logger.debug(
            f"In-loop context guard: measured request size {measured_total} tokens "
            f"(provenance={measured_provenance.value}) plus the {reservation}-token output reservation "
            f"fits the {window}-token window of {model or '<default>'}; the in-loop request is sent "
            "unchanged."
        )
        return [True] * len(conversation_messages)
    logger.warning(
        f"In-loop context guard: measured request size {measured_total} tokens "
        f"(provenance={measured_provenance.value}) plus the {reservation}-token output reservation "
        f"exceeds the {window}-token window of {model or '<default>'}; the oldest complete units are "
        "dropped before the next in-loop request."
    )
    shrunk, fits = last_resort_fit(
        conversation_messages,
        budget=window - reservation,
        measured_total=measured_total,
        measured_request_counted_tokens=estimate_messages_tokens(conversation_messages),
        truncate=False,
        protected_first=True,
    )
    if not fits:
        counters.increment("in_loop_guard_truncated_still_over")
        logger.error(
            f"In-loop context guard: even dropping every droppable unit cannot bring the request under the "
            f"{window - reservation}-token input space of {model or '<default>'}; the best-effort list is "
            "sent and the provider's verdict remains the ground truth."
        )
    else:
        counters.increment("in_loop_guard_truncated")
    kept_ids = {id(msg) for msg in shrunk}
    return [id(msg) in kept_ids for msg in conversation_messages]


@inject_database
async def emergency_summarization(db: Database, storage_id: int, thread_id: int) -> CompactionResult | None:
    """Collapse the thread history into a single summary turn.

    The compaction state machine runs its phases in one ordered code path:

    1. **measure**: the request size comes from the single measurement
       entry point (:func:`chibi.services.context_measurement.measure_request_size`),
       a durable provider-measured base plus a counted appended part, with
       the documented degradation to the tokenizer estimate when no valid
       record exists.
    2. **compact**: admission control for the summarizer call itself: the
       fit budget is the space the summarizer call actually has (the
       context window of the active model minus the output reservation,
       with the historical share of ``max_history_tokens`` only as the
       fallback for an unresolvable window), the two-tier fit trims the
       input under that budget, and the summarizer call reserves exactly
       what the failed request reserved (:func:`_summarizer_max_tokens`,
       derived through the same provider path the turn used), never more
       output than the request that just failed. Full collapse is the
       intended normal case: with a resolvable large-window model the whole
       history reaches the summarizer intact.
    3. **replace**: the whole thread history is replaced with the marked
       seed through the single replacement helper
       (:func:`replace_thread_history`), which also invalidates the usage
       cache.
    4. **verify**: the compaction outcome is decided with the exact
       predicate ``prompt_tokens + requested_max_tokens <= budget``
       (:func:`verify_compaction`). On this path the post-compaction figure
       is, by construction, a PREDICTION (the replacement drops the durable
       provider record, and the provider has not seen the compacted history
       yet), so the best outcome this check can record is
       ``PREDICTED_FITS``, an advisory result that never claims
       verification; the authoritative provider-side confirmation is the
       same check the budget-choke-point recovery consumes with the retry's
       reported usage. A compaction that did not reduce the size is rejected
       as a failure. When the resulting request still does not fit, the
       synchronous last-resort path (:func:`last_resort_fit`) drops the
       oldest complete units and truncates the largest messages so the next
       request can be sent; it is visibly distinct from a successful
       compaction.

    The whole machine runs under the per-thread lock (the same key the
    provider-call path uses), so a compaction can never overlap itself or a
    concurrent chat turn's history mutations within one process.

    Args:
        db: Database instance (injected).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        thread_id: The message thread identifier (0 for the main thread).

    Returns:
        A :class:`CompactionResult` describing how the attempt ended (the
        outcome plus whether the last-resort fallback ran and fitted), or
        ``None`` when no compaction was attempted (empty summarizer input or
        an empty summary, the thread is left untouched in both cases).
    """
    lock = await LockManager().get_lock(key=f"{storage_id}_{thread_id}")
    async with lock:
        user = await db.get_or_create_user(user_id=storage_id)

        chat_history = await db.get_conversation_messages(user=user, thread_id=thread_id)
        # In-loop tool results (shape-(i) pairs) are excluded from the summarizer
        # input exactly as before: assistant tool_calls and their role="tool"
        # results are not useful summarization context and their pairing is
        # provider-protocol sensitive. The pair guard of the fit's second tier is
        # deliberately NOT this filter: it lives inside the fit, operates on
        # complete units, and stays applicable to lists that do contain pairs.
        # Hidden messages (private CoT from ack_silently turns) are excluded too
        # (task 5 PM ruling): folding them into a summary would launder private
        # reasoning into durable context. Note: the summary replaces the whole
        # history below, so post-summarization those messages leave the
        # conversation entirely (existing summarization behaviour, not a
        # retention change; the storage rows are untouched.
        summarizable = [
            msg for msg in chat_history if not msg.tool_calls and not msg.tool_call_id and msg.visibility != "hidden"
        ]
        # The summarizer call below uses the thread's active provider and
        # model, so the output reservation is derived from the failed
        # request's own reservation through the same provider path.
        model = user.get_active_llm_model(thread_id=thread_id)
        provider = user.get_active_llm_provider(thread_id=thread_id)
        # Phase 1: measure, real-scale admission control for the summarizer
        # call. The full stored conversation (not the summarizable subset) is
        # passed so the durable record aligns with the message count it was
        # measured at; an empty active model name can never match a record,
        # which is the intended degradation to the tokenizer estimate. The
        # same full conversation is counted for the fit's attribution ratio,
        # keeping the credited per-drop reduction on the measured scale. The
        # shape-(i) tool traffic the measured base contains is not droppable
        # by the fit (it is excluded from the candidates), which makes the fit
        # conservative. The database instance is injected by the decorator, as
        # everywhere else.
        measured_total, measured_provenance = await measure_request_size_with_provenance(
            user_id=storage_id,
            thread_id=thread_id,
            model=model or "",
            messages=chat_history,
        )
        # Phase 2: compact, admission control for the summarizer call: the
        # fit budget, the two-tier fit under it, and the call itself at the
        # output reservation derived from the failed request.
        budget = _summarizer_fit_budget(provider, model)
        logger.info(
            f"Emergency summarization for user {storage_id} thread {thread_id}: measured request size "
            f"{measured_total} tokens (provenance={measured_provenance.value}), summarizer fit budget "
            f"{budget} tokens, {len(summarizable)} summarizable message(s) of {len(chat_history)} stored."
        )
        kept = _fit_messages_to_budget(
            summarizable,
            budget=budget,
            measured_total=measured_total,
            measured_request_counted_tokens=estimate_messages_tokens(chat_history),
        )
        chat_history_string = "\n".join(msg.content for msg in kept)
        if not chat_history_string.strip():
            # Empty-input guard: with nothing to summarise the request would
            # return an empty (or hallucinated) summary, and writing that as the
            # seed would destroy the thread's history for nothing. Skip the
            # compaction and leave the thread untouched, the failure is logged,
            # never silently swallowed, and the pre-call guard / reactive
            # recovery remain the backstops for an oversized request.
            ContextDecisionCounters().increment("compaction_skipped")
            logger.warning(
                f"Emergency summarization skipped for user {storage_id} thread {thread_id}: "
                "the fitted summarizer input is empty, so no seed is written and the "
                "thread is left untouched."
            )
            return None

        user_messages: list[Message] = [Message(role="user", content=chat_history_string)]
        response, _ = await provider.get_chat_response(
            messages=user_messages,
            user=user,
            model=model,
            max_tokens=_summarizer_max_tokens(provider, model),
            system_prompt=(
                "Summarize this conversation, keeping the most important and useful information using English."
            ),
            caller_storage_id=storage_id,
            caller_thread_id=thread_id,
        )
        if not response.answer or not response.answer.strip():
            # Empty-summary guard: an empty or whitespace-only summary must never
            # be written as the seed, the thread would lose its whole history in
            # exchange for nothing. Abort and leave the thread untouched; the
            # failed compaction is logged, not swallowed.
            ContextDecisionCounters().increment("compaction_skipped")
            logger.warning(
                f"Emergency summarization aborted for user {storage_id} thread {thread_id}: "
                "the summarizer returned an empty answer, so the thread is left "
                "untouched instead of being seeded with an empty summary."
            )
            return None
        # Phase 3: replace, the whole history becomes the marked seed through
        # the single replacement helper (which also invalidates the usage
        # cache). The seed is marked so a repeated compaction recognises it;
        # the whole-thread swap means it can never append.
        seed: list[Message] = build_compaction_seed(summary=response.answer)
        await replace_thread_history.__wrapped__(db, storage_id=storage_id, thread_id=thread_id, seed=seed)
        # Phase 4: verify, the outcome is decided with the exact predicate
        # ``prompt_tokens + requested_max_tokens <= budget`` inside
        # verify_compaction. previous_tokens is measured_total, computed by
        # the same measurement entry point over the pre-replacement list;
        # it is provider-anchored (the durable ThreadUsageRecord base)
        # whenever a record valid for this model existed at measure time.
        # post_tokens is a PREDICTION, and this path says so: after a
        # whole-thread replacement the durable base is invalidated by design
        # (reset_chat_history drops the record, user.thread_usage.pop) and
        # the provider has not seen the compacted history yet, so no
        # provider-reported figure exists here and none is fabricated, the
        # post side is always the counted estimate, which UNDER-reports a
        # real request: it omits the whole-request overhead the provider
        # figure includes (system prompt, skills, tool schemas), so the
        # reduction check is NOT conservative in direction, its bias is
        # toward "reduced"/"fits". What carries the verdict is the
        # order-of-magnitude margin between the sides: the pre figure is the
        # oversized history that tripped the trigger, the post side a
        # two-message seed bounded by the summarizer's own output
        # reservation, so no realistic counting error can flip the
        # comparison.
        # The predicted figure is therefore passed with
        # TokenProvenance.PREDICTED, whose best case is PREDICTED_FITS, an
        # advisory outcome that can never be recorded as a verified success.
        # The authoritative provider-side confirmation is the SAME
        # verify_compaction check in its PROVIDER_REPORTED mode, consumed at
        # the next measured turn (the budget-choke-point recovery calls it
        # with the retry's reported usage).
        post_tokens, post_provenance = await measure_request_size_with_provenance(
            user_id=storage_id,
            thread_id=thread_id,
            model=model or "",
            messages=seed,
        )
        reservation = _summarizer_max_tokens(provider, model)
        verification_budget = _verification_budget(model)
        outcome = verify_compaction(
            post_tokens=post_tokens,
            previous_tokens=measured_total,
            requested_max_tokens=reservation,
            budget=verification_budget,
            post_provenance=TokenProvenance.PREDICTED,
        )
        if outcome is CompactionOutcome.REDUCED_AND_FITS:
            # Only reachable from a provider-reported post figure, which this
            # path never has; kept for symmetry with the authoritative mode.
            ContextDecisionCounters().increment("compaction_succeeded")
            logger.info(
                f"Compaction verified for user {storage_id} thread {thread_id}: "
                f"{measured_total} -> {post_tokens} tokens (provider-reported), "
                f"resulting request fits the {verification_budget}-token budget."
            )
            return CompactionResult(outcome=outcome)
        if outcome is CompactionOutcome.PREDICTED_FITS:
            # Advisory: the predicted post-compaction size fits, but this is
            # a prediction, not a verification. No destructive fallback runs
            # on a prediction; the provider's own usage at the next turn is
            # the confirmation (the budget-choke-point recovery re-checks
            # with the reported figure in the authoritative mode).
            ContextDecisionCounters().increment("compaction_succeeded")
            logger.info(
                f"Compaction predicted to fit for user {storage_id} thread {thread_id}: "
                f"{measured_total} -> {post_tokens} tokens (predicted post-compaction size, "
                "not yet provider-verified; the authoritative check runs when the provider "
                f"next reports usage) against the {verification_budget}-token budget."
            )
            return CompactionResult(outcome=outcome)
        # Failure shapes: REDUCED_BUT_OVER_BUDGET or DID_NOT_REDUCE. Both are
        # rejected as failures, a no-op summarizer call is never a success,
        # and the lossy last-resort path runs so the next request can be sent.
        ContextDecisionCounters().increment(
            "compaction_rejected_did_not_reduce"
            if outcome is CompactionOutcome.DID_NOT_REDUCE
            else "compaction_rejected_over_budget"
        )
        logger.warning(
            f"Compaction rejected for user {storage_id} thread {thread_id}: outcome "
            f"{outcome.value} ({measured_total} -> {post_tokens} tokens, post side predicted) "
            f"against the {verification_budget}-token budget."
        )
        # The shrink mutates the seed messages in place (content only), so the
        # change is detected against a fingerprint taken before it ran, not by
        # list identity or equality.
        seed_fingerprint = [(msg.id, msg.content) for msg in seed]
        shrunk, fits = last_resort_fit(
            seed,
            budget=_verification_budget(model) - reservation,
            measured_total=post_tokens,
            measured_request_counted_tokens=estimate_messages_tokens(seed),
        )
        if not fits:
            logger.error(
                f"Last-resort shrink failed to make the request fit for user {storage_id} "
                f"thread {thread_id}; the thread is left in its best-effort state."
            )
        if [(msg.id, msg.content) for msg in shrunk] != seed_fingerprint:
            # The shrink changed something (dropped or truncated): persist it
            # through the single replacement helper, which keeps message ids
            # intact. When nothing changed, the seed is already the best
            # effort and is not rewritten.
            await replace_thread_history.__wrapped__(db, storage_id=storage_id, thread_id=thread_id, seed=shrunk)
        if fits:
            ContextDecisionCounters().increment("compaction_fallback_succeeded")
        else:
            ContextDecisionCounters().increment("compaction_fallback_failed")
        return CompactionResult(outcome=outcome, fallback_ran=True, fallback_fitted=fits)


@inject_database
async def _reactive_context_recovery(
    db: Database,
    storage_id: int,
    thread_id: int,
    user: "User",
    interface: UserInterface,
    provider: "Provider",
    model: str | None,
    new_message_to_llm: Message,
    overflow_error: ContextLengthExceededError,
) -> ChatResponseSchema:
    """Reactive backstop after the provider rejected a request as oversized.

    Lives at the same single call site the budget guard guards
    (``get_llm_chat_completion_answer``) and runs AFTER that function's
    lock block has exited: the compaction machine
    (:func:`emergency_summarization`) re-enters the same per-thread lock,
    and ``asyncio.Lock`` is not re-entrant.

    The recovery is fresh by construction: it re-measures and re-assembles
    the request from the same source of truth the guard used (storage
    history through the single measurement entry point, plus the current
    turn's message, which is persisted only after a successful response)
    instead of reusing the failed request's message list. Neither failure
    of the old decorator-level retry can repeat:

    - the request is never re-sent untrimmed: when the compaction did not
      reduce the measured request size (including the compaction being
      skipped entirely), the recovery fails honestly instead of retrying;
    - the output reservation is never larger than the one the failed
      request carried: it is the single shared expression
      (:func:`_summarizer_max_tokens`) resolved through the same provider
      path the failed request resolved its ``max_tokens`` with.

    A reduced request that still does not fit a resolvable window is
    force-fitted by the transient last-resort path before the single retry
    call; when the window does not resolve, the reduced request is sent as
    the backstop's last move and the provider's verdict decides. The retry
    is a single direct provider call: overflow is deliberately excluded
    from failover (the error-normalization work item guarantees the error
    type and the retry predicate), and one retry is the whole contract of
    this backstop. When the retry itself overflows, the original error is
    re-raised and the presentation layer (``handle_gpt_exceptions``)
    reports it honestly.

    After a successful retry this is the only place in the system where the
    authoritative post-compaction verification is possible: the provider
    reports the ``prompt_tokens`` of the compacted request, and
    :func:`verify_compaction` consumes it in its
    :attr:`TokenProvenance.PROVIDER_REPORTED` mode against the measured
    pre-compaction size. When the provider reports no usage on that call,
    the verification is skipped (logged) and the predicted advisory outcome
    of the compaction stands unconfirmed.

    Args:
        db: Database instance (injected).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        thread_id: The message thread identifier (0 for the main thread).
        user: The requesting user.
        interface: Interface of the current turn.
        provider: The provider instance the failed request used.
        model: The model identifier the failed request used (None = provider
            default).
        new_message_to_llm: The current turn's message (not yet persisted).
        overflow_error: The original overflow error, re-raised when the
            recovery cannot produce a reduced request.

    Returns:
        The retried turn's response, with the compression note appended.

    Raises:
        ContextLengthExceededError: When reactive recovery is disabled, the
            compaction did not reduce the request, or the force-fitted
            request still does not fit a resolvable window.
    """
    if not gpt_settings.reactive_context_recovery:
        ContextDecisionCounters().increment("recovery_disabled")
        logger.warning(
            f"Reactive context recovery is disabled; the overflow error for user {storage_id} "
            f"thread {thread_id} is presented as-is."
        )
        raise overflow_error

    lock = await LockManager().get_lock(key=f"{storage_id}_{thread_id}")
    history = await db.get_conversation_messages(user=user, thread_id=thread_id)
    failed_request_messages = [*history, new_message_to_llm]
    previous_tokens, previous_provenance = await measure_request_size_with_provenance(
        user_id=storage_id,
        thread_id=thread_id,
        model=model or "",
        messages=failed_request_messages,
    )
    reservation = _summarizer_max_tokens(provider, model)
    ContextDecisionCounters().increment("recovery_overflow_caught")
    logger.info(
        f"Reactive recovery for user {storage_id} thread {thread_id}: ContextLengthExceededError caught; "
        f"the failed request was re-measured fresh from storage at {previous_tokens} tokens "
        f"(provenance={previous_provenance.value}), no figure reused from the failed request."
    )

    compaction_result = await emergency_summarization(storage_id=storage_id, thread_id=thread_id)
    if compaction_result is None:
        logger.info(
            f"Reactive recovery for user {storage_id} thread {thread_id}: no compaction was attempted "
            "(the compaction machine skipped the thread)."
        )
    else:
        logger.info(
            f"Reactive recovery for user {storage_id} thread {thread_id}: compaction outcome "
            f"{compaction_result.outcome.value}, fallback_ran={compaction_result.fallback_ran}, "
            f"fallback_fitted={compaction_result.fallback_fitted}."
        )

    async with lock:
        retry_messages = await db.get_conversation_messages(user=user, thread_id=thread_id)
        retry_messages.append(new_message_to_llm)
        post_tokens, post_provenance = await measure_request_size_with_provenance(
            user_id=storage_id,
            thread_id=thread_id,
            model=model or "",
            messages=retry_messages,
        )
        outcome = verify_compaction(
            post_tokens=post_tokens,
            previous_tokens=previous_tokens,
            requested_max_tokens=reservation,
            budget=_verification_budget(model),
            post_provenance=TokenProvenance.PREDICTED,
        )
        if outcome is CompactionOutcome.DID_NOT_REDUCE:
            ContextDecisionCounters().increment("recovery_failed_not_reduced")
            logger.error(
                f"Reactive recovery failed for user {storage_id} thread {thread_id}: the request was not "
                f"reduced ({previous_tokens} -> {post_tokens} tokens); the original overflow error is "
                "presented instead of re-sending an untrimmed request."
            )
            raise overflow_error
        if outcome is CompactionOutcome.REDUCED_BUT_OVER_BUDGET:
            window = get_model_context_window(model)
            if window is None:
                logger.warning(
                    f"Reactive recovery for user {storage_id} thread {thread_id}: the request was reduced "
                    f"({previous_tokens} -> {post_tokens} tokens) but no context window is known for "
                    f"{model or '<default>'}; the reduced request is sent as the backstop's last move."
                )
            else:
                shrunk, fits = last_resort_fit(
                    retry_messages,
                    budget=window - reservation,
                    measured_total=post_tokens,
                    measured_request_counted_tokens=estimate_messages_tokens(retry_messages),
                )
                if not fits:
                    ContextDecisionCounters().increment("recovery_failed_still_over")
                    logger.error(
                        f"Reactive recovery failed for user {storage_id} thread {thread_id}: even the "
                        f"last-resort shrink cannot bring the request under the {window - reservation}-token "
                        "input space; the original overflow error is presented."
                    )
                    raise overflow_error
                retry_messages = shrunk
        ContextDecisionCounters().increment("recovery_retry_sent")
        logger.info(
            f"Reactive recovery for user {storage_id} thread {thread_id}: retrying once with the re-assembled "
            f"request ({previous_tokens} -> {post_tokens} tokens, post side {post_provenance.value})."
        )
        response, new_messages = await provider.get_chat_response(
            messages=retry_messages,
            user=user,
            model=model,
            interface=interface,
            track_prompt_size=True,
            caller_storage_id=storage_id,
            caller_thread_id=thread_id,
        )
        # Authoritative post-compaction verification: the provider has now
        # seen the compacted history and reported the prompt_tokens of the
        # request it served. This is the only authoritative verification
        # point in the system.
        reported_tokens = 0
        if isinstance(response.usage, UsageSchema):
            reported_tokens = UsageCacheStore._raw_prompt_tokens(usage=response.usage, provider=provider.name)
        if reported_tokens > 0:
            authoritative = verify_compaction(
                post_tokens=reported_tokens,
                previous_tokens=previous_tokens,
                requested_max_tokens=reservation,
                budget=_verification_budget(model),
                post_provenance=TokenProvenance.PROVIDER_REPORTED,
            )
            ContextDecisionCounters().increment(f"verification_authoritative_{authoritative.value}")
            logger.info(
                f"Authoritative compaction verification for user {storage_id} thread {thread_id}: the provider "
                f"reported {reported_tokens} prompt tokens against the {previous_tokens}-token pre-compaction "
                f"request, outcome {authoritative.value}."
            )
        else:
            ContextDecisionCounters().increment("verification_skipped_no_usage")
            logger.warning(
                f"Authoritative compaction verification for user {storage_id} thread {thread_id} skipped: "
                "the provider reported no usage on the retry call; the predicted advisory outcome stands "
                "unconfirmed."
            )
        await db.add_message(user=user, message=new_message_to_llm, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
        for message in new_messages:
            await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
        answer = response.answer + "\n\n_(Context was compressed to stay within limits.)_"
        return response.model_copy(update={"answer": answer})


@inject_database
async def save_telegram_document_metadata(db: Database, storage_id: int, file_metadata: dict[str, Any]) -> str:
    user = await db.get_or_create_user(user_id=storage_id)
    file_meta = TelegramFileMeta(**file_metadata)
    files_update = {
        file_meta.file_unique_id: file_meta,
    }
    if user.telegram_files:
        user.telegram_files.update(files_update)
    else:
        user.telegram_files = files_update

    await db.save_user(user)
    return file_meta.file_unique_id


@inject_database
async def get_telegram_documents(db: Database, storage_id: int, limit: int = 0) -> dict[str, TelegramFileMeta]:
    user = await db.get_or_create_user(user_id=storage_id)
    files = user.telegram_files
    if limit == 0:
        return files

    last_files = dict(islice(files.items(), max(len(files) - limit, 0), None))
    return last_files


@inject_database
async def get_telegram_document(db: Database, storage_id: int, file_unique_id: str) -> TelegramFileMeta | None:
    user = await db.get_or_create_user(user_id=storage_id)
    return user.telegram_files.get(file_unique_id)


def _log_failed_notification(task: "asyncio.Task[None]") -> None:
    """Done-callback for fire-and-forget failover notification tasks.

    Logs (never raises) when a warning message could not be delivered to
    the chat — a failed notification must never surface as an error to the
    user or break the answer path.

    Args:
        task: The completed notification task.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(f"Failover: failed to deliver the chat notification: {exc}")


def _deliver_failover_notification(interface: UserInterface, pending: list[asyncio.Task[None]], message: str) -> None:
    """Fire-and-forget delivery of a failover notification to the chat.

    A slow or failed chat send must never delay or break the failover
    walk itself; the created task is appended to ``pending`` so the caller
    can await it before returning (warning lands before the answer/error),
    and a failed delivery is logged, never raised.

    Args:
        interface: The user interface to deliver the message through.
        pending: Mutable list of in-flight notification tasks to append to.
        message: The notification message to deliver.
    """
    task = asyncio.create_task(interface.send_message(message=message))
    pending.append(task)
    task.add_done_callback(_log_failed_notification)


def _failover_fallback_hook(
    interface: UserInterface, pending: list[asyncio.Task[None]], trigger: FailoverTrigger, pair: FailoverPair
) -> None:
    """Deliver a fallback warning to the chat (fire-and-forget).

    Args:
        interface: The user interface to deliver the message through.
        pending: Mutable list of in-flight notification tasks to append to.
        trigger: The trigger that fired the fallback.
        pair: The (provider, model) pair about to be attempted.
    """
    _deliver_failover_notification(interface, pending, failover_warning_message(trigger=trigger, fallback=pair))


def _failover_failure_hook(
    interface: UserInterface,
    pending: list[asyncio.Task[None]],
    trigger: FailoverTrigger,
    fallback_targets: list[FailoverPair],
) -> None:
    """Deliver the final honest-failure message to the chat (fire-and-forget).

    Args:
        interface: The user interface to deliver the message through.
        pending: Mutable list of in-flight notification tasks to append to.
        trigger: The last trigger that fired.
        fallback_targets: The fallback pairs that were actually attempted.
    """
    _deliver_failover_notification(
        interface, pending, failover_failure_message(trigger=trigger, fallback_targets=fallback_targets)
    )


@inject_database
async def get_llm_chat_completion_answer(
    db: Database,
    storage_id: int,
    interface: UserInterface,
    user_text_message: str | None = None,
    user_voice_message: BytesIO | None = None,
    user_caption: str | None = None,
    tool_message: Optional["ToolResponseSchema"] = None,
    scheduler_trigger_text: str | None = None,
    scheduler_job_id: str | None = None,
) -> ChatResponseSchema:
    """Produce an LLM answer for any incoming turn of the given thread.

    Builds the JSON prompt for the turn according to its kind — user message,
    tool response, file caption, or scheduler trigger (the third system
    message type) — appends it to the thread history, and asks the active
    provider for a completion. All new messages (prompt and answer) are
    persisted to the thread history.

    The provider call is the single shared call site of the context budget:
    under the per-thread lock and immediately before it, the pre-call guard
    (:func:`_enforce_context_budget`) refuses to send an oversized request
    and truncates transiently where the model window resolves. When the
    provider still rejects the request as oversized, the reactive backstop
    (:func:`_reactive_context_recovery`) runs at this same call site after
    the lock block has exited; every other provider error propagates to the
    presentation layer unchanged.

    Args:
        db: Database instance (injected).
        storage_id: The storage key of the conversation (user_id for private
            chats, chat_id for groups).
        interface: Interface of the current turn, used for the provider call
            context (storage/thread ids, editor context).
        user_text_message: Raw text typed by the user, if any.
        user_voice_message: Voice message audio, if any.
        user_caption: Caption of an uploaded file, if any.
        tool_message: Result of a background tool execution, if any.
        scheduler_trigger_text: Trigger text from a fired scheduler job
            (action type ``self``), if any.
        scheduler_job_id: Identifier of the scheduler job that fired; only
            meaningful together with ``scheduler_trigger_text``.

    Returns:
        The provider's answer for this turn.

    Raises:
        ValueError: If no prompt data of any kind is provided, or a voice
            message is given while the user has no STT provider.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    thread_id = interface.thread_id
    lock = await LockManager().get_lock(key=f"{storage_id}_{thread_id}")

    if (
        not user_text_message
        and not user_voice_message
        and not tool_message
        and not user_caption
        and not scheduler_trigger_text
    ):
        raise ValueError("No prompt data provided")

    if user_voice_message and not user.stt_provider:
        raise ValueError("Can't compute voice message: no STT provide available.")

    prompt: dict[str, Any]
    editor_ctx = interface.editor_context if isinstance(interface, EditorContextProvider) else None

    if scheduler_trigger_text:
        prompt = {
            "type": "scheduled_trigger",
            "desc": (
                "Service trigger from the Chibi scheduler. This is NOT a user message and not a tool "
                "response: the agent was woken up by a scheduled job in this thread. Handle the "
                "trigger_text below using the thread context. If there is nothing worth reporting to "
                "the user, end the turn with ack_silently(); otherwise deliver via say()."
            ),
            "job_id": scheduler_job_id,
            "trigger_text": scheduler_trigger_text,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    elif tool_message:
        prompt = {
            "type": "tool response",
            "desc": "background task is done",
            "tool_name": tool_message.tool_name,
            "tool_response": tool_message.model_dump(),
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    elif user_caption:
        prompt = {
            "type": "system message",
            "desc": "user uploaded a file",
            "caption": user_caption,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
        }
    else:
        user_message = (
            await user.stt_provider.transcribe(audio=user_voice_message) if user_voice_message else user_text_message
        )
        assert user_message
        prompt = {
            "user": interface.user_data,
            "prompt": user_message,
            "datetime_now": datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z%z"),
            "type": "user message",
            "transcribed_from_voice_message": bool(user_voice_message),
        }

    if editor_ctx:
        selection = editor_ctx.get("selection")
        editor_context: dict[str, Any] = {
            "active_file": editor_ctx.get("active_file"),
            "language_id": editor_ctx.get("language_id"),
            "cursor_position": editor_ctx.get("cursor_position"),
        }
        workspace_root = editor_ctx.get("workspace_root")
        if isinstance(workspace_root, str):
            editor_context["workspace_root"] = workspace_root.rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if isinstance(selection, dict) and isinstance(selection.get("text"), str) and selection["text"]:
            editor_context["selection"] = {
                "start_line": selection.get("start_line"),
                "end_line": selection.get("end_line"),
                "text": selection["text"][:4096],
            }
        prompt["editor_context"] = editor_context

    async with lock:
        conversation_messages: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)
        new_message_to_llm = Message(role="user", content=json.dumps(prompt))
        conversation_messages.append(new_message_to_llm)

        active_provider = user.get_active_llm_provider(thread_id=thread_id)
        active_model = user.get_active_llm_model(thread_id=thread_id)

        # Pre-call budget guard (the single enforcement point of the context
        # budget): under this thread lock, after the message list is fully
        # assembled and immediately before the provider call. Where the model
        # window resolves, an oversized request is refused and truncated
        # transiently; where it does not, the guard falls through and the
        # reactive recovery below carries the case.
        conversation_messages = await _enforce_context_budget(
            provider=active_provider,
            model=active_model,
            storage_id=storage_id,
            thread_id=thread_id,
            messages=conversation_messages,
        )

        # Failover-guarded chat call. The retry loop lives inside the lock
        # for history consistency; the emergency summarization path above is
        # deliberately NOT failover-guarded (determinism where determinism
        # matters). ``emergency_summarization`` and moderation are out of
        # scope per the failover plan.
        #
        # Notification wiring (failover_notification task): fallback warnings
        # and the final honest failure go to the chat via the interface.
        # Delivery is fire-and-forget so a slow chat send can never delay
        # the failover walk itself; pending sends are awaited before this
        # function returns/raises (warning lands before the answer/error).
        pending_notifications: list[asyncio.Task[None]] = []
        overflow_error: ContextLengthExceededError | None = None
        try:
            failover_result = await run_with_failover(
                role="master",
                user=user,
                primary_provider=active_provider,
                model=active_model,
                call=lambda provider, model: provider.get_chat_response(
                    messages=conversation_messages,
                    user=user,
                    model=model,
                    interface=interface,
                    track_prompt_size=True,
                    caller_storage_id=interface.storage_id,
                    caller_thread_id=interface.thread_id,
                ),
                on_fallback=partial(_failover_fallback_hook, interface, pending_notifications),
                on_failure=partial(_failover_failure_hook, interface, pending_notifications),
            )
        except ContextLengthExceededError as exc:
            # The reactive backstop catches the overflow at this single call
            # site; every other error keeps propagating to the presentation
            # layer unchanged. Overflow is never retried by the tenacity
            # predicate and never failed over (the error-normalization work
            # item guarantees both).
            overflow_error = exc
        finally:
            if pending_notifications:
                await asyncio.gather(*pending_notifications, return_exceptions=True)

        if overflow_error is None:
            chat_response, new_messages = failover_result.response, failover_result.new_messages
            assert chat_response is not None  # Type narrowing for mypy: a returned walk always carries a response
            if not is_pure_ack(chat_response.answer):
                chat_response.answer = strip_ack_sentinel(chat_response.answer)
                for message in new_messages:
                    if message.role == "assistant":
                        message.content = strip_ack_sentinel(message.content)
            await db.add_message(
                user=user, message=new_message_to_llm, ttl=gpt_settings.messages_ttl, thread_id=thread_id
            )
            for message in new_messages:
                await db.add_message(user=user, message=message, ttl=gpt_settings.messages_ttl, thread_id=thread_id)
            return chat_response

    # The lock block above has exited before the recovery runs: the
    # compaction machine re-enters the same per-thread lock (asyncio.Lock is
    # not re-entrant). The recovery owns the single retry call and the
    # turn's persistence; on failure it re-raises the original overflow
    # error for the presentation layer (handle_gpt_exceptions stays a pure
    # error presenter).
    assert overflow_error is not None
    return await _reactive_context_recovery.__wrapped__(
        db,
        storage_id=storage_id,
        thread_id=thread_id,
        user=user,
        interface=interface,
        provider=active_provider,
        model=active_model,
        new_message_to_llm=new_message_to_llm,
        overflow_error=overflow_error,
    )


@inject_database
async def generate_image(
    db: Database,
    interface: UserInterface,
    prompt: str,
    model: str | None = None,
    provider_name: str | None = None,
    images: list[tuple[bytes, str]] | None = None,
) -> list[str] | list[BytesIO]:
    """Generate images via the user's selected or active image provider.

    Args:
        db: Database instance (injected).
        interface: User interface bound to the current request.
        prompt: Image generation prompt.
        model: Optional image model override. Ignored when ``provider_name`` is
            not given, in which case the user's active image model is used.
        provider_name: Optional provider override. When omitted, the user's
            active image provider is used together with its active model.
        images: Optional reference images as (bytes, mime type) tuples for
            models with image-to-image support.

    Returns:
        List of image URLs, or list of BytesIO objects with image data,
        as produced by the provider.

    Raises:
        NoProviderSelectedError: If no image provider is available.
    """
    user = await db.get_or_create_user(user_id=interface.user_id)

    if provider_name:
        provider = user.providers.get(provider_name)
        selected_model = model
    else:
        provider = user.get_active_image_provider(thread_id=interface.thread_id)
        selected_model = user.get_active_image_model(thread_id=interface.thread_id)

    if not provider:
        raise NoProviderSelectedError("No image provider available")
    result = await provider.get_images(prompt=prompt, model=selected_model, images=images)
    if interface.user_id not in gpt_settings.image_generations_whitelist:
        await db.count_image(interface.user_id)
    return result


@inject_database
async def generate_video(
    db: Database,
    interface: UserInterface,
    prompt: str,
    model: str | None = None,
    provider_name: str | None = None,
    duration: int | None = None,
) -> VideoResult:
    """Generate a video via the selected provider and count the usage.

    Args:
        db: Database instance.
        interface: User interface carrying the requester's identity.
        prompt: Video generation prompt.
        model: Optional video model name override.
        provider_name: Optional provider name override.
        duration: Optional clip length in seconds, if the model supports it.

    Returns:
        The fully downloaded video generation result.

    Raises:
        NoProviderSelectedError: If no video provider can be resolved.
    """
    user = await db.get_or_create_user(user_id=interface.user_id)

    if provider_name:
        provider = user.providers.get(provider_name)
        selected_model = model
    else:
        provider = user.get_active_video_provider(thread_id=interface.thread_id)
        selected_model = user.get_active_video_model(thread_id=interface.thread_id)

    if not provider:
        raise NoProviderSelectedError("No video provider available")
    result = await provider.get_videos(prompt=prompt, model=selected_model, duration=duration)
    if interface.user_id not in gpt_settings.video_generations_whitelist:
        await db.count_video(interface.user_id)
    return result


@inject_database
async def describe_image(
    db: Database,
    user_id: int,
    image: bytes,
    mime_type: str,
    model: str | None = None,
    prompt: str | None = None,
) -> VisionResultSchema:
    """Analyze an image using the user's vision provider.

    Args:
        db: Database instance.
        user_id: User ID.
        image: Image data as bytes.
        mime_type: MIME type of the image.
        model: Optional vision model override.
        prompt: Optional prompt to guide the vision analysis.

    Returns:
        Vision analysis result.
    """
    user = await db.get_or_create_user(user_id=user_id)
    provider = user.vision_provider

    return await provider.vision(image=image, model=model, mime_type=mime_type, prompt=prompt)


@inject_database
async def ocr_pdf(
    db: Database,
    user_id: int,
    pdf: bytes,
    model: str | None = None,
) -> VisionResultSchema:
    """Extract text from a PDF using the user's OCR-capable provider.

    Args:
        db: Database instance.
        user_id: User ID.
        pdf: PDF data as bytes.
        model: Optional model override.

    Returns:
        OCR result as VisionResultSchema.

    Raises:
        ValueError: If no OCR-capable provider is available.
    """
    user = await db.get_or_create_user(user_id=user_id)
    provider = user.ocr_provider
    return await provider.ocr(pdf=pdf, model=model)


# @cached(ttl=3600)
@inject_database
async def get_user_cached_models(
    db: Database, user_id: int, image_generation: bool = False, video_generation: bool = False
) -> list[ModelChangeSchema]:
    user = await db.get_or_create_user(user_id=user_id)
    return await user.get_available_models(image_generation=image_generation, video_generation=video_generation)


@inject_database
async def get_models_available(
    db: Database, user_id: int, image_generation: bool = False, video_generation: bool = False, thread_id: int = 0
) -> list[ModelChangeSchema]:
    user = await db.get_or_create_user(user_id=user_id)
    user_models = await get_user_cached_models(
        user_id=user_id, image_generation=image_generation, video_generation=video_generation
    )

    if not user_models:
        return []

    available_models = deepcopy(user_models)

    if video_generation:
        active_provider = user.get_active_video_provider(thread_id=thread_id)
        active_model = user.get_active_video_model(thread_id=thread_id) or active_provider.default_video_model
    elif image_generation:
        active_provider = user.get_active_image_provider(thread_id=thread_id)
        active_model = user.get_active_image_model(thread_id=thread_id) or active_provider.default_image_model
    else:
        active_provider = user.get_active_llm_provider(thread_id=thread_id)
        active_model = user.get_active_llm_model(thread_id=thread_id) or active_provider.default_model

    for model in available_models:
        if model.name == active_model and model.provider == active_provider.name:
            model.display_name = f"🟢 {model.display_name}️"
    return available_models


@inject_database
async def user_has_reached_images_generation_limit(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    return user.has_reached_image_limits


@inject_database
async def user_has_reached_videos_generation_limit(db: Database, user_id: int) -> bool:
    user = await db.get_or_create_user(user_id=user_id)
    return user.has_reached_video_limits


@inject_database
async def set_api_key(db: Database, user_id: int, api_key: str, provider_name: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.tokens[provider_name] = api_key
    await db.save_user(user)
    return None


@inject_database
async def get_info(db: Database, user_id: int) -> str:
    user = await db.get_or_create_user(user_id=user_id)
    return user.info


@inject_database
async def set_info(db: Database, user_id: int, new_info: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.info = new_info
    await db.save_user(user)


@inject_database
async def set_thread_notes(db: Database, user_id: int, thread_id: int, notes: str) -> None:
    """Replace the thread-scoped notes for a specific thread.

    The notes text is stored per thread on the user object and persisted with
    the whole-user save. Full-replace semantics: the stored text is swapped
    entirely, there is no merging. An empty string is allowed and clears the
    notes for the thread. Users loaded from older persisted records are
    handled naturally by the pydantic default of the dict field.

    Args:
        db: The database instance.
        user_id: The storage ID of the user.
        thread_id: The ID of the thread to set the notes for.
        notes: The complete new notes text; an empty string clears the notes.
    """
    user = await db.get_or_create_user(user_id=user_id)
    user.thread_notes[thread_id] = notes
    await db.save_user(user)


@inject_database
async def activate_llm_skill(db: Database, user_id: int, skill_name: str, skill_payload: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.llm_skills[skill_name] = skill_payload
    await db.save_user(user)


@inject_database
async def deactivate_llm_skill(db: Database, user_id: int, skill_name: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    if skill_name not in user.llm_skills.keys():
        raise ValueError(f"The skill {skill_name} seems never been activated")
    user.llm_skills.pop(skill_name)
    await db.save_user(user)


@inject_database
async def set_working_dir(db: Database, user_id: int, new_wd: str) -> None:
    user = await db.get_or_create_user(user_id=user_id)
    user.working_dir = new_wd
    await db.save_user(user)


@inject_database
async def get_cwd(db: Database, user_id: int) -> str:
    user = await db.get_or_create_user(user_id=user_id)
    return user.working_dir


@inject_database
async def set_thread_working_dir(db: Database, user_id: int, thread_id: int, new_wd: str) -> None:
    """Set the working directory override for a specific thread.

    The path is normalized via ``Path(...).expanduser()`` only — no existence
    check is performed. Users loaded from older persisted records are handled
    naturally by the pydantic default of the new dict field.

    Args:
        db: The database instance.
        user_id: The storage ID of the user.
        thread_id: The ID of the thread to set the working directory for.
        new_wd: The new working directory path.
    """
    user = await db.get_or_create_user(user_id=user_id)
    user.thread_working_dirs[thread_id] = str(Path(new_wd).expanduser())
    await db.save_user(user)


@inject_database
async def get_moderation_provider(db: Database, user_id: int) -> "Provider":
    user = await db.get_or_create_user(user_id=user_id)
    return user.moderation_provider


@inject_database
async def drop_tool_call_history(db: Database, storage_id: int, thread_id: int) -> None:
    user = await db.get_or_create_user(user_id=storage_id)
    chat_history: list[Message] = await db.get_conversation_messages(user=user, thread_id=thread_id)
    seed: list[Message] = []
    for message in chat_history:
        if message.role == "tool":
            continue
        message.tool_calls = None
        message.tool_call_id = None
        seed.append(message)
    await replace_thread_history.__wrapped__(db, storage_id=storage_id, thread_id=thread_id, seed=seed)


@inject_database
async def save_thread_name(db: Database, storage_id: int, thread_id: int, name: str) -> None:
    """Save the name of a specific thread for the user.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        thread_id: The ID of the thread.
        name: The name to save for the thread.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    user.thread_names[thread_id] = name
    await db.save_user(user)


@inject_database
async def delete_thread_from_map(db: Database, storage_id: int, thread_id: int) -> None:
    """Delete a thread mapping from the user's saved thread names.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        thread_id: The ID of the thread to delete.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    user.thread_names.pop(thread_id)
    await db.save_user(user)


@inject_database
async def clone_thread_messages(
    db: Database,
    storage_id: int,
    old_thread_id: int,
    new_thread_id: int,
    name: str | None = None,
) -> int:
    """Clone messages and settings from an old thread to a new thread.

    Args:
        db: The database instance.
        storage_id: The storage ID (user_id for private, chat_id for groups).
        old_thread_id: The ID of the thread to clone from.
        new_thread_id: The ID of the thread to clone to.
        name: The name of the new thread. Defaults to None.

    Returns:
        The number of messages that were cloned.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    existing_messages = await db.get_conversation_messages(user=user, thread_id=old_thread_id)

    for i, message in enumerate(existing_messages):
        cloned = deepcopy(message)
        cloned.id = time.time_ns() + i
        await db.add_message(user=user, message=cloned, thread_id=new_thread_id)

    # Re-fetch the user: LocalStorage.add_message persists into a reloaded copy, so
    # the object above is stale and saving it would wipe the just-cloned messages.
    user = await db.get_or_create_user(user_id=storage_id)

    if old_thread_id in user.thread_selected_llm:
        user.thread_selected_llm[new_thread_id] = user.thread_selected_llm[old_thread_id]
    if old_thread_id in user.thread_selected_image_model:
        user.thread_selected_image_model[new_thread_id] = user.thread_selected_image_model[old_thread_id]
    if old_thread_id in user.thread_selected_video_model:
        user.thread_selected_video_model[new_thread_id] = user.thread_selected_video_model[old_thread_id]
    if old_thread_id in user.thread_working_dirs:
        user.thread_working_dirs[new_thread_id] = user.thread_working_dirs[old_thread_id]

    user.thread_names[new_thread_id] = name or str(new_thread_id)

    await db.save_user(user)
    return len(existing_messages)
