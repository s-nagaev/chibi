"""Measured, durable per-thread context size and the single request-size prediction entry point.

The provider-reported ``prompt_tokens`` of the last successful turn is persisted
per ``(user, thread)`` on the user's storage record, together with the model it
was measured on and the number of conversation messages the measured prompt
contained. Everything that needs to know "how big is this request" goes through
:func:`measure_request_size`; nothing else may compute a request size.
"""

import json
from enum import Enum
from typing import Protocol

from loguru import logger

from chibi.models import Message, ThreadUsageRecord
from chibi.schemas.app import UsageSchema
from chibi.services.usage_cache import UsageCacheStore
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database


class Tokenizer(Protocol):
    """A minimal token counter for a model family."""

    def count_tokens(self, text: str) -> int:
        """Count the tokens in a text payload.

        Args:
            text: The text to count tokens for.

        Returns:
            The estimated number of tokens.
        """
        ...


class HeuristicTokenizer:
    """Uncalibrated character-based fallback tokenizer.

    Uses the long-standing ``len(text) // 4`` approximation. It feeds
    everything a real tokenizer would see through one formula (message
    content, roles and serialized tool-call arguments all pass through it via
    :func:`estimate_messages_tokens`), but the ratio itself is NOT measured
    against any reference tokenizer: no tokenizer ships with Chibi, so no
    true reference exists locally. For whole prompts the ``len // 4`` family
    of heuristics is known to UNDER-estimate real provider counts, often
    severalfold (roughly 4.8x smaller for whole prompts): it counts only the
    conversation content and roles, while the provider figure covers the
    entire request (system prompt, skills, tool schemas, tool-call arguments
    and structural overhead). The durable provider measurement, not this
    estimate, is the source of truth, and the estimate is an accepted
    temporary accuracy loss.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens as one quarter of the character count.

        Args:
            text: The text to count tokens for.

        Returns:
            The estimated number of tokens (0 for empty text).
        """
        return len(text) // 4


_HEURISTIC_TOKENIZER = HeuristicTokenizer()


def get_tokenizer(model: str | None = None) -> Tokenizer:
    """Return the tokenizer for the given model family.

    No third-party tokenizer ships with Chibi: adding one would either pull a
    heavy dependency into the Alpine image or download BPE files from the
    network at first use, and the durable measurement (not the tokenizer) is
    the source of truth for request size. Every family therefore gets the
    uncalibrated :class:`HeuristicTokenizer`. The ``Tokenizer`` protocol and the
    ``model`` argument are the insertion point for a per-family implementation
    later: a real tokenizer is slotted in here and an import failure must
    degrade to the heuristic, never raise.

    Args:
        model: The model name whose family the tokenizer is chosen for, or
            None for the family-agnostic default.

    Returns:
        The tokenizer instance for the family.
    """
    return _HEURISTIC_TOKENIZER


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate the token size of conversation messages with the heuristic tokenizer.

    Assistant ``tool_calls`` argument payloads are included explicitly: the
    per-message ``Message.estimate_tokens`` heuristic ignores them entirely,
    and for tool-heavy turns they are the bulk of the volume.

    Args:
        messages: The conversation messages.

    Returns:
        The estimated token count of the whole message list.
    """
    tokenizer = get_tokenizer()
    total = 0
    for msg in messages:
        total += tokenizer.count_tokens(msg.content) + tokenizer.count_tokens(msg.role)
        if msg.tool_calls:
            total += tokenizer.count_tokens(json.dumps([tool_call.model_dump() for tool_call in msg.tool_calls]))
    return total


class SizeProvenance(str, Enum):
    """Whether a request-size figure is anchored by a provider measurement or purely estimated.

    The binary behind the observability contract: a reader of a decision log must never
    have to guess whether a size came from the provider or from a local estimate.

    - :attr:`PROVIDER_MEASURED`: the figure's base is the durable provider-reported
      ``prompt_tokens`` of the thread's last measured turn (``ThreadUsageRecord``), plus a
      local tokenizer count of the messages appended since that measurement. The anchor is
      provider-measured; the appended part is always a local count.
    - :attr:`ESTIMATED`: no valid provider measurement exists for this thread and model
      (absent record, model mismatch, wrong-sized or stale record): the whole list was
      counted with the local heuristic tokenizer. Such a figure UNDER-reports a real
      request and carries no provider authority.
    """

    PROVIDER_MEASURED = "provider_measured"
    ESTIMATED = "estimated"


def predict_request_size_with_provenance(
    record: ThreadUsageRecord | None, model: str, messages: list[Message]
) -> tuple[int, SizeProvenance]:
    """Predict the outgoing request size and state where the figure came from.

    Same arithmetic and the same degradation contract as :func:`predict_request_size`
    (which delegates here and drops the provenance), returning the provenance alongside
    the figure so decision points can log whether the size was provider-measured or
    estimated without re-deriving it.

    Contract (mirrors the task's accessor contract):

    - A missing record is normal, not an error: the whole history is counted
      with the heuristic tokenizer.
    - A record measured on a different model is treated as unknown (a
      different model means a different tokenizer and a different window) and
      degrades to the heuristic estimate.
    - A record with a non-positive ``message_count`` (or a non-positive
      ``prompt_tokens``) is a wrong-sized entry, not a measurement: the
      message count is what tells the reader which part of the history the
      measured prompt already contained, and a count of 0 would make the
      whole history be counted on top of a ``prompt_tokens`` that already
      included it. Such a record degrades to the heuristic estimate.
    - A record whose ``message_count`` exceeds the current history length is
      stale (the history was truncated or replaced without a measurement
      since) and degrades to the heuristic estimate.
    - A degraded prediction can be wrong, but it can never block a request or
      trigger a compaction on its own: it is an accuracy loss only.

    Args:
        record: The last durable measurement for the thread, or None.
        model: The model about to be called.
        messages: The full conversation messages about to be sent.

    Returns:
        A tuple of the predicted request size in tokens and the provenance of
        that figure (see :class:`SizeProvenance`).
    """
    if record is None or record.model != model or record.prompt_tokens <= 0 or record.message_count <= 0:
        return estimate_messages_tokens(messages), SizeProvenance.ESTIMATED
    if len(messages) < record.message_count:
        return estimate_messages_tokens(messages), SizeProvenance.ESTIMATED
    appended = messages[record.message_count :]
    return record.prompt_tokens + estimate_messages_tokens(appended), SizeProvenance.PROVIDER_MEASURED


def predict_request_size(record: ThreadUsageRecord | None, model: str, messages: list[Message]) -> int:
    """Predict the outgoing request size as ``measured + counted(appended)``.

    Provenance-aware wrapper over :func:`predict_request_size_with_provenance`: the
    arithmetic and the degradation contract live there; this function keeps the original
    int-only signature for existing callers and drops the provenance. New decision points
    should call the provenance-returning variant so they can log where the figure came
    from.

    Args:
        record: The last durable measurement for the thread, or None.
        model: The model about to be called.
        messages: The full conversation messages about to be sent.

    Returns:
        The predicted request size in tokens (see
        :func:`predict_request_size_with_provenance` for the degradation contract).
    """
    return predict_request_size_with_provenance(record=record, model=model, messages=messages)[0]


async def _read_measured_record(db: Database, user_id: int, thread_id: int) -> ThreadUsageRecord | None:
    """Read the durable measurement record for a thread, tolerating any absence.

    Args:
        db: The database instance.
        user_id: The storage user identifier.
        thread_id: The message thread identifier (0 for the main thread).

    Returns:
        The stored record, or None when it is absent or storage fails.
    """
    try:
        user = await db.get_or_create_user(user_id=user_id)
        # Tolerate any absence or odd shape (a backend or fixture that does
        # not carry the field at all): degrade to "unknown", never raise.
        return user.thread_usage.get(thread_id)
    except Exception as e:
        logger.warning(f"Failed to read the measured context size for user {user_id}: {e}")
        return None


@inject_database
async def get_last_measured_usage(db: Database, user_id: int, thread_id: int, model: str) -> ThreadUsageRecord | None:
    """Return the last durable measurement for a thread, validated for the model.

    The record carries the measured ``prompt_tokens``, the ``model`` it was
    measured on and the ``message_count`` of the measured prompt, so a reader
    can decide whether it is still valid without reaching into storage. A
    record measured on a different model is reported as unknown (None) for the
    same reason :func:`predict_request_size` degrades on a mismatch.

    Args:
        db: The database instance (injected).
        user_id: The storage user identifier.
        thread_id: The message thread identifier (0 for the main thread).
        model: The model about to be called; a mismatch voids the record.

    Returns:
        The matching record, or None when absent, storage-mismatched or
        measured on a different model.
    """
    record = await _read_measured_record(db=db, user_id=user_id, thread_id=thread_id)
    if record is None or record.model != model:
        return None
    return record


@inject_database
async def measure_request_size(db: Database, user_id: int, thread_id: int, model: str, messages: list[Message]) -> int:
    """Answer "how big is this request" for a thread: the single size entry point.

    Combines the durable per-thread measurement with a tokenizer count of the
    messages appended since it was taken: ``measured + counted(appended)``.
    This is the only function consumers (the summarizer fit budget, the
    pre-call guard) may use to size a request; they must not call
    ``estimate_tokens`` or read the usage stores directly.

    Args:
        db: The database instance (injected).
        user_id: The storage user identifier.
        thread_id: The message thread identifier (0 for the main thread).
        model: The model about to be called.
        messages: The full conversation messages about to be sent.

    Returns:
        The predicted request size in tokens (see
        :func:`predict_request_size` for the degradation contract).
    """
    record = await _read_measured_record(db=db, user_id=user_id, thread_id=thread_id)
    return predict_request_size(record=record, model=model, messages=messages)


@inject_database
async def measure_request_size_with_provenance(
    db: Database, user_id: int, thread_id: int, model: str, messages: list[Message]
) -> tuple[int, SizeProvenance]:
    """Answer "how big is this request" and state whether the figure is measured or estimated.

    The provenance-bearing variant of :func:`measure_request_size` (the single size entry
    point): same arithmetic, same degradation contract, same single-source rule: the
    figure still comes from the durable per-thread provider measurement plus a tokenizer
    count of the appended messages, or degrades to the whole-list heuristic estimate.
    Decision points (the budget guard, the in-loop check, the compaction machine, the
    reactive recovery) call this variant so their decision logs can carry the provenance
    of the observed size instead of leaving a reader to guess.

    Args:
        db: The database instance (injected).
        user_id: The storage user identifier.
        thread_id: The message thread identifier (0 for the main thread).
        model: The model about to be called.
        messages: The full conversation messages about to be sent.

    Returns:
        A tuple of the predicted request size in tokens and the provenance of that
        figure (:attr:`SizeProvenance.PROVIDER_MEASURED` when a valid durable
        provider-reported record anchors it, :attr:`SizeProvenance.ESTIMATED` when the
        whole list was counted with the local heuristic tokenizer).
    """
    record = await _read_measured_record(db=db, user_id=user_id, thread_id=thread_id)
    return predict_request_size_with_provenance(record=record, model=model, messages=messages)


@inject_database
async def persist_measured_usage(
    db: Database,
    user_id: int,
    thread_id: int,
    usage: UsageSchema,
    provider: str,
    model: str,
    message_count: int | None,
) -> None:
    """Persist the last provider-measured prompt size for a thread.

    Called on every successful response whose usage is reported. A response
    without reported usage (raw size 0) never overwrites an existing record:
    a missing measurement must not zero out a good one. The same protection
    applies to an unknown ``message_count`` (``None``): a record without a
    trustworthy count is a wrong-sized entry in the making (the accessor
    cannot tell which part of the history the measured prompt contained), so
    it is never persisted: the thread keeps its previous good record or
    falls back to the estimate. Storage failures are logged and swallowed:
    the durable record is an optimization over the heuristic estimate and
    must never break the chat answer.

    Args:
        db: The database instance (injected).
        user_id: The storage user identifier.
        thread_id: The message thread identifier (0 for the main thread).
        usage: Provider usage schema of the completed response.
        provider: Provider name (Anthropic reports cached input separately).
        model: The model the measurement was taken on.
        message_count: Number of conversation messages the measured prompt
            contained, or None when the count is unknown (no record is then
            written).
    """
    try:
        raw = UsageCacheStore._raw_prompt_tokens(usage=usage, provider=provider)
        if raw == 0:
            return None
        if message_count is None:
            return None
        user = await db.get_or_create_user(user_id=user_id)
        user.thread_usage[thread_id] = ThreadUsageRecord(prompt_tokens=raw, model=model, message_count=message_count)
        await db.save_user(user)
    except Exception as e:
        logger.warning(f"Failed to persist the measured context size for user {user_id} thread {thread_id}: {e}")
