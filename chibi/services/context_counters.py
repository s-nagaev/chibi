"""In-process counters for the context-management decision points.

The counters make the context-management decisions countable, not only visible in log
lines: the canonical question they answer is "why does compaction never fire": a high
``context_guard_noop`` / ``in_loop_guard_noop`` count (every turn was checked and already
fitted) next to zero ``compaction_*`` counts is a different diagnosis from a high
``compaction_skipped`` count (the compaction machine ran but attempted nothing).

Counted decisions and their names:

- ``context_guard_noop`` / ``in_loop_guard_noop``: the pre-call (in-loop) budget guard
  measured the request and it already fitted the window: nothing was done.
- ``context_guard_unknown_window`` / ``in_loop_guard_unknown_window``: the model's
  context window does not resolve; the request is sent unmeasured.
- ``in_loop_guard_measurement_failed``: the in-loop request size could not be measured.
- ``context_guard_truncated`` / ``in_loop_guard_truncated``: the guard refused an
  oversized request and the lossy last-resort shrink produced a fitting list.
- ``context_guard_truncated_still_over`` / ``in_loop_guard_truncated_still_over``: even
  the shrink could not make the request fit; the best-effort list is sent.
- ``compaction_skipped``: the compaction machine ran but attempted no compaction
  (empty summarizer input or an empty summary; the thread is left untouched).
- ``compaction_succeeded``: the compaction was verified (or predicted) to fit and no
  fallback ran.
- ``compaction_rejected_did_not_reduce`` / ``compaction_rejected_over_budget``: the
  verification rejected the compaction outcome.
- ``compaction_fallback_succeeded`` / ``compaction_fallback_failed``: the lossy
  last-resort fallback ran after a rejected compaction and fitted (or did not).
- ``recovery_overflow_caught``: the reactive recovery was entered after the provider
  rejected a request as oversized.
- ``recovery_disabled``: the recovery was entered but is switched off.
- ``recovery_failed_not_reduced`` / ``recovery_failed_still_over``: the recovery could
  not produce a reduced, fitting request.
- ``recovery_retry_sent``: the recovery sent its single retry call.
- ``verification_skipped_no_usage``: the authoritative post-compaction verification was
  skipped because the provider reported no usage.
- ``verification_authoritative_<outcome>``: the authoritative verification recorded the
  given :class:`~chibi.services.user.CompactionOutcome` value.

The store is intentionally ephemeral and process-lifetime (the same contract as
:class:`~chibi.services.usage_cache.UsageCacheStore`): a restart yields zero counters,
and nothing is persisted or exported anywhere.
"""

from chibi.utils.app import SingletonMeta


class ContextDecisionCounters(metaclass=SingletonMeta):
    """Process-lifetime counters of the context-management decisions.

    A plain name -> count mapping over the decision names catalogued in the module
    docstring. Not thread-safe by design: the bot runs as one asyncio process and every
    increment is a single dict update.
    """

    def __init__(self) -> None:
        """Initialize the empty counter store."""
        self._counts: dict[str, int] = {}

    def increment(self, name: str, amount: int = 1) -> int:
        """Count one decision of the given name.

        Args:
            name: The decision name (see the module docstring for the catalogue).
            amount: How many decisions to count; defaults to one.

        Returns:
            The counter's value after the increment.
        """
        self._counts[name] = self._counts.get(name, 0) + amount
        return self._counts[name]

    def snapshot(self) -> dict[str, int]:
        """Return a copy of all counters accumulated so far.

        Returns:
            A mapping of decision name to count; absent names were never counted.
        """
        return dict(self._counts)

    def reset(self) -> None:
        """Clear every counter (used by tests and diagnostics that need a clean slate)."""
        self._counts.clear()
