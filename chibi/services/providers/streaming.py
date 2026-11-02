"""Shared live response-delta streaming helpers for LLM provider adapters.

Live deltas leave the provider layer only through
:class:`AckSentinelStreamSession`, an async context manager that owns the ACK
sentinel gate for exactly one streaming attempt. Emission and lifecycle are the
same object: an adapter that does not open a session has no way to emit a
delta, and one that does cannot leave the gate undrained or shared across
attempts.
"""

from __future__ import annotations

import logging
from types import TracebackType

from chibi.interfaces.interface import UserInterface
from chibi.utils.ack_sentinel import ACK_SENTINEL

logger = logging.getLogger(__name__)


def delta_streaming_allowed(interface: UserInterface | None) -> bool:
    """Check whether live delta emission may start for the current request.

    Two conditions must hold: the interface must have opted in to streaming
    (``streaming_enabled``) and the request-local retry latch must be unset.
    Once any delta has been emitted for the request, every subsequent
    attempt, fallback and turn of that request must stream no further live
    deltas (plan D2).

    Args:
        interface: The active user interface, if any.

    Returns:
        True when streaming may emit live deltas for this request.
    """
    if interface is None:
        return False
    if getattr(interface, "delta_emitted", False):
        return False
    return getattr(interface, "streaming_enabled", False) is True


def delta_latched(interface: UserInterface | None) -> bool:
    """Check whether the request latch forbids further live delta emission.

    Args:
        interface: The active user interface, if any.

    Returns:
        True when at least one delta was already emitted for the request.
    """
    return getattr(interface, "delta_emitted", False) is True


def non_streaming_fallback_spent_error(interface: UserInterface | None, provider_name: str) -> BaseException | None:
    """Return the error raised by this provider's already-spent non-streaming fallback.

    The failover walk reuses one request interface for every provider
    attempt, so the stored entry carries the name of the provider that spent
    its fallback: only that provider's retry re-entry may consume it, while
    every other provider attempt still gets its own fresh fallback.

    Args:
        interface: The active user interface, if any.
        provider_name: Name of the provider asking for the spent entry.

    Returns:
        The exception the provider's single non-streaming fallback attempt
        raised, or None while that provider's fallback has not failed yet.
    """
    if interface is None:
        return None
    spent = getattr(interface, "non_streaming_fallback_error", None)
    if spent is None:
        return None
    spent_provider, error = spent
    return error if spent_provider == provider_name else None


def mark_non_streaming_fallback_spent(
    interface: UserInterface | None, provider_name: str, error: BaseException
) -> None:
    """Store the error raised by the provider's single non-streaming fallback attempt.

    A stream rejected with 400/404 falls back to one non-streaming call. When
    that call fails with a transient error, the retry decorator around
    ``get_chat_response`` re-enters the whole method, the stream creation fails
    deterministically again, and the stored error is re-raised on re-entry: the
    fallback is never issued twice, while the transient error keeps driving the
    retries and the provider failover classification. The entry is keyed to the
    provider that spent its fallback, because the failover walk shares one
    interface across provider attempts: a foreign provider must neither
    inherit the suppression nor re-raise another provider's error.

    Args:
        interface: The active user interface, if any.
        provider_name: Name of the provider spending its fallback.
        error: The exception raised by the fallback attempt.
    """
    if interface is None:
        return
    interface.non_streaming_fallback_error = (provider_name, error)


class AckSentinelStreamGate:
    """Buffer live deltas until the streamed answer can no longer be a pure ACK sentinel.

    While the accumulated body text (ignoring surrounding whitespace) is
    still compatible with the sentinel, nothing is emitted: an answer that
    ends up being exactly the sentinel must never reach the interface as a
    delta. As soon as the text diverges from the sentinel, the gate releases the whole
    buffer and passes further chunks through, stripping any complete sentinel
    occurrences. A short trailing fragment that could still complete into the
    sentinel with the next chunk is held back and emitted by :meth:`flush`.
    """

    def __init__(self) -> None:
        """Initialize an empty, unreleased gate."""
        self._buffer = ""
        self._tail = ""
        self._released = False

    def filter(self, text: str) -> str:
        """Filter a single streamed body-text chunk.

        Args:
            text: The next body-text chunk from the stream.

        Returns:
            The part of the text that may be emitted right now (possibly empty).
        """
        if not text:
            return ""
        if self._released:
            return self._strip_sentinel(text)
        self._buffer += text
        stripped = self._buffer.strip()
        if ACK_SENTINEL.startswith(stripped):
            return ""
        self._released = True
        return self._strip_sentinel(self._buffer)

    def flush(self) -> str:
        """Drain the gate at the end of the stream.

        Returns:
            Any held-back text that still must be emitted; empty when the whole
            streamed answer turned out to be the pure sentinel.
        """
        if self._released:
            return self._drain_tail()
        remainder = self._buffer
        self._buffer = ""
        self._released = True
        if remainder.strip() == ACK_SENTINEL:
            return ""
        return self._strip_sentinel(remainder) + self._drain_tail()

    def _strip_sentinel(self, text: str) -> str:
        """Remove complete sentinels and hold back a possible partial-sentinel tail.

        Args:
            text: Incoming text (released mode) or the released buffer.

        Returns:
            Text safe to emit now; a suffix that could still complete into the
            sentinel is retained until the next chunk or :meth:`flush`.
        """
        combined = self._tail + text
        self._tail = ""
        while True:
            index = combined.find(ACK_SENTINEL)
            if index < 0:
                break
            combined = combined[:index] + combined[index + len(ACK_SENTINEL) :]
        for size in range(min(len(combined), len(ACK_SENTINEL) - 1), 0, -1):
            if ACK_SENTINEL.startswith(combined[-size:]):
                self._tail = combined[-size:]
                return combined[:-size]
        return combined

    def _drain_tail(self) -> str:
        """Return and clear the held-back partial-sentinel tail."""
        tail = self._tail
        self._tail = ""
        return tail

    @property
    def released(self) -> bool:
        """Report whether filtering has already released buffered text.

        A gate that never released has only seen text still compatible with
        the sentinel; everything it holds is speculative partial output of an
        answer that has not materialized yet.
        """
        return self._released


class AckSentinelStreamSession:
    """Live delta channel scoped to a single streaming attempt.

    The session owns the ACK sentinel gate for exactly one streaming attempt:
    the gate is created with the session, lives only inside the session, and is
    drained when the session exits. Nothing is stored on the interface, so
    buffered state cannot leak across attempts, fallbacks, requests or runner
    lifetimes, and the terminal drain runs on every exit path of the
    ``async with`` block: normal completion, early ``return``, raised
    exceptions and cancellation alike.

    Emission is possible only through :meth:`emit` on the session object. There
    is no free delta-emission function in this module, so an adapter that
    forgets the lifecycle has no way to stream at all, instead of silently
    streaming through a stale or undrained gate.
    """

    def __init__(self, interface: UserInterface | None) -> None:
        """Bind the session to one streaming attempt's interface.

        Args:
            interface: The active user interface, if any.
        """
        self._interface = interface
        self._gate = AckSentinelStreamGate()
        self._entered = False

    async def __aenter__(self) -> AckSentinelStreamSession:
        """Enter the streaming attempt with a fresh gate."""
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Drain the gate on every exit path of the streaming attempt."""
        await self._finish(aborted=exc_type is not None)

    async def emit(self, text: str) -> None:
        """Emit a single live response-text chunk and latch the current request.

        Empty chunks and non-streaming interfaces are skipped silently. The text
        is routed through the session's ACK sentinel gate first, so a pure-ACK
        answer never reaches the interface and complete sentinels are stripped
        from mixed answers on every adapter. Calling this sets the latch on the
        interface, so retries, fallbacks and recursive turns of the same request
        never stream live deltas again. The latch does not gate emission within
        the currently-live turn; that is governed by the provider's turn-level
        tool-call signal.

        Args:
            text: The partial body-text chunk to stream.
        """
        if not self._entered:
            raise RuntimeError(
                "AckSentinelStreamSession.emit called outside its 'async with' block. Open the "
                "session with 'async with AckSentinelStreamSession(interface) as deltas' so the "
                "gate is guaranteed to be drained on every exit path of the streaming attempt."
            )
        interface = self._interface
        if interface is None or not text:
            return
        if getattr(interface, "streaming_enabled", False) is not True:
            return
        emit_text = self._gate.filter(text)
        if not emit_text:
            return
        await interface.send_delta(emit_text)
        interface.delta_emitted = True

    async def _finish(self, aborted: bool) -> None:
        """Drain the session gate and emit any held-back text.

        On a normal exit every held-back fragment is emitted: the stream is
        over, so a trailing fragment can no longer complete into the sentinel
        and is legitimate answer text. On an aborted attempt a gate that never
        released holds only speculative output of an answer that did not
        complete, output a non-streaming fallback may regenerate, so it is
        drained and discarded; a released gate's short tail, provably part of an
        answer whose earlier chunks were already emitted, is still delivered.
        A failure while emitting during an abort is logged and swallowed so it
        cannot mask the in-flight exception.

        Args:
            aborted: Whether an exception is propagating out of the attempt.
        """
        interface = self._interface
        if interface is None:
            return
        if aborted and not self._gate.released:
            self._gate.flush()
            return
        held_back = self._gate.flush()
        if not held_back or getattr(interface, "streaming_enabled", False) is not True:
            return
        try:
            await interface.send_delta(held_back)
            interface.delta_emitted = True
        except Exception:
            if not aborted:
                raise
            logger.warning("Could not flush the ACK sentinel gate tail of an aborted stream.", exc_info=True)
