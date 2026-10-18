"""Shared live response-delta streaming helpers for LLM provider adapters."""

from chibi.services.interface import UserInterface
from chibi.utils.ack_sentinel import ACK_SENTINEL


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


_ACK_SENTINEL_GATE_ATTR = "_ack_sentinel_stream_gate"


def _get_ack_sentinel_gate(interface: UserInterface) -> AckSentinelStreamGate:
    """Return the interface-attached ACK gate, creating it lazily.

    Args:
        interface: The active user interface.

    Returns:
        The gate instance bound to this interface for the current attempt.
    """
    gate = getattr(interface, _ACK_SENTINEL_GATE_ATTR, None)
    if gate is None:
        gate = AckSentinelStreamGate()
        setattr(interface, _ACK_SENTINEL_GATE_ATTR, gate)
    return gate


def reset_ack_sentinel_gate(interface: UserInterface | None) -> None:
    """Attach a fresh ACK gate to the interface for a new streaming attempt.

    Every streaming adapter calls this once before its stream loop, so the
    buffered state never leaks across attempts, fallbacks or turns.

    Args:
        interface: The active user interface, if any.
    """
    if interface is None:
        return
    setattr(interface, _ACK_SENTINEL_GATE_ATTR, AckSentinelStreamGate())


async def flush_ack_sentinel_gate(interface: UserInterface | None) -> None:
    """Drain the interface-attached ACK gate and emit any held-back text.

    Called once per streaming attempt after the stream loop. For a pure-ACK
    answer the gate returns nothing; otherwise a held-back partial-sentinel
    tail (a legitimate answer ending inside the sentinel pattern) is emitted
    here. The drained text bypasses re-gating — it is already sentinel-safe.

    Args:
        interface: The active user interface, if any.
    """
    if interface is None:
        return
    gate = getattr(interface, _ACK_SENTINEL_GATE_ATTR, None)
    if gate is None:
        return
    held_back = gate.flush()
    if held_back and getattr(interface, "streaming_enabled", False) is True:
        await interface.send_delta(held_back)
        interface.delta_emitted = True


async def emit_delta(interface: UserInterface | None, text: str) -> None:
    """Emit a single live response-text chunk and latch the current request.

    Empty chunks and non-streaming interfaces are skipped silently. The text
    is routed through the interface-attached ACK sentinel gate first, so a
    pure-ACK answer never reaches the interface and complete sentinels are
    stripped from mixed answers on every adapter, including ones that forgot
    to wire the gate explicitly. Calling this sets the latch on the
    interface, so retries, fallbacks and recursive turns of the same request
    never stream live deltas again. The latch does not gate emission within
    the currently-live turn; that is governed by the provider's turn-level
    tool-call signal.

    Args:
        interface: The active user interface, if any.
        text: The partial body-text chunk to stream.
    """
    if interface is None or not text:
        return
    if getattr(interface, "streaming_enabled", False) is not True:
        return
    emit_text = _get_ack_sentinel_gate(interface).filter(text)
    if not emit_text:
        return
    await interface.send_delta(emit_text)
    interface.delta_emitted = True
