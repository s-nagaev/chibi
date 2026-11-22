"""Shared live response-delta streaming helpers for LLM provider adapters.

Live deltas leave the provider layer only through
:class:`AckSentinelStreamSession`, an async context manager that owns the ACK
sentinel gate for exactly one streaming attempt. Emission and lifecycle are the
same object: an adapter that does not open a session has no way to emit a
delta, and one that does cannot leave the gate undrained or shared across
attempts.

A turn may carry its visible answer in two shapes: as body text (streamed
directly, while no tool call has appeared in the chunk stream) or as the
``text`` argument of the terminal ``say`` tool call. The second shape is
recovered by :class:`SayAnswerDeltaStreamer`, which every adapter feeds with
the accumulated arguments of the tool calls it observes.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from types import TracebackType

from chibi.interfaces.interface import UserInterface
from chibi.services.providers.terminal_calls import SAY_TOOL_NAME, TERMINAL_TOOL_NAMES
from chibi.utils.ack_sentinel import ACK_SENTINEL

logger = logging.getLogger(__name__)

SAY_TEXT_ARGUMENT = "text"

_JSON_STRING_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HIGH_SURROGATE_MIN = 0xD800
_HIGH_SURROGATE_MAX = 0xDBFF
_LOW_SURROGATE_MIN = 0xDC00
_LOW_SURROGATE_MAX = 0xDFFF


def _argument_value_start(raw_arguments: str, key: str) -> int | None:
    """Locate the opening quote of a string-valued argument in a partial JSON object.

    Every occurrence of the quoted key is considered, so a key mentioned
    inside an earlier string value cannot be mistaken for the argument: the
    candidate is accepted only when a colon and the value's opening quote
    follow it, which no string content can produce.

    Args:
        raw_arguments: The accumulated tool-call arguments JSON.
        key: The string-valued key to locate.

    Returns:
        The index just past the value's opening quote, or None when that
        opening quote has not arrived yet.
    """
    needle = f'"{key}"'
    search_from = 0
    while True:
        index = raw_arguments.find(needle, search_from)
        if index < 0:
            return None
        cursor = index + len(needle)
        while cursor < len(raw_arguments) and raw_arguments[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(raw_arguments) and raw_arguments[cursor] == ":":
            cursor += 1
            while cursor < len(raw_arguments) and raw_arguments[cursor] in " \t\r\n":
                cursor += 1
            if cursor < len(raw_arguments) and raw_arguments[cursor] == '"':
                return cursor + 1
            if cursor >= len(raw_arguments):
                return None
        search_from = index + len(needle)


def _decode_json_string_prefix(raw_arguments: str, start: int) -> str:
    """Decode a JSON string value from a partial document, stopping at the first ambiguity.

    The walk handles every escape sequence the JSON grammar defines, including
    surrogate pairs, and returns as soon as the input stops being decodable
    without guessing: a truncated escape, a partial unicode escape or a
    high surrogate whose low companion has not arrived yet all end the walk
    with the prefix decoded so far. Re-running the function on a longer
    accumulated fragment therefore only ever extends the result, and a
    complete document decodes to exactly ``json.loads(document)[key]``.

    Args:
        raw_arguments: The accumulated tool-call arguments JSON.
        start: Index of the first character of the value (past its opening quote).

    Returns:
        The decoded value prefix.
    """
    decoded: list[str] = []
    cursor = start
    while cursor < len(raw_arguments):
        char = raw_arguments[cursor]
        if char == '"':
            break
        if char != "\\":
            decoded.append(char)
            cursor += 1
            continue
        if cursor + 1 >= len(raw_arguments):
            break
        escape = raw_arguments[cursor + 1]
        if escape != "u":
            mapped = _JSON_STRING_ESCAPES.get(escape)
            if mapped is None:
                break
            decoded.append(mapped)
            cursor += 2
            continue
        code, consumed = _decode_unicode_escape(raw_arguments, cursor)
        if code is None:
            break
        decoded.append(code)
        cursor += consumed
    return "".join(decoded)


def _decode_unicode_escape(raw_arguments: str, cursor: int) -> tuple[str | None, int]:
    """Decode one ``\\uXXXX`` escape, pairing it with a low surrogate when present.

    Args:
        raw_arguments: The accumulated tool-call arguments JSON.
        cursor: Index of the escape's leading backslash.

    Returns:
        The decoded character and the number of characters the escape
        occupies, or (None, 0) when the escape is not decodable yet.
    """
    if cursor + 6 > len(raw_arguments):
        return None, 0
    digits = raw_arguments[cursor + 2 : cursor + 6]
    if any(digit not in _HEX_DIGITS for digit in digits):
        return None, 0
    code = int(digits, 16)
    if _HIGH_SURROGATE_MIN <= code <= _HIGH_SURROGATE_MAX:
        if cursor + 8 > len(raw_arguments):
            return None, 0
        if raw_arguments[cursor + 6 : cursor + 8] != "\\u":
            return None, 0
        if cursor + 12 > len(raw_arguments):
            return None, 0
        low_digits = raw_arguments[cursor + 8 : cursor + 12]
        if any(digit not in _HEX_DIGITS for digit in low_digits):
            return None, 0
        low = int(low_digits, 16)
        if not _LOW_SURROGATE_MIN <= low <= _LOW_SURROGATE_MAX:
            return None, 0
        combined = 0x10000 + ((code - _HIGH_SURROGATE_MIN) << 10) + (low - _LOW_SURROGATE_MIN)
        return chr(combined), 12
    if _LOW_SURROGATE_MIN <= code <= _LOW_SURROGATE_MAX:
        return None, 0
    return chr(code), 6


def partial_json_string_argument(raw_arguments: str, key: str) -> str | None:
    """Decode the string value of ``key`` from possibly-incomplete JSON arguments.

    Provider adapters streaming tool-call arguments forward the arguments JSON
    fragment by fragment, so the value can be cut anywhere: before the key,
    inside the value, in the middle of an escape sequence or in the middle of
    a unicode escape. The function decodes the prefix that is already
    unambiguous and stops at the first ambiguous character, so calling it
    again on the next accumulated fragment yields a monotonically growing
    string that ends up exactly equal to ``json.loads(raw_arguments)[key]``
    once the JSON is complete.

    Args:
        raw_arguments: The accumulated tool-call arguments JSON.
        key: The string-valued key to decode.

    Returns:
        The decoded value prefix, or None when the key or the value's opening
        quote has not arrived yet.
    """
    start = _argument_value_start(raw_arguments, key)
    if start is None:
        return None
    return _decode_json_string_prefix(raw_arguments, start)


class SayAnswerDeltaStreamer:
    """Recover the live visible answer of a terminal ``say`` call from its streamed arguments.

    A turn that ends through the ``say`` tool call delivers its user-facing
    answer inside the call's ``text`` argument, and the adapters stop emitting
    body-text deltas from the first tool-call signal of the turn on, because a
    function-call turn carries no body text to stream. Without this streamer
    the whole answer therefore reached the client only in the terminal
    ``result`` frame and live streaming appeared broken on every model that
    answers through ``say``.

    The streamer follows at most one call, the first ``say`` of the response,
    and returns the not-yet-returned suffix of its ``text`` argument on every
    :meth:`update`. The returned text is always a prefix of the value the
    result frame will carry, whatever the provider's fragmentation pattern:

    - providers that stream the arguments in fragments (OpenAI-compatible,
      Anthropic, Mistral) produce several progressively growing chunks, which
      the delta channel coalesces on its own schedule;
    - providers that deliver the whole call at once (Gemini) produce one chunk
      carrying the whole text.

    Guards, so the delta channel never carries text the user is not being
    answered with:

    - an ordinary (work) tool call anywhere in the response blocks the
      streamer for good. Such a mixed turn folds the ``say`` text back into
      the private reasoning channel and ends on a later, different answer, so
      the final frame supersedes whatever the mixed turn streamed. Residual,
      documented: text emitted before the work call appeared cannot be
      retracted, and the client replaces it on the result frame;
    - ``ack_silently`` and any other terminal call without a visible payload
      are ignored without blocking, because a ``say`` call in the same
      response still wins (``plan_terminal_turn`` honours the first ``say``).
    """

    def __init__(self) -> None:
        """Initialize an unarmed streamer that follows no call."""
        self._armed = False
        self._followed_index: Hashable | None = None
        self._emitted_chars = 0
        self._blocked = False

    @property
    def blocked(self) -> bool:
        """Report whether an ordinary tool call was seen, which stops all emission.

        Returns:
            True once the response has shown a non-terminal tool call.
        """
        return self._blocked

    def update(self, tool_name: str | None, raw_arguments: str, index: Hashable | None = None) -> str:
        """Register one accumulated tool call and return the new ``say`` answer text.

        Args:
            tool_name: The call's function name, or None while the provider has
                not revealed it yet.
            raw_arguments: The call's accumulated raw arguments JSON.
            index: The call's stream index, when the provider keys its calls by
                one. Calls of one response must be distinguishable, otherwise a
                second terminal call would be read as a continuation of the
                first.

        Returns:
            The part of the followed ``say`` call's ``text`` argument that has
            not been returned yet, or an empty string.
        """
        if tool_name is not None and tool_name not in TERMINAL_TOOL_NAMES:
            self._blocked = True
        if self._blocked:
            return ""
        if tool_name == SAY_TOOL_NAME:
            if not self._armed:
                self._armed = True
                self._followed_index = index
            elif self._followed_index != index:
                return ""
        elif tool_name is None:
            if not self._armed or self._followed_index != index:
                return ""
        else:
            return ""
        text = partial_json_string_argument(raw_arguments, SAY_TEXT_ARGUMENT)
        if not text:
            return ""
        suffix = text[self._emitted_chars :]
        self._emitted_chars = len(text)
        return suffix


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
