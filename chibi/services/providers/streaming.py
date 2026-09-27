"""Shared live response-delta streaming helpers for LLM provider adapters."""

from chibi.services.interface import UserInterface


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


async def emit_delta(interface: UserInterface | None, text: str) -> None:
    """Emit a single live response-text chunk and latch the current request.

    Empty chunks and non-streaming interfaces are skipped silently. Calling
    this sets the latch on the interface, so retries, fallbacks and
    recursive turns of the same request never stream live deltas again. The
    latch does not gate emission within the currently-live turn; that is
    governed by the provider's turn-level tool-call signal.

    Args:
        interface: The active user interface, if any.
        text: The partial body-text chunk to stream.
    """
    if interface is None or not text:
        return
    if getattr(interface, "streaming_enabled", False) is not True:
        return
    await interface.send_delta(text)
    interface.delta_emitted = True
