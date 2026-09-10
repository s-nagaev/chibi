"""Runtime working-directory change events for IDE stdio clients.

Deep code (the ``set_working_dir`` tool) knows nothing about the IDE
transport, yet it is the code that actually changes an agent thread's
effective working directory. The tracker is the tiny bridge between the
two: the IDE stdio runner registers itself as the active sink when the
client opted in via the ``cwd_updates`` capability, and every working
-directory change call is forwarded to that sink. Without a sink (every
non-IDE runner, and IDE sessions without the capability) the calls are
silent no-ops with zero wire traffic.
"""

from __future__ import annotations

from typing import Protocol

from chibi.utils.app import SingletonMeta


class CwdEventSink(Protocol):
    """Structural interface the IDE stdio runner exposes to the tracker."""

    def cwd_changed(self, thread_id: int) -> None:
        """Report that a thread's effective working directory changed."""
        ...


class CwdEventTracker(metaclass=SingletonMeta):
    """Routes working-directory change calls from tool code to the active session sink."""

    def __init__(self) -> None:
        """Initialize the sink slot once (the metaclass may re-enter __init__)."""
        if not hasattr(self, "_sink"):
            self._sink: CwdEventSink | None = None

    def set_sink(self, sink: CwdEventSink | None) -> None:
        """Bind or clear the sink that receives working-directory change calls.

        Args:
            sink: The active session's sink, or None when no session opted in.
        """
        self._sink = sink

    def release(self, sink: CwdEventSink) -> None:
        """Clear the sink only when the given session still owns the slot.

        Args:
            sink: The sink that wants to unregister itself.
        """
        if self._sink is sink:
            self._sink = None

    def cwd_changed(self, thread_id: int) -> None:
        """Forward a working-directory change to the active sink.

        Args:
            thread_id: The thread whose effective working directory changed.
        """
        if self._sink is not None:
            self._sink.cwd_changed(thread_id)


cwd_tracker = CwdEventTracker()
