"""Runtime thread-rename events for IDE stdio clients.

Deep code (the ``rename_thread`` tool and the hidden ``/rename_thread``
command) knows nothing about the IDE transport, yet it is the code that
changes an agent thread's display name. The tracker is the tiny bridge
between the two: the IDE stdio runner registers itself as the active sink
when the client opted in via the ``thread_renames`` capability, and every
thread-rename call is forwarded to that sink. Without a sink (every
non-IDE runner, and IDE sessions without the capability) the calls are
silent no-ops with zero wire traffic.
"""

from __future__ import annotations

from typing import Protocol

from chibi.utils.app import SingletonMeta


class ThreadNameEventSink(Protocol):
    """Structural interface the IDE stdio runner exposes to the tracker."""

    def thread_renamed(self, thread_id: int, name: str) -> None:
        """Report that a thread's display name changed.

        Args:
            thread_id: The thread whose display name changed.
            name: The new validated display name.
        """
        ...


class ThreadNameEventTracker(metaclass=SingletonMeta):
    """Routes thread-rename calls from tool and command code to the active session sink."""

    def __init__(self) -> None:
        """Initialize the sink slot once (the metaclass may re-enter __init__)."""
        if not hasattr(self, "_sink"):
            self._sink: ThreadNameEventSink | None = None

    def set_sink(self, sink: ThreadNameEventSink | None) -> None:
        """Bind or clear the sink that receives thread-rename calls.

        Args:
            sink: The active session's sink, or None when no session opted in.
        """
        self._sink = sink

    def release(self, sink: ThreadNameEventSink) -> None:
        """Clear the sink only when the given session still owns the slot.

        Args:
            sink: The sink that wants to unregister itself.
        """
        if self._sink is sink:
            self._sink = None

    def thread_renamed(self, thread_id: int, name: str) -> None:
        """Forward a thread rename to the active sink.

        Args:
            thread_id: The thread whose display name changed.
            name: The new validated display name.
        """
        if self._sink is not None:
            self._sink.thread_renamed(thread_id, name)


thread_name_tracker = ThreadNameEventTracker()
