"""Per-request subagent lifecycle events for IDE stdio clients.

Deep code (the delegation tool, bot command handlers) knows nothing about
the IDE transport, yet it is the code that actually spawns and kills
subagents. The tracker is the tiny bridge between the two: the IDE stdio
runner registers itself as the active sink when the client opted in, and
every subagent lifecycle call is forwarded to that sink. Without a sink
(every non-IDE runner, and IDE sessions without the ``subagents``
capability) the calls are silent no-ops with zero wire traffic.
"""

from __future__ import annotations

from typing import Protocol

from chibi.utils.app import SingletonMeta


class SubagentEventSink(Protocol):
    """Structural interface the IDE stdio runner exposes to the tracker."""

    def subagent_started(self, thread_id: int, name: str | None = None) -> None:
        """Count one accepted subagent start for the request on this thread."""
        ...

    def subagent_finished(self, thread_id: int, name: str | None = None) -> None:
        """Count one subagent finish for the request on this thread."""
        ...

    def kill_flush(self, thread_id: int) -> None:
        """Emit the single kill-flush event for the request on this thread."""
        ...

    async def drain_subagent_events(self) -> None:
        """Wait until every in-flight agent_event emission has been written."""
        ...


class SubagentEventTracker(metaclass=SingletonMeta):
    """Routes subagent lifecycle calls from tool code to the active session sink."""

    def __init__(self) -> None:
        """Initialize the sink slot once (the metaclass may re-enter __init__)."""
        if not hasattr(self, "_sink"):
            self._sink: SubagentEventSink | None = None

    def set_sink(self, sink: SubagentEventSink | None) -> None:
        """Bind or clear the sink that receives lifecycle calls.

        Args:
            sink: The active session's sink, or None when no session opted in.
        """
        self._sink = sink

    def release(self, sink: SubagentEventSink) -> None:
        """Clear the sink only when the given session still owns the slot.

        Args:
            sink: The sink that wants to unregister itself.
        """
        if self._sink is sink:
            self._sink = None

    def subagent_started(self, thread_id: int, name: str | None = None) -> None:
        """Forward a subagent start to the active sink.

        Args:
            thread_id: Thread whose running request owns the counters.
            name: Trivially available subagent label (the delegated model), if any.
        """
        if self._sink is not None:
            self._sink.subagent_started(thread_id, name)

    def subagent_finished(self, thread_id: int, name: str | None = None) -> None:
        """Forward a subagent finish to the active sink.

        Args:
            thread_id: Thread whose running request owns the counters.
            name: Trivially available subagent label, if any.
        """
        if self._sink is not None:
            self._sink.subagent_finished(thread_id, name)

    def kill_flush(self, thread_id: int) -> None:
        """Forward a kill-flush request to the active sink.

        Args:
            thread_id: Thread whose request state is being retired.
        """
        if self._sink is not None:
            self._sink.kill_flush(thread_id)

    async def drain(self) -> None:
        """Wait until the active sink has written every queued emission."""
        if self._sink is not None:
            await self._sink.drain_subagent_events()


subagent_tracker = SubagentEventTracker()
