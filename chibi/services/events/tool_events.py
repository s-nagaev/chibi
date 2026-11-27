"""Per-request tool-call lifecycle events for IDE stdio clients.

Deep code (the ``ChibiTool`` dispatch path) knows nothing about the IDE
transport, yet it is the code that actually runs every foreground tool call.
The tracker is the tiny bridge between the two: the IDE stdio runner registers
itself as the active sink when the client opted in to the ``tool_events``
capability, and every foreground tool call is forwarded to that sink. Without a
sink (every non-IDE runner, and IDE sessions without the ``tool_events``
capability) the calls are silent no-ops with zero wire traffic.

A tool call only ever reaches the tracker when a request context exists: the
dispatch path skips background runs and interface-less (sub-agent-internal)
calls before calling it, and the sink additionally resolves the owning request
id from its active-request map, so a call with no live request emits nothing.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from chibi.utils.app import SingletonMeta


class ToolEventSink(Protocol):
    """Structural interface the IDE stdio runner exposes to the tracker."""

    async def tool_started(self, thread_id: int, name: str, args: dict[str, Any]) -> None:
        """Emit a ``tool_event`` start frame for the active request on this thread."""
        ...

    async def tool_finished(self, thread_id: int, name: str, ok: bool, result_preview: str | None) -> None:
        """Emit a ``tool_event`` end frame for the active request on this thread."""
        ...


class ToolEventTracker(metaclass=SingletonMeta):
    """Routes foreground tool-call lifecycle calls to the active session sink."""

    def __init__(self) -> None:
        """Initialize the sink slot once (the metaclass may re-enter __init__)."""
        if not hasattr(self, "_sink"):
            self._sink: ToolEventSink | None = None

    @property
    def enabled(self) -> bool:
        """Report whether a sink is currently bound, i.e. a client opted in.

        Returns:
            True when a session has bound itself as the receiver of tool events.
        """
        return self._sink is not None

    def set_sink(self, sink: ToolEventSink | None) -> None:
        """Bind or clear the sink that receives lifecycle calls.

        Args:
            sink: The active session's sink, or None when no session opted in.
        """
        self._sink = sink

    def release(self, sink: ToolEventSink) -> None:
        """Clear the sink only when the given session still owns the slot.

        Args:
            sink: The sink that wants to unregister itself.
        """
        if self._sink is sink:
            self._sink = None

    async def tool_started(self, thread_id: int, name: str, args: dict[str, Any]) -> None:
        """Forward a tool start to the active sink.

        A failing emission must never break the tool it describes, so any
        error raised by the sink is logged and swallowed here.

        Args:
            thread_id: Thread whose running request owns the tool call.
            name: The tool name being invoked.
            args: The tool arguments, as supplied by the model.
        """
        if self._sink is None:
            return
        try:
            await self._sink.tool_started(thread_id, name, args)
        except Exception:
            logger.exception("Failed to deliver a tool_event start for thread {}", thread_id)

    async def tool_finished(self, thread_id: int, name: str, ok: bool, result_preview: str | None) -> None:
        """Forward a tool finish to the active sink.

        A failing emission must never break the tool it describes, so any
        error raised by the sink is logged and swallowed here.

        Args:
            thread_id: Thread whose running request owns the tool call.
            name: The tool name that finished.
            ok: Whether the tool reported a successful result.
            result_preview: A textual preview of the tool result, if any.
        """
        if self._sink is None:
            return
        try:
            await self._sink.tool_finished(thread_id, name, ok, result_preview)
        except Exception:
            logger.exception("Failed to deliver a tool_event end for thread {}", thread_id)


tool_event_tracker = ToolEventTracker()
