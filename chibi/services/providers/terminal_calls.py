"""Terminal tool-call handling for the provider agent loops.

``say`` and ``ack_silently`` (sub-task 4a) are terminal tools: a model
response that carries one of them ends the turn immediately. They are never
dispatched through the tool registry and never receive a tool result — a
result would send the turn back to the model, which reproduces the infinite
``say()`` → result → ``ack_silently()`` → ``ack_silently()`` → ... loop that
reconnaissance reproduced empirically (recon C7, design §3.2). Instead the
terminal call is normalized at ingest into a plain assistant ``Message``:

- ``say(text)`` → ``Message(content=text, visibility="visible")`` and the
  turn's answer is the delivered text;
- ``ack_silently()`` → ``Message(content=<private CoT>, visibility="hidden")``
  and the turn's answer is empty. When there is no CoT and no delivered text
  the assistant turn is omitted entirely (design R2: never persist an empty
  text block).
"""

from typing import NamedTuple

from loguru import logger

from chibi.models import Message
from chibi.services.providers.tools.common import AckSilentlyTool, SayTool
from chibi.services.providers.tools.schemas import ToolCallSchema

SAY_TOOL_NAME = SayTool.name
ACK_SILENTLY_TOOL_NAME = AckSilentlyTool.name
TERMINAL_TOOL_NAMES = frozenset({SAY_TOOL_NAME, ACK_SILENTLY_TOOL_NAME})


class TerminalTurnPlan(NamedTuple):
    """Outcome of a terminal turn, ready to be persisted and returned.

    Attributes:
        answer: The turn's user-facing answer. The delivered text for a
            ``say`` turn, empty for a silent turn.
        message: The normalized assistant message to persist, or None when
            the turn would persist an empty-content message and must be
            omitted entirely (design R2).
    """

    answer: str
    message: Message | None


def is_terminal_call(tool_name: str) -> bool:
    """Check whether a parsed tool call is one of the terminal tools.

    Args:
        tool_name: The parsed tool name from the model response.

    Returns:
        True when the call is ``say`` or ``ack_silently``.
    """
    return tool_name in TERMINAL_TOOL_NAMES


def partition_terminal_calls(calls: list[ToolCallSchema]) -> tuple[list[ToolCallSchema], list[ToolCallSchema]]:
    """Split parsed tool calls into terminal and ordinary work calls.

    Args:
        calls: Parsed tool calls from the model response.

    Returns:
        A tuple of (terminal calls, work calls), both preserving the
        original order.
    """
    terminal_calls = [call for call in calls if is_terminal_call(call.tool_name)]
    work_calls = [call for call in calls if not is_terminal_call(call.tool_name)]
    return terminal_calls, work_calls


def _first_say_call(terminal_calls: list[ToolCallSchema]) -> ToolCallSchema | None:
    return next((call for call in terminal_calls if call.tool_name == SAY_TOOL_NAME), None)


def plan_terminal_turn(terminal_calls: list[ToolCallSchema], thoughts: str | None) -> TerminalTurnPlan:
    """Build the normalized turn for a response whose calls are all terminal.

    Exactly one terminal call is honoured per turn; extra ones are ignored
    with a warning. A ``say`` turn delivers its ``text`` argument and persists
    it as a visible assistant message. An ``ack_silently`` turn delivers
    nothing and persists the private CoT text as a hidden message; when the
    CoT is empty the assistant turn is omitted entirely (design R2) instead of
    persisting an empty-content message.

    Args:
        terminal_calls: The terminal calls parsed from the response (must be
            non-empty; use ``partition_terminal_calls`` first).
        thoughts: The assistant's private text channel (CoT) for this turn.

    Returns:
        The terminal turn plan with the turn's answer and the normalized
        message (or None when nothing may be persisted).
    """
    say_call = _first_say_call(terminal_calls)
    if say_call is not None:
        text = str(say_call.args.get("text") or "")
        message = Message(role="assistant", content=text, visibility="visible") if text else None
        return TerminalTurnPlan(answer=text, message=message)

    if len(terminal_calls) > 1:
        logger.warning(f"Ignoring {len(terminal_calls) - 1} redundant ack_silently call(s) in one turn.")

    content = thoughts or ""
    message = Message(role="assistant", content=content, visibility="hidden") if content else None
    return TerminalTurnPlan(answer="", message=message)


def fold_terminal_calls_into_thoughts(thoughts: str | None, terminal_calls: list[ToolCallSchema]) -> str:
    """Fold a mixed turn's terminal payloads into the assistant text channel.

    A backend that ignores the parallel-tool-use ban may emit a terminal call
    alongside work calls (design §3.2). The turn continues with the work
    calls only, and the terminal call's text is folded into the assistant
    content so nothing is silently dropped; the model is expected to
    terminate cleanly on a later step.

    Args:
        thoughts: The assistant's private text channel (CoT) for this turn.
        terminal_calls: The terminal calls parsed from the same response.

    Returns:
        The text channel with the folded ``say`` text appended.
    """
    say_call = _first_say_call(terminal_calls)
    if say_call is None:
        return thoughts or ""

    text = str(say_call.args.get("text") or "")
    if not text:
        return thoughts or ""

    return f"{thoughts}\n{text}" if thoughts else text
