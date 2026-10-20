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

from collections.abc import Callable, Sequence
from typing import NamedTuple, TypeVar

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from chibi.models import Message
from chibi.services.interface import UserInterface
from chibi.services.providers.tools.common import AckSilentlyTool, SayTool
from chibi.services.providers.tools.schemas import ToolCallSchema

SAY_TOOL_NAME = SayTool.name
ACK_SILENTLY_TOOL_NAME = AckSilentlyTool.name
TERMINAL_TOOL_NAMES = frozenset({SAY_TOOL_NAME, ACK_SILENTLY_TOOL_NAME})

T = TypeVar("T")


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


def offered_tools(
    tools: Sequence[T],
    interface: UserInterface | None,
    name_of: Callable[[T], str],
) -> list[T]:
    """Return the tools a request may be offered (ruling D1: role-derived visibility).

    The rule: a request's tool list contains the terminal tools ``say`` and
    ``ack_silently`` if and only if the request carries a user interface.
    Interface-less turns — delegated sub-agents at any depth and the emergency
    summarizer — are offered neither tool and must end with a non-empty
    plain-text report. This is a likelihood reduction, not a guarantee: the
    terminal interception in the agent loops is by tool-name membership and
    never consults the offered list, so a hallucinated terminal call still ends
    the turn. The structural closure of that residual is the loud-empty
    delegation rule in ``DelegateTool``.

    The signal is **interface presence**, not an explicit role parameter. That
    is a deliberate coupling: at the three loop entry points (master
    ``chibi/services/user.py``, sub-agent
    ``chibi/services/providers/tools/utils.py``, summarizer
    ``chibi/services/user.py``) "``interface is None``" holds exactly when the
    request has no user channel, so no role plumbing through five loops is
    needed. If a future actor class breaks that equivalence (an interface-less
    turn that may legitimately speak to the user), this helper and the tests
    pinning the assumption are the places to revisit first.

    Args:
        tools: The candidate tool list in the caller's native format.
        interface: The user interface carried by the request, or None.
        name_of: Extracts the tool name from one list entry (the formats
            differ per adapter: OpenAI-style dicts, Anthropic ``ToolParam``,
            Google ``Tool``).

    Returns:
        The offered tool list: unchanged when the request carries a user
        interface, otherwise the list minus the terminal tools.
    """
    if interface is not None:
        return list(tools)
    return [tool for tool in tools if name_of(tool) not in TERMINAL_TOOL_NAMES]


def offered_tool_definitions(
    definitions: Sequence[ChatCompletionToolParam],
    interface: UserInterface | None,
) -> list[ChatCompletionToolParam]:
    """``offered_tools`` for OpenAI-format tool definitions (name under function).

    Args:
        definitions: Tool definitions in the OpenAI chat-completions format.
        interface: The user interface carried by the request, or None.

    Returns:
        The offered tool list per the D1 rule.
    """
    return offered_tools(definitions, interface, name_of=lambda definition: definition["function"]["name"])
