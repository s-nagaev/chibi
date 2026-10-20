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

The module also hosts the turn-end validator (sub-task 4c): the shared rule
that judges how a turn ended on every adapter seam, costs at most one bounded
repair round-trip, and fails loudly instead of letting a turn pass silently.
"""

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from json import JSONDecodeError
from typing import Any, Literal, NamedTuple, TypeVar, cast

from loguru import logger
from openai.types.chat import ChatCompletionToolParam

from chibi.config import gpt_settings
from chibi.exceptions import SilentTurnError
from chibi.models import Message, MessageVisibility
from chibi.schemas.app import ChatResponseSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.tools.common import AckSilentlyTool, SayTool
from chibi.services.providers.tools.schemas import ToolCallSchema

SAY_TOOL_NAME = SayTool.name
ACK_SILENTLY_TOOL_NAME = AckSilentlyTool.name
TERMINAL_TOOL_NAMES = frozenset({SAY_TOOL_NAME, ACK_SILENTLY_TOOL_NAME})

_MISUSE_GREP_TOKEN = "TURN_MISUSE"
_FORCED_CHOICE_GREP_TOKEN = "FORCED_CHOICE"

# The minimal argument schema every adapter accepts for the terminal tools
# (recon C4: verified working on all five backends; mandatory on Mistral at
# the SDK layer even when empty).
EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

T = TypeVar("T")


class TerminalTurnPlan(NamedTuple):
    """Outcome of a terminal turn, ready to be persisted and returned.

    Attributes:
        answer: The turn's user-facing answer. The delivered text for a
            ``say`` turn, empty for a silent turn.
        message: The normalized assistant message to persist, or None when
            the turn would persist an empty-content message and must be
            omitted entirely (design R2).
        visibility: The turn's visibility: ``"visible"`` for a ``say`` turn,
            ``"hidden"`` for a silent ``ack_silently`` turn. Renderers key
            their suppression gate on this (task 5); an empty ``say`` text
            still reports ``"visible"`` because the 4c validator owns that
            ending (misuse ``empty_turn``), not the renderer gate.
    """

    answer: str
    message: Message | None
    visibility: MessageVisibility


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
        return TerminalTurnPlan(answer=text, message=message, visibility="visible")

    if len(terminal_calls) > 1:
        logger.warning(f"Ignoring {len(terminal_calls) - 1} redundant ack_silently call(s) in one turn.")

    content = thoughts or ""
    message = Message(role="assistant", content=content, visibility="hidden") if content else None
    return TerminalTurnPlan(answer="", message=message, visibility="hidden")


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


# ---------------------------------------------------------------------------
# Turn-end validator (sub-task 4c).
#
# Forcing is an optimisation; the validator is the guarantee. It runs at the
# turn-end seam of every adapter loop and judges the OUTCOME of the turn, not
# the offered tool list: terminal interception is by tool-name membership and
# offer-blind, so a hallucinated terminal call produces exactly the same
# empty outcome as an empty text report and is rejected identically.
# ---------------------------------------------------------------------------

WAKE_USER_MESSAGE = "user message"
WAKE_TOOL_RESPONSE = "tool response"
WAKE_SCHEDULER_TRIGGER = "scheduled_trigger"
WAKE_CAPTION = "system message"
WAKE_TYPES = frozenset({WAKE_USER_MESSAGE, WAKE_TOOL_RESPONSE, WAKE_SCHEDULER_TRIGGER, WAKE_CAPTION})
REPAIR_NUDGE_TYPE = "repair_nudge"
_BACKGROUND_DISPATCH_STATUS = "tool is running in background"

TurnEndOutcome = Literal["ok", "degraded_visible", "misuse"]
TurnEndMisuse = Literal["empty_turn", "hallucinated_ack", "overeager_ack", "no_terminal_call_text"]


class TurnEndVerdict(NamedTuple):
    """The validator's verdict for one turn ending.

    Attributes:
        outcome: ``ok`` — the turn may end as the model produced it;
            ``degraded_visible`` — an interface turn ended with plain text and
            no terminal call, which is delivered as visible (graceful
            degradation, behaviour preserved); ``misuse`` — the turn must
            either cost a repair round-trip or fail loudly.
        misuse: The misuse kind, set for both ``degraded_visible`` (measured
            under-use) and ``misuse``; None for ``ok``.
    """

    outcome: TurnEndOutcome
    misuse: TurnEndMisuse | None


def _json_payload(content: str | None) -> dict[str, Any] | None:
    """Parse a message content string into a JSON object, tolerating anything else.

    Args:
        content: The message content (wake prompts and tool results are
            JSON objects; plain user text and summaries are not).

    Returns:
        The parsed dict, or None when the content is not a JSON object.
    """
    if not content or not content.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(content)
    except (ValueError, JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def detect_wake_type(conversation: Sequence[Message]) -> str | None:
    """Detect what woke this turn from the conversation's last real user message.

    Every master-entry turn appends exactly one JSON wake prompt (user
    message, tool response, scheduler trigger or file caption); sub-agent
    prompts are JSON without a ``type`` key. The scan skips tool-result
    messages, which some adapters normalize as ``role="user"`` with a
    ``tool_call_id``.

    Args:
        conversation: The normalized conversation messages of the turn.

    Returns:
        One of ``WAKE_TYPES``, or None when the wake is not recognizable
        (unknown prompt shape). None is treated strictly by the validator.
    """
    for message in reversed(conversation):
        if message.role != "user" or message.tool_call_id:
            continue
        payload = _json_payload(message.content)
        if payload is None:
            return None
        wake_type = payload.get("type")
        return str(wake_type) if wake_type in WAKE_TYPES else None
    return None


def background_work_pending(conversation: Sequence[Message]) -> bool:
    """Check whether any background tool dispatch in the conversation is unanswered.

    A background dispatch is a tool result carrying the
    ``tool is running in background`` status; it is answered when a later
    wake prompt of type ``tool response`` reports back for the same tool
    name. Unknown shapes resolve permissively (no pending work), because a
    false "pending" only suppresses a check while a false "not pending"
    would nag a legitimate silence.

    Args:
        conversation: The normalized conversation messages of the turn.

    Returns:
        True when at least one background dispatch has no matching result.
    """
    pending: set[str] = set()
    for message in conversation:
        payload = _json_payload(message.content)
        if payload is None:
            continue
        if message.role == "tool" and payload.get("status") == _BACKGROUND_DISPATCH_STATUS:
            pending.add(str(payload.get("tool_name")))
        elif message.role == "user" and payload.get("type") == WAKE_TOOL_RESPONSE:
            tool_response = payload.get("tool_response")
            if isinstance(tool_response, dict):
                pending.discard(str(tool_response.get("tool_name")))
    return bool(pending)


def evaluate_turn_end(
    *,
    interface: UserInterface | None,
    answer: str | None,
    ended_by_terminal_call: bool,
    terminal_tool_name: str | None,
    conversation: Sequence[Message] = (),
) -> TurnEndVerdict:
    """Judge the outcome of a turn that is about to end.

    The rules, from the design authority:
    (a) an interface turn with no terminal call, no open background work and
        non-empty text degrades to visible (today's behaviour, preserved);
    (b) a turn with no terminal call and nothing delivered is a misuse
        (defensive: also covers the never-reproduced 200-OK-empty-parts);
    (c) an ``ack_silently()`` with nothing pending is a misuse;
    the class-aware ending rule: an interface-less turn may end only with a
    non-empty text report — an empty one (empty report, empty ``say``, or a
    hallucinated ``ack_silently()``) is a loud failure, wherever it comes
    from, because the validator sees the outcome, not the offer.

    Args:
        interface: The user interface carried by the request, or None for
            interface-less turns (delegated sub-agents, the summarizer).
        answer: The turn's delivered answer (empty for silent turns).
        ended_by_terminal_call: Whether the turn ended by a terminal call.
        terminal_tool_name: The honoured terminal tool name, when applicable.
        conversation: The normalized conversation messages, used only to
            classify interface ``ack_silently`` endings as legitimate absorbs
            (tool-response or scheduler wake, or unanswered background work).

    Returns:
        The verdict for the ending.
    """
    delivered = (answer or "").strip()
    if ended_by_terminal_call and terminal_tool_name == SAY_TOOL_NAME and delivered:
        # say(text): delivery for interface turns; the report itself for
        # interface-less turns (a hallucinated say passes its text through
        # as the report — benign, per the architecture pass §7).
        return TurnEndVerdict("ok", None)
    if not ended_by_terminal_call and delivered:
        if interface is not None:
            return TurnEndVerdict("degraded_visible", "no_terminal_call_text")
        return TurnEndVerdict("ok", None)
    if ended_by_terminal_call and terminal_tool_name == ACK_SILENTLY_TOOL_NAME:
        if interface is None:
            # Class-aware ending rule: silence has no receiver one floor
            # down; an ack there is a hallucinated terminal call.
            return TurnEndVerdict("misuse", "hallucinated_ack")
        wake_type = detect_wake_type(conversation)
        if wake_type in (WAKE_TOOL_RESPONSE, WAKE_SCHEDULER_TRIGGER):
            return TurnEndVerdict("ok", None)
        # wake_type is None (unrecognized prompt) or a plain user message:
        # the silence is only legitimate when background work is unanswered.
        if background_work_pending(conversation):
            return TurnEndVerdict("ok", None)
        return TurnEndVerdict("misuse", "overeager_ack")
    # (b) nothing delivered: empty plain-text ending or an empty say().
    return TurnEndVerdict("misuse", "empty_turn")


def honored_terminal_name(terminal_calls: list[ToolCallSchema]) -> str:
    """Return the terminal tool name a terminal turn actually honours.

    ``plan_terminal_turn`` honours exactly one ``say`` when one is present,
    regardless of position; otherwise the turn is an ``ack_silently``.

    Args:
        terminal_calls: The terminal calls parsed from the response.

    Returns:
        The honoured terminal tool name.
    """
    return SAY_TOOL_NAME if _first_say_call(terminal_calls) else ACK_SILENTLY_TOOL_NAME


def repair_nudge_text(misuse: str, interface: bool) -> str:
    """Build the JSON wake prompt for the single bounded repair round-trip.

    The nudge follows the house wake-prompt shape (a JSON user message with a
    ``type``), so it persists and re-reads exactly like every other system
    wake. It never instructs the model to call a terminal tool by name on the
    interface-less path — there the only legal ending is the plain-text
    report.

    Args:
        misuse: The validator's misuse kind.
        interface: Whether the turn carries a user interface.

    Returns:
        The nudge content string.
    """
    if interface:
        desc = (
            "Your previous response reached the user as NOTHING (it ended silently or delivered an empty "
            "message). If the user is owed an answer, deliver it now via the say tool with the full text. "
            "Do not end the turn silently again."
        )
    else:
        desc = (
            "Your previous response ended without the required text report (it was empty or ended with no "
            "output). Reply NOW with your final report as plain text in this message. Do not call any tool "
            "to end the turn."
        )
    return json.dumps({"type": REPAIR_NUDGE_TYPE, "desc": desc})


def log_turn_misuse(
    *,
    model: str,
    provider_name: str,
    interface: UserInterface | None,
    kind: str | None,
    action: str,
) -> None:
    """Emit one greppable misuse-instrumentation line (the counter sink).

    The documented counting mechanism: every event is a single log line
    starting with the fixed token ``TURN_MISUSE``; per-model and per-role
    counters are `grep 'TURN_MISUSE' | grep 'model=<m>' | grep 'role=<r>' |
    grep 'action=<a>' | wc -l`. Format:
    ``TURN_MISUSE model=<m> role=<master|delegated> kind=<k|-> action=<a> provider=<p>``
    where action is one of ``degraded_visible`` (info), ``repair`` /
    ``recovered`` (warning) or ``loud_failure`` (error). The role dimension
    reuses the D1 interface-presence signal, not a second role notion.

    Args:
        model: The model that produced the turn.
        provider_name: The provider serving the model.
        interface: The interface carried by the request, or None.
        kind: The misuse kind, or None for recovery events.
        action: One of ``degraded_visible``, ``repair``, ``recovered``,
            ``loud_failure``.
    """
    role = "master" if interface is not None else "delegated"
    line = f"{_MISUSE_GREP_TOKEN} model={model} role={role} kind={kind or '-'} action={action} provider={provider_name}"
    if action == "loud_failure":
        logger.error(line)
    elif action in ("repair", "recovered"):
        logger.warning(line)
    else:
        logger.info(line)


async def resolve_turn_end_misuse(
    *,
    verdict: TurnEndVerdict,
    model: str,
    provider_name: str,
    interface: UserInterface | None,
    repair_used: bool,
    append_nudge: Callable[[str], None],
    reissue: Callable[[bool], Awaitable[tuple[ChatResponseSchema, T]]],
) -> tuple[ChatResponseSchema, T]:
    """Act on a misuse verdict: one bounded repair round-trip, else a loud failure.

    The hard cap: at most one repair per turn (threaded by the caller through
    ``repair_used``). A repair attempt is another model call inside the loop,
    so it carries the same loop-trap risk recon C7 reproduced — the cap is
    what makes it safe. A repair never creates a tool result for a terminal
    call: it only appends a nudge user message and re-enters the same loop.

    Args:
        verdict: The misuse verdict from ``evaluate_turn_end``.
        model: The model that produced the turn.
        provider_name: The provider serving the model.
        interface: The interface carried by the request, or None.
        repair_used: Whether the single repair round-trip was already spent.
        append_nudge: Seam callback appending the nudge in the adapter's
            native message format (and to the conversation mirror).
        reissue: Seam callback re-entering the same loop; the boolean is the
            ``repair_used`` value for the continued turn.

    Returns:
        The reissued turn's result when the repair recovered it.

    Raises:
        SilentTurnError: When the repair was already used — the turn then
            takes the loud failure path and never passes silently.
    """
    if not repair_used:
        log_turn_misuse(
            model=model, provider_name=provider_name, interface=interface, kind=verdict.misuse, action="repair"
        )
        append_nudge(repair_nudge_text(verdict.misuse or "empty_turn", interface is not None))
        return await reissue(True)
    log_turn_misuse(
        model=model, provider_name=provider_name, interface=interface, kind=verdict.misuse, action="loud_failure"
    )
    raise SilentTurnError(
        provider=provider_name,
        model=model,
        detail=(
            f"Turn ended in misuse ({verdict.misuse}) and the single repair round-trip did not recover it: "
            "the turn still delivered nothing. Failing loudly instead of passing silently."
        ),
    )


def append_repair_nudge(
    *,
    nudge_text: str,
    messages: list[Any],
    conversation_messages: list[Message] | None,
    to_wire: Callable[[str], Any],
) -> None:
    """Append the repair nudge to the wire dialog and the conversation mirror.

    On the Responses path the wire list and the conversation mirror are the
    same object, so the identity guard appends exactly once.

    Args:
        nudge_text: The nudge content string.
        messages: The adapter's native (wire) dialog list.
        conversation_messages: The normalized conversation mirror, or None.
        to_wire: Builds the adapter-native message from the nudge text.
    """
    messages.append(to_wire(nudge_text))
    if conversation_messages is not None and conversation_messages is not messages:
        conversation_messages.append(Message(role="user", content=nudge_text))


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


def forced_choice_applicable(interface: UserInterface | None) -> bool:
    """Decide whether a request should carry a protocol-level forced tool choice.

    This is the single gate behind which every adapter's forced-choice mapping
    sits (one mechanism, not five). Forcing is an optimisation, never
    load-bearing for correctness: the turn-end validator is the guarantee, and
    a backend that ignores the forced choice still produces a correct,
    validated turn.

    Two conditions must hold:

    - the operator switched forcing on (``gpt_settings.force_terminal_tool_choice``;
      default OFF, because forcing is the empirically identified trigger of the
      infinite-ack loop, recon C7, and the task-6 loop-trap regression does not
      exist yet);
    - the request carries a user interface, which per ruling D1 is exactly the
      condition under which the terminal tools ``say``/``ack_silently`` are
      offered. Forcing a call to a tool the request was not offered is the
      hallucination shape the loud-empty rule has to catch.

    Args:
        interface: The user interface carried by the request, or None.

    Returns:
        True when the adapter may attach its forced tool_choice field.
    """
    return gpt_settings.force_terminal_tool_choice and interface is not None


def anthropic_forced_tool_choice(
    interface: UserInterface | None,
    thinking: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the Anthropic agent-loop forced tool_choice, or None when unsafe.

    Uses ``{"type": "any"}`` (recon C1, VERIFIED live) rather than the named
    form ``{"type": "tool", "name": ...}``: the turn may legitimately end via
    either terminal tool (and may still need ordinary work calls first), so
    naming one tool would both suppress legitimate ``ack_silently`` turns and
    force premature termination.

    Hard-400 guard (recon C5, VERIFIED live both directions): legacy
    ``thinking.type = "enabled"`` combined with a forced choice is rejected by
    the API ("Thinking may not be enabled when tool_choice forces tool use"),
    while adaptive thinking plus a forced choice is accepted. The repo sends
    no thinking parameter today; if a future change adds one, this guard drops
    the forced choice for legacy-enabled thinking instead of shipping the 400.

    Args:
        interface: The user interface carried by the request, or None.
        thinking: The request's thinking configuration, when the adapter sends
            one (None today; see the no-thinking pin test).

    Returns:
        The forced tool_choice dict, or None when forcing must not be applied.
    """
    if not forced_choice_applicable(interface):
        return None
    if thinking is not None and thinking.get("type") == "enabled":
        logger.warning(
            "Dropping the forced tool choice: legacy thinking.type='enabled' combined with a forced "
            "choice is a hard 400 ('Thinking may not be enabled when tool_choice forces tool use', "
            "recon C5). Use adaptive thinking if forcing is needed."
        )
        return None
    return {"type": "any"}


def ensure_function_parameters(
    definitions: Sequence[ChatCompletionToolParam],
) -> list[ChatCompletionToolParam]:
    """Guarantee every OpenAI-format tool definition carries a ``parameters`` field.

    The Mistral SDK rejects a function tool without ``parameters`` client-side
    before any request is sent, even when the schema is empty (recon C4, the
    exact validation error is quoted in the recon report). The registry's
    definitions all carry ``parameters`` today; this defensive normalisation
    works on copies so the shared registry dicts are never mutated.

    Args:
        definitions: Tool definitions in the OpenAI chat-completions format.

    Returns:
        Definitions with a guaranteed ``parameters`` object on every function.
    """
    normalized: list[ChatCompletionToolParam] = []
    for definition in definitions:
        function = definition.get("function", {})
        if function.get("parameters") is None:
            definition = cast(
                ChatCompletionToolParam,
                {**definition, "function": {**function, "parameters": dict(EMPTY_OBJECT_SCHEMA)}},
            )
        normalized.append(definition)
    return normalized


def log_forced_choice_outcome(
    *,
    model: str,
    provider_name: str,
    interface: UserInterface | None,
    outcome: Literal["honoured", "ignored"],
) -> None:
    """Emit one greppable forced-choice outcome line (the same sink as TURN_MISUSE).

    Counting mechanism identical to the misuse counters: a single log line per
    event, ``grep 'FORCED_CHOICE' | grep 'model=<m>' | grep 'role=<r>' |
    grep 'outcome=<o>' | wc -l``. ``honoured``: a forced request ended its
    turn through a terminal call; ``ignored``: the backend answered a forced
    request with prose and no tool call at all (the validator still handles
    that ending — forcing is only an optimisation). No new metrics service,
    no new storage.

    Args:
        model: The model that produced the turn.
        provider_name: The provider serving the model.
        interface: The interface carried by the request, or None.
        outcome: Whether the backend honoured or ignored the forced choice.
    """
    role = "master" if interface is not None else "delegated"
    logger.info(f"{_FORCED_CHOICE_GREP_TOKEN} model={model} role={role} outcome={outcome} provider={provider_name}")
