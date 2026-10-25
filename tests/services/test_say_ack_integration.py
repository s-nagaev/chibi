"""Cross-cutting integration layer for the say()/ack_silently() rule (task 6).

The per-task minimal tests live in their own modules (AGENTS.md §3.9). This
module closes the specific cross-cutting gaps those tasks routed to task 6:

- the sub-agent tool-exclusion regression driven through the REAL sub-agent
  flow (``get_sub_agent_response`` → failover → provider loop), asserting the
  INVARIANT (the interface-less list is the global list minus exactly
  ``{say, ack_silently}``, ``delegate_task`` remains offered) — never a
  hardcoded tool count, which is environment-dependent (video tools
  deregister by process config);
- the master-side control through the REAL master entry point
  (``get_llm_chat_completion_answer`` with an interface): both terminal tools
  are present, and the offered list equals the full global list;
- the scheduler-hint regression at the integration layer: the system payload
  that reaches a sub-agent's completion call contains no ``ack_silently``
  instruction even when the scheduler tool gate would allow the hint;
- the chained end-to-end hallucination flow: an interface-less sub-agent
  hallucinates ``ack_silently()`` → offer-blind empty ending → exactly one
  repair → loud ``SilentTurnError`` → the parent receives an error tool
  result, never a successful ``{"response": ""}``;
- the prompt-content tripwire: the assembled master prompt carries the
  reconciled ``DELEGATION_RULES`` sentence verbatim;
- the summarizer exclusion (task 5 PM ruling): a hidden message never reaches
  the emergency summarizer input.

Gemini fixture discipline: this module replays ZERO Gemini function-call
parts (the sub-agent fixtures ride the OpenAI chat-completions seam via
MoonshotAI), so the ``thought_signature`` constraint is satisfied trivially
here; see the report for where the constraint was checked.
"""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.constants import DELEGATION_RULES, SCHEDULER_HINT_PROMPT, get_llm_prompt
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema
from chibi.services.interface import UserInterface
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.common import DelegateTool
from chibi.services.providers.tools.utils import get_sub_agent_response
from chibi.services.user import emergency_summarization, get_llm_chat_completion_answer
from chibi.storage.database import _db_provider

TEST_TOKEN = "test-token-for-ack-integration"
TEST_MODEL = "kimi-k2.6"

INTERFACE_STUB = SimpleNamespace(
    storage_id=1,
    thread_id=9,
    user_data={"first_name": "Tester"},
    editor_context=None,
    uses_uploaded_file_storage=False,
    streaming_enabled=False,
    delta_emitted=False,
    captures_llm_thoughts=False,
    send_message=AsyncMock(),
    send_llm_thoughts=AsyncMock(),
)


def _completion_with_answer(answer: str) -> ChatCompletion:
    choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=answer, tool_calls=None),
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _completion_with_tool_calls(tool_calls: list[ChatCompletionMessageToolCall], content: str = "") -> ChatCompletion:
    choice = Choice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls),
        finish_reason="tool_calls",
    )
    return ChatCompletion(
        id="test-id",
        choices=[choice],
        created=1234567890,
        model=TEST_MODEL,
        object="chat.completion",
    )


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def _cc_client(create_mock: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))


def _stub_user(provider: Any) -> Mock:
    """A user double wiring one real provider instance into every registry seam."""
    user = Mock()
    user.id = 1
    user.providers = Mock()
    user.providers.get = Mock(return_value=provider)
    user.providers.chat_ready = {}
    user.providers.available_instances = []
    user.stt_provider = None
    user.get_effective_working_dir = Mock(return_value="/tmp/ack-integration")
    user.get_active_llm_provider = Mock(return_value=provider)
    user.get_active_llm_model = Mock(return_value=TEST_MODEL)
    user.get_available_models = AsyncMock(return_value=[])
    user.llm_skills = []
    user.info = "test user"
    user.thread_notes = {}
    return user


def _stub_db(user: Any, conversation: list[Message] | None = None) -> AsyncMock:
    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=conversation or [])
    return db


def _global_tool_names() -> set[str]:
    return {definition["function"]["name"] for definition in RegisteredChibiTools.get_tool_definitions()}


def _offered_names(create_mock: AsyncMock) -> set[str]:
    tools = create_mock.await_args.kwargs["tools"]  # type: ignore[union-attr]
    return {definition["function"]["name"] for definition in tools}


def _prepare_prompt_patches(user: Any, scheduler_gate_on: bool = True):
    """Real ``prepare_system_prompt`` with only its environment edges patched.

    The scheduler gate is pinned ON so the sub-agent assertion below proves
    the hint is withheld by the interface precondition itself, not by a
    disabled gate.
    """
    return (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_available_skills", return_value=[]),
        patch(
            "chibi.services.providers.tools.scheduler.application_settings",
            SimpleNamespace(scheduler_tool_enabled=scheduler_gate_on, client="telegram"),
        ),
    )


def _no_retry_patch():
    """Bypass the tenacity decorator on the shared CC ``get_chat_response``.

    The shared chat-completions ``get_chat_response`` (``provider.py``, the
    class MoonshotAI inherits) is wrapped in ``@retry(stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=20, min=30, max=180), reraise=True)``
    with NO ``retry=`` condition, so it retries EVERY exception — including
    the loud ``SilentTurnError`` — four times with 30s+ sleeps. The silence
    contract is what these tests pin; the retry policy is pre-existing
    transport behaviour, so the tests call the undecorated function
    (``__wrapped__``) directly. The production implication of that policy
    (a loud silent turn is retried before it surfaces) is noted in the task
    report as an observation, out of scope here.
    """
    undecorated = getattr(MoonshotAI.get_chat_response, "__wrapped__", None)
    return patch.object(MoonshotAI, "get_chat_response", undecorated or MoonshotAI.get_chat_response)


class TestSubAgentToolExclusionIntegration:
    """The sub-agent exclusion invariant, driven through the real sub-agent flow.

    ``get_sub_agent_response`` resolves the provider from the user registry,
    walks the failover engine and reaches the real provider loop, so the
    captured completion kwargs are the actual request payload the sub-agent's
    LLM would see. The invariant (ruling D1, role-scoping review wording):
    the interface-less list is the global list minus exactly
    ``{say, ack_silently}``, and ``delegate_task`` remains offered. The
    absolute count is environment-dependent and is never asserted.
    """

    @pytest.mark.asyncio
    async def test_sub_agent_payload_excludes_terminal_tools_and_keeps_delegate_task(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("plain report"))
        user = _stub_user(provider)
        db = _stub_db(user)

        patches = (
            patch.object(_db_provider, "get_database", AsyncMock(return_value=db)),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=_cc_client(create_mock)),
            _no_retry_patch(),
            *_prepare_prompt_patches(user),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            response = await get_sub_agent_response(
                user_id=1,
                prompt="do things",
                model_name=TEST_MODEL,
                provider_name="MoonshotAI",
                caller_storage_id=1,
                caller_thread_id=9,
            )

        assert response.answer == "plain report"
        offered = _offered_names(create_mock)
        # Neither terminal tool is offered...
        assert "say" not in offered
        assert "ack_silently" not in offered
        # ...and the delegation tool survives.
        assert "delegate_task" in offered
        # The INVARIANT, count-free: interface-less list == global list minus
        # exactly the two terminal tools. Both directions are checked, so a
        # config-driven registration change can never satisfy a stale count.
        assert offered == _global_tool_names() - {"say", "ack_silently"}

    @pytest.mark.asyncio
    async def test_sub_agent_system_payload_carries_no_ack_instruction(self) -> None:
        """The scheduler hint never reaches an interface-less actor (architecture pass §9).

        The gate itself is the role-scoping change in
        ``chibi/services/providers/utils.py`` (``prepare_system_prompt``, the
        ``interface is not None`` precondition with the
        ``_scheduler_hint_applicable`` predicate); the process-level
        tool-registration gate at ``tools/scheduler.py`` (``_scheduler_tools_register``)
        is a different predicate that this test pins ON on purpose.
        """
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("plain report"))
        user = _stub_user(provider)
        db = _stub_db(user)

        patches = (
            patch.object(_db_provider, "get_database", AsyncMock(return_value=db)),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=_cc_client(create_mock)),
            _no_retry_patch(),
            *_prepare_prompt_patches(user, scheduler_gate_on=True),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            await get_sub_agent_response(
                user_id=1,
                prompt="do things",
                model_name=TEST_MODEL,
                provider_name="MoonshotAI",
                caller_storage_id=1,
                caller_thread_id=9,
            )

        dialog = create_mock.await_args.kwargs["messages"]  # type: ignore[union-attr]
        assert dialog[0]["role"] == "system"
        payload = json.loads(dialog[0]["content"])
        # The raw JSON string and the parsed prompt block are both checked:
        # no ack instruction anywhere in what the sub-agent actually sees.
        assert "ack_silently" not in dialog[0]["content"]
        assert "ack_silently" not in payload["system_prompt"]
        assert SCHEDULER_HINT_PROMPT not in payload["system_prompt"]


class TestMasterControlIntegration:
    """Control: the master entry point with an interface keeps both terminal tools."""

    @pytest.mark.asyncio
    async def test_interface_turn_is_offered_both_terminal_tools_and_everything_else(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        create_mock = AsyncMock(return_value=_completion_with_answer("master answer"))
        user = _stub_user(provider)
        db = _stub_db(user)
        interface = cast(UserInterface, INTERFACE_STUB)

        patches = (
            patch.object(_db_provider, "get_database", AsyncMock(return_value=db)),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=_cc_client(create_mock)),
            _no_retry_patch(),
            *_prepare_prompt_patches(user),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            response = await get_llm_chat_completion_answer(storage_id=1, interface=interface, user_text_message="hi")

        assert response.answer == "master answer"
        offered = _offered_names(create_mock)
        assert {"say", "ack_silently"} <= offered
        # Count-free invariant, the complement direction: an interface request
        # keeps the FULL global list, unfiltered.
        assert offered == _global_tool_names()


class TestChainedHallucinationEndToEnd:
    """The chained e2e flow the role-scoping review routed here.

    One flow: an interface-less sub-agent hallucinates ``ack_silently()`` →
    the offer-blind ending leaves the turn empty → the validator spends its
    single repair round-trip → loud ``SilentTurnError`` → the empty answer is
    converted into an error tool result for the parent. The parent never sees
    a successful ``{"response": ""}``. The two existing part-tests
    (``test_hallucinated_ack_costs_a_repair_then_fails_loudly`` and the
    loud-empty delegation tests) are kept unchanged; this is additional.
    """

    @pytest.mark.asyncio
    async def test_hallucinated_sub_agent_ack_reaches_the_parent_as_a_loud_error_tool_result(self) -> None:
        provider = MoonshotAI(token=TEST_TOKEN)
        ack_response = _completion_with_tool_calls([_tool_call("ack_silently", "{}")], content="")
        create_mock = AsyncMock(side_effect=[ack_response, ack_response, AssertionError("repair cap exceeded")])
        user = _stub_user(provider)
        db = _stub_db(user)

        registry_call_mock = AsyncMock(side_effect=AssertionError("A terminal call reached the dispatcher"))
        patches = (
            patch.object(_db_provider, "get_database", AsyncMock(return_value=db)),
            patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=_cc_client(create_mock)),
            _no_retry_patch(),
            *_prepare_prompt_patches(user),
            patch.object(RegisteredChibiTools, "call", registry_call_mock),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            result = await DelegateTool.tool(
                prompt="compare the two images",
                caller_model="caller-model",
                caller_provider="MoonshotAI",
                user_id=1,
                caller_storage_id=1,
                caller_thread_id=9,
            )

        # The sub-agent turn: offer-blind (ack_silently was never offered),
        # ended empty, cost exactly one repair, then failed loudly.
        assert create_mock.await_count == 2
        assert "ack_silently" not in _offered_names(create_mock)
        # The parent-visible shape: an ERROR tool result carrying the loud
        # failure text — never a successful {"response": ""}.
        assert result.status == "error"
        assert isinstance(result.result, str)
        assert "hallucinated_ack" in result.result
        assert "Failing loudly" in result.result
        # The terminal call itself never reached the dispatcher: the only
        # tool result in this flow is the error result for delegate_task.
        registry_call_mock.assert_not_awaited()


class TestDelegationRulesIntegrationTripwire:
    """The assembled master prompt carries the reconciled DELEGATION_RULES sentence.

    Unit-level tripwires pin the fragments of the constant; this is the single
    integration-layer restatement the architecture pass asks for: the prompt
    actually assembled for a delegation-enabled master turn contains the
    post-ruling mechanism sentence verbatim (whitespace-normalized).
    """

    def test_assembled_prompt_carries_the_reconciled_delegation_sentence(self) -> None:
        prompt = get_llm_prompt(filesystem_access=False, allow_delegation=True)
        normalized = " ".join(prompt.split())
        assert (
            "Reason: sub-agents have no user channel. The say/ack_silently delivery tools are not offered to "
            "a request without a user interface, and nothing a sub-agent writes outside its final report "
            "reaches the user. The sub-agent's report is returned to you as the delegate_task result; YOU "
            "deliver it to the user via say()." in normalized
        )

    def test_reconciled_sentence_source_is_the_delegation_rules_constant(self) -> None:
        # Guard the tripwire's own anchor: the sentence lives in DELEGATION_RULES
        # (constants.py), so a relocation of the text fails loudly here too.
        normalized = " ".join(DELEGATION_RULES.split())
        assert "The sub-agent's report is returned to you as the delegate_task result" in normalized


class TestSummarizerExcludesHiddenMessages:
    """A hidden message never reaches the emergency summarizer input (task 5 PM ruling).

    Consequence (existing summarization behaviour, not a retention change):
    the summary replaces the WHOLE history afterwards, so the hidden message
    leaves the conversation entirely once summarization runs; nothing is
    dropped from storage here.
    """

    @pytest.mark.asyncio
    async def test_hidden_message_does_not_reach_the_summarizer_input(self) -> None:
        visible = Message(role="user", content="the visible question about cycle routes")
        hidden = Message(role="assistant", content="SECRET private CoT about the cycle routes", visibility="hidden")
        user = _stub_user(provider=Mock())
        db = _stub_db(user, conversation=[visible, hidden])
        summarizer_provider = Mock()
        summarizer_provider.get_chat_response = AsyncMock(
            return_value=(ChatResponseSchema(answer="the summary", provider="prov", model="m", usage=None), [])
        )
        user.get_active_llm_provider = Mock(return_value=summarizer_provider)
        user.get_active_llm_model = Mock(return_value="m")

        with patch.object(_db_provider, "get_database", AsyncMock(return_value=db)):
            await emergency_summarization(storage_id=1, thread_id=9)

        input_kwargs = summarizer_provider.get_chat_response.await_args.kwargs
        summarizer_input = input_kwargs["messages"][0].content
        assert "the visible question about cycle routes" in summarizer_input
        assert "SECRET private CoT" not in summarizer_input

        # Summary-replaces-history consequence: the whole history is dropped
        # (``reset_chat_history`` delegates to ``db.drop_messages``) and
        # rebuilt from the summary alone — the hidden content appears in
        # neither persisted message.
        db.drop_messages.assert_awaited_once()
        assert db.add_message.await_count == 2
        for call in db.add_message.await_args_list:
            assert "SECRET private CoT" not in call.kwargs["message"].content
