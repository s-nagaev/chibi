"""Tests for the say()/ack_silently() terminal tools and the migrated prompt contract."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

# The tools package is imported before chibi.constants on purpose: it pulls in
# chibi.config first, which fully initializes chibi.constants. If chibi.constants
# were the first chibi import instead, its own `from chibi.config import ...`
# would re-enter the partially initialized chibi.config and fail — which is
# exactly what happens when this module is collected standalone
# (`pytest tests/services/tools/test_say_ack_tools.py` with no earlier imports).
import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.constants import DELEGATION_RULES, SCHEDULER_HINT_PROMPT, get_llm_prompt
from chibi.schemas.app import ChatResponseSchema, ModelChangeSchema
from chibi.services.providers.terminal_calls import TERMINAL_TOOL_NAMES
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.common import AckSilentlyTool, DelegateTool, SayTool
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.schemas import ToolResponseSchema


class TestTerminalToolRegistration:
    """Both terminal tools must be registered through the existing tool registry."""

    def test_both_terminal_tools_are_registered(self) -> None:
        assert RegisteredChibiTools.tools_map["say"] is SayTool
        assert RegisteredChibiTools.tools_map["ack_silently"] is AckSilentlyTool

    def test_definitions_reach_the_model_via_the_registry(self) -> None:
        names = {definition["function"]["name"] for definition in RegisteredChibiTools.get_tool_definitions()}
        assert {"say", "ack_silently"} <= names


class TestTerminalToolSchemas:
    """Schemas must be strict: type object with an explicit properties object."""

    def test_say_schema_is_object_with_explicit_properties(self) -> None:
        params = SayTool.definition["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert params["properties"]["text"]["type"] == "string"
        assert params["required"] == ["text"]

    def test_ack_silently_schema_is_object_with_explicit_empty_properties(self) -> None:
        params = AckSilentlyTool.definition["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params["required"] == []

    def test_terminal_tools_do_not_expose_the_background_switch(self) -> None:
        for tool in (SayTool, AckSilentlyTool):
            params = cast(dict[str, Any], tool.definition["function"]["parameters"])
            assert "run_in_background" not in params["properties"]


class TestTerminalToolContractDescriptions:
    """The descriptions carry the contract, since they are the model's only guidance."""

    def test_say_description_states_user_only_channel_and_terminal_semantics(self) -> None:
        description = SayTool.definition["function"]["description"]
        assert "ONLY channel" in description
        assert "ENDS" in description

    def test_ack_silently_description_states_invisible_service_response_and_terminal_semantics(self) -> None:
        description = AckSilentlyTool.definition["function"]["description"]
        assert "NEVER sees" in description
        assert "ENDS" in description

    def test_ack_silently_description_states_the_answer_owed_admission_rule(self) -> None:
        description = AckSilentlyTool.definition["function"]["description"]
        assert "no longer owed" in description
        assert "REFUSED" in description


class TestTerminalToolsHaveNoResultPath:
    """Neither tool may carry an implementation that produces a tool result.

    The loop-exit intercept belongs to sub-task 4b; until then the base-class
    function body (NotImplementedError) is the only behaviour, so the dispatch
    plumbing cannot silently manufacture a successful result for a terminal
    call.
    """

    @pytest.mark.asyncio
    async def test_say_function_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await SayTool.function()

    @pytest.mark.asyncio
    async def test_ack_silently_function_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            await AckSilentlyTool.function()


class TestMigratedPromptContract:
    """The system prompt must teach the tool channel, never the free-text sentinel."""

    def test_prompt_presents_the_two_tool_channel_not_the_sentinel(self) -> None:
        prompt = get_llm_prompt(filesystem_access=False, allow_delegation=False)
        assert "<chibi>ACK</chibi>" not in prompt
        assert 'THE "ACK" RULE' not in prompt
        assert "say(text)" in prompt
        assert "ack_silently()" in prompt

    def test_prompt_tells_the_two_tools_apart(self) -> None:
        prompt = get_llm_prompt(filesystem_access=True, allow_delegation=True)
        delivery_block = prompt.split("DELIVERY AND SILENCE")[1].split("# Guiding Principles")[0]
        assert "ONLY channel for user-facing communication" in delivery_block
        assert "NEVER sees" in delivery_block
        assert "MUST say() it" in delivery_block
        assert "NEVER ack_silently() in its place" in delivery_block

    def test_scheduler_hint_points_at_ack_silently_not_the_sentinel(self) -> None:
        assert "ack_silently()" in SCHEDULER_HINT_PROMPT
        assert "<chibi>ACK</chibi>" not in SCHEDULER_HINT_PROMPT
        assert "else ACK" not in SCHEDULER_HINT_PROMPT


async def _fake_models_available(user_id: int, **kwargs: Any) -> list[ModelChangeSchema]:
    return [ModelChangeSchema(provider="prov", name="gpt-x", display_name="GPT-X", image_generation=False)]


def _subagent_response(answer: str) -> ChatResponseSchema:
    return ChatResponseSchema(answer=answer, provider="prov", model="gpt-x", usage=None)


class TestLoudEmptyDelegationContract:
    """An empty sub-agent answer is a loud error tool result to the parent, never success.

    Structural context (ruling D1): terminal interception in the loops is by
    tool-name membership and never consults the offered list, so a
    hallucinated ``ack_silently()`` from an interface-less sub-agent still ends
    its turn with an empty answer. ``DelegateTool`` is the sole caller of
    ``get_sub_agent_response`` and the background resumption path funnels
    through it, so this single rule closes the swallow for both delegation
    modes. Raising ``ToolException`` here is what turns the empty answer into
    an error tool result (``ChibiTool._get_tool_call_result`` catches it into
    ``ToolResponseSchema(status="error")``) — no tool result is ever created
    for the terminal call itself.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_answer", ["", "   \n\t "])
    async def test_empty_sub_agent_answer_raises_a_loud_error(self, empty_answer: str) -> None:
        with (
            patch(
                "chibi.services.providers.tools.common.get_sub_agent_response",
                new=AsyncMock(return_value=_subagent_response(empty_answer)),
            ),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", _fake_models_available),
        ):
            with pytest.raises(ToolException, match="EMPTY report"):
                await DelegateTool.function(
                    prompt="do something",
                    model_name="gpt-x",
                    provider_name="prov",
                    user_id=1,
                    caller_model="caller-model",
                    caller_provider="caller-provider",
                    caller_storage_id=1,
                    caller_thread_id=42,
                )

    @pytest.mark.asyncio
    async def test_normal_report_still_returns_the_response_dict(self) -> None:
        with (
            patch(
                "chibi.services.providers.tools.common.get_sub_agent_response",
                new=AsyncMock(return_value=_subagent_response("the actual report")),
            ),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", _fake_models_available),
        ):
            result = await DelegateTool.function(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                user_id=1,
                caller_model="caller-model",
                caller_provider="caller-provider",
                caller_storage_id=1,
                caller_thread_id=42,
            )

        assert result == {"response": "the actual report"}

    @pytest.mark.asyncio
    async def test_empty_report_reaches_the_parent_as_an_error_tool_result(self) -> None:
        """The parent-visible shape: dispatch converts the raise into status='error'."""
        with (
            patch(
                "chibi.services.providers.tools.common.get_sub_agent_response",
                new=AsyncMock(return_value=_subagent_response("")),
            ),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", _fake_models_available),
        ):
            result: ToolResponseSchema = await DelegateTool.tool(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                user_id=1,
                caller_model="caller-model",
                caller_provider="caller-provider",
                caller_storage_id=1,
                caller_thread_id=0,
            )

        assert result.status == "error"
        assert "EMPTY report" in str(result.result)


class TestDelegationRulesPromptContract:
    """DELEGATION_RULES must describe the post-ruling mechanism, not the old guard."""

    def test_stale_silence_claim_is_gone(self) -> None:
        """The old claim was true via the delivery guard the new say() bypassed."""
        assert "will not be able to say or send anything to the user" not in DELEGATION_RULES

    def test_rules_describe_the_actual_mechanism(self) -> None:
        assert "no user channel" in DELEGATION_RULES
        assert "say/ack_silently" in DELEGATION_RULES
        assert "delegate_task result" in DELEGATION_RULES
        assert "via say()" in DELEGATION_RULES

    def test_no_prompt_constant_tells_an_interface_less_actor_to_be_silent_via_a_terminal_tool(self) -> None:
        """The scheduler hint is the only silence instruction left; it is interface-gated.

        The hint text itself is untouched (approved in 4a); the gate in
        ``prepare_system_prompt`` (interface is not None) is what keeps it away
        from interface-less actors, covered behaviourally in
        ``tests/services/providers/test_utils.py``.
        """
        assert "ack_silently()" in SCHEDULER_HINT_PROMPT
        assert TERMINAL_TOOL_NAMES == {"say", "ack_silently"}


class TestLoudEmptyDelegationBackgroundVariant:
    """The loud-empty rule holds on the background delegation path too.

    ``DelegateTool`` defaults to background execution (``run_in_background_by_default``
    with a user interface): ``ChibiTool.tool`` schedules the dispatch coroutine
    through ``task_manager``, and the coroutine funnels through the same
    ``DelegateTool.function`` boundary before its result is handed to
    ``handle_tool_response`` for the wake. This test captures the scheduled
    coroutine and awaits it directly, so the background resumption variant is
    covered without live services.
    """

    @pytest.mark.asyncio
    async def test_background_delegation_reports_an_empty_result_as_an_error_tool_response(self) -> None:
        captured: dict[str, Any] = {}

        def _capture_run_task(coro: Any, **kwargs: Any) -> None:
            captured["coro"] = coro

        interface = SimpleNamespace(
            storage_id=1, thread_id=1, streaming_enabled=False, delta_emitted=False, captures_llm_thoughts=False
        )

        with (
            patch(
                "chibi.services.providers.tools.common.get_sub_agent_response",
                new=AsyncMock(return_value=_subagent_response("")),
            ),
            patch("chibi.services.providers.tools.common.get_models_available_to_user", _fake_models_available),
            patch("chibi.services.providers.tools.tool.task_manager") as task_manager_mock,
            patch("chibi.services.bot.handle_tool_response", new=AsyncMock()) as handle_mock,
        ):
            task_manager_mock.run_task.side_effect = _capture_run_task
            acknowledgement = await DelegateTool.tool(
                prompt="do something",
                model_name="gpt-x",
                provider_name="prov",
                user_id=1,
                caller_model="caller-model",
                caller_provider="caller-provider",
                caller_storage_id=1,
                caller_thread_id=42,
                interface=cast(Any, interface),
            )

            # The dispatch went to the background scheduler, not the
            # foreground path.
            assert acknowledgement.status == "tool is running in background"
            assert "coro" in captured

            # The background coroutine resumes (still inside the patched
            # seams): the loud rule converts the empty report into an error
            # tool response for the wake.
            await captured["coro"]

        handle_mock.assert_awaited_once()
        assert handle_mock.await_args is not None
        delivered = handle_mock.await_args.kwargs["tool_response"]
        assert delivered.status == "error"
        assert "EMPTY report" in str(delivered.result)
