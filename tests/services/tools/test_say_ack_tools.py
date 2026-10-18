"""Tests for the say()/ack_silently() terminal tools and the migrated prompt contract."""

from typing import Any, cast

import pytest

# The tools package is imported before chibi.constants on purpose: it pulls in
# chibi.config first, which fully initializes chibi.constants. If chibi.constants
# were the first chibi import instead, its own `from chibi.config import ...`
# would re-enter the partially initialized chibi.config and fail — which is
# exactly what happens when this module is collected standalone
# (`pytest tests/services/tools/test_say_ack_tools.py` with no earlier imports).
import chibi.services.providers.tools  # noqa: F401  (side effect: primes chibi.config)
from chibi.constants import SCHEDULER_HINT_PROMPT, get_llm_prompt
from chibi.services.providers.tools import RegisteredChibiTools
from chibi.services.providers.tools.common import AckSilentlyTool, SayTool


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
