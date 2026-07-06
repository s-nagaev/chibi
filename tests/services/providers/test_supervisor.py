"""Unit tests for the Supervisor feature.

Covers SupervisorAnswer validation, config fallback chain, tool-call hook
(all 5 call sites), final-answer hook, fail-open behavior, and
supervisor_enabled=False gating.
"""

from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from pydantic import ValidationError

from chibi.config import gpt_settings
from chibi.models import Message, User
from chibi.schemas.app import (
    ChatResponseSchema,
    SupervisorAnswer,
    SupervisorCategory,
    SupervisorVerdict,
)
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import (
    Provider,
    RegisteredProviders,
)
from chibi.services.providers.tools.schemas import ToolCallSchema, ToolResponseSchema

TEST_TOKEN = "test-token"
TEST_USER = User(id=12345, name="test_user")


# ==============================================================================
# Helpers
# ==============================================================================


def _make_ok_answer() -> SupervisorAnswer:
    """Return a SupervisorAnswer with an OK verdict."""
    return SupervisorAnswer(verdict=SupervisorVerdict.OK)


def _make_intervene_answer() -> SupervisorAnswer:
    """Return a SupervisorAnswer with an INTERVENE verdict."""
    return SupervisorAnswer(
        verdict=SupervisorVerdict.INTERVENE,
        category=SupervisorCategory.SCOPE_CREEP,
        reason="action outside executor scope",
    )


def _make_chat_response(answer: str = "final answer") -> ChatResponseSchema:
    """Return a minimal ChatResponseSchema for testing."""
    return ChatResponseSchema(answer=answer, provider="test", model="test-model", usage=None)


def _mock_provider_with_supervise(
    provider_cls: type[Provider], supervise_return: SupervisorAnswer
) -> tuple[Provider, AsyncMock]:
    """Create a provider instance and a mocked supervise().

    The caller MUST activate the mock via
    ``with patch.object(provider, "supervise", mock_supervise):``.

    Returns a tuple of (provider_instance, supervise_mock).
    """
    provider = provider_cls(token=TEST_TOKEN)
    mock_supervise = AsyncMock(return_value=supervise_return)
    return provider, mock_supervise


# ==============================================================================
# 1. SupervisorAnswer Validation
# ==============================================================================


def test_supervisor_answer_ok_without_category_reason():
    """Test that OK verdict without category/reason is valid."""
    answer = SupervisorAnswer(verdict=SupervisorVerdict.OK)
    assert answer.verdict == SupervisorVerdict.OK
    assert answer.category is None
    assert answer.reason is None


def test_supervisor_answer_intervene_with_both_fields():
    """Test that INTERVENE with both category and reason is valid."""
    answer = SupervisorAnswer(
        verdict=SupervisorVerdict.INTERVENE,
        category=SupervisorCategory.PROTOCOL_SKIP,
        reason="skipped required review step",
    )
    assert answer.verdict == SupervisorVerdict.INTERVENE
    assert answer.category == SupervisorCategory.PROTOCOL_SKIP
    assert answer.reason == "skipped required review step"


def test_supervisor_answer_intervene_without_category_raises():
    """Test that INTERVENE without category raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SupervisorAnswer(verdict=SupervisorVerdict.INTERVENE, reason="some reason")
    assert "category and reason are required" in str(exc_info.value)


def test_supervisor_answer_intervene_without_reason_raises():
    """Test that INTERVENE without reason raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SupervisorAnswer(verdict=SupervisorVerdict.INTERVENE, category=SupervisorCategory.ROLE_VIOLATION)
    assert "category and reason are required" in str(exc_info.value)


def test_supervisor_answer_intervene_with_empty_reason_raises():
    """Test that INTERVENE with empty string reason raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SupervisorAnswer(verdict=SupervisorVerdict.INTERVENE, category=SupervisorCategory.SCOPE_CREEP, reason="")
    assert "category and reason are required" in str(exc_info.value)


# ==============================================================================
# 2. Config fallback chain resolution
# ==============================================================================


class TestConfigFallbackChain:
    """Tests for supervisor_provider_resolved / supervisor_model_resolved."""

    def test_supervisor_provider_explicit(self):
        """Test that explicit supervisor_provider is used when set."""
        with patch.object(gpt_settings, "supervisor_provider", "OpenAI"):
            with patch.object(gpt_settings, "moderation_provider", "Anthropic"):
                assert gpt_settings.supervisor_provider_resolved == "OpenAI"

    def test_supervisor_fallback_to_moderation_provider(self):
        """Test fallback to moderation_provider when supervisor_provider is not set."""
        with patch.object(gpt_settings, "supervisor_provider", None):
            with patch.object(gpt_settings, "moderation_provider", "Anthropic"):
                assert gpt_settings.supervisor_provider_resolved == "Anthropic"

    def test_supervisor_provider_none_when_neither_set(self):
        """Test that None is returned when neither is set."""
        with patch.object(gpt_settings, "supervisor_provider", None):
            with patch.object(gpt_settings, "moderation_provider", None):
                assert gpt_settings.supervisor_provider_resolved is None

    def test_supervisor_model_explicit(self):
        """Test that explicit supervisor_model is used when set."""
        with patch.object(gpt_settings, "supervisor_model", "gpt-5-mini"):
            with patch.object(gpt_settings, "moderation_model", "claude-haiku"):
                assert gpt_settings.supervisor_model_resolved == "gpt-5-mini"

    def test_supervisor_model_fallback_to_moderation_model(self):
        """Test fallback to moderation_model when supervisor_model is not set."""
        with patch.object(gpt_settings, "supervisor_model", None):
            with patch.object(gpt_settings, "moderation_model", "claude-haiku"):
                assert gpt_settings.supervisor_model_resolved == "claude-haiku"

    def test_supervisor_model_none_when_neither_set(self):
        """Test that None is returned when neither is set."""
        with patch.object(gpt_settings, "supervisor_model", None):
            with patch.object(gpt_settings, "moderation_model", None):
                assert gpt_settings.supervisor_model_resolved is None


# ==============================================================================
# 3. Tool-call hook — call_functions() supervisor integration
# ==============================================================================


class TestToolCallHookOpenAIFriendly:
    """Tests for supervisor tool-call hook via OpenAIFriendlyProvider.call_functions()."""

    @pytest.mark.asyncio
    async def test_supervisor_enabled_intervene_blocks_tool(self):
        """Test that INTERVENE verdict blocks the tool call."""
        provider, mock_supervise = _mock_provider_with_supervise(OpenAI, _make_intervene_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={"arg1": "val1"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    results = await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].tool_name == "test_tool"
        assert "action outside executor scope" in results[0].result
        mock_supervise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_supervisor_enabled_ok_allows_tool(self):
        """Test that OK verdict allows the tool call."""
        provider, mock_supervise = _mock_provider_with_supervise(OpenAI, _make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={"arg1": "val1"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ) as mock_tool_call:
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                            user_id=TEST_USER.id,
                        )

        assert len(results) == 1
        assert results[0].status == "ok"
        mock_supervise.assert_awaited_once()
        mock_tool_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_supervisor_disabled_skips_supervision(self):
        """Test that when supervisor_enabled=False, supervise() is not called."""
        provider, mock_supervise = _mock_provider_with_supervise(OpenAI, _make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={"arg1": "val1"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", False):
                with patch(
                    "chibi.services.providers.provider.RegisteredChibiTools.call",
                    new_callable=AsyncMock,
                    return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                ):
                    await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        mock_supervise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_supervisor_disabled_tool_called_normally(self):
        """Test that when supervisor is disabled, tools execute normally."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock(return_value=_make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={"arg1": "val1"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", False):
                with patch(
                    "chibi.services.providers.provider.RegisteredChibiTools.call",
                    new_callable=AsyncMock,
                    return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                ) as mock_tool_call:
                    results = await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        assert len(results) == 1
        assert results[0].status == "ok"
        mock_tool_call.assert_awaited_once()
        mock_supervise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parallel_calls_mixed_verdicts(self):
        """Test mixed intervene/ok verdicts for parallel tool calls."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock()
        mock_supervise.side_effect = [
            _make_intervene_answer(),
            _make_ok_answer(),
            _make_intervene_answer(),
        ]

        calls = [
            ToolCallSchema(tool_name="blocked_tool_1", args={}),
            ToolCallSchema(tool_name="allowed_tool", args={}),
            ToolCallSchema(tool_name="blocked_tool_2", args={}),
        ]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="allowed_tool", status="ok", result="done"),
                    ) as mock_tool_call:
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                            user_id=TEST_USER.id,
                        )

        assert len(results) == 3
        # blocked tools should have error status
        assert results[0].status == "error"
        assert results[0].tool_name == "blocked_tool_1"
        # allowed tool should have ok status
        assert results[1].status == "ok"
        assert results[1].tool_name == "allowed_tool"
        # second blocked tool
        assert results[2].status == "error"
        assert results[2].tool_name == "blocked_tool_2"
        # Only one real tool call was made (allowed_tool)
        assert mock_tool_call.call_count == 1


class TestToolCallHookAnthropicFriendly:
    """Supervisor tool-call hook via AnthropicFriendlyProvider.call_functions()."""

    @pytest.mark.asyncio
    async def test_anthropic_intervene_blocks_tool(self):
        """Test INTERVENE via Anthropic-friendly path blocks the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(Anthropic, _make_intervene_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={"arg1": "val1"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    results = await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        assert len(results) == 1
        assert results[0].status == "error"
        mock_supervise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anthropic_ok_allows_tool(self):
        """Test OK via Anthropic-friendly path allows the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(Anthropic, _make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ) as mock_tool_call:
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                        )

        assert len(results) == 1
        assert results[0].status == "ok"
        mock_tool_call.assert_awaited_once()


class TestToolCallHookGemini:
    """Supervisor tool-call hook via Gemini.call_functions()."""

    @pytest.mark.asyncio
    async def test_gemini_intervene_blocks_tool(self):
        """Test INTERVENE via Gemini path blocks the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(Gemini, _make_intervene_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    results = await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        assert len(results) == 1
        assert results[0].status == "error"
        mock_supervise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gemini_ok_allows_tool(self):
        """Test OK via Gemini path allows the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(Gemini, _make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ):
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                        )

        assert len(results) == 1
        assert results[0].status == "ok"


class TestToolCallHookMistral:
    """Supervisor tool-call hook via MistralAI.call_functions()."""

    @pytest.mark.asyncio
    async def test_mistral_intervene_blocks_tool(self):
        """Test INTERVENE via Mistral path blocks the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(MistralAI, _make_intervene_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    results = await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test prompt",
                        user_id=TEST_USER.id,
                    )

        assert len(results) == 1
        assert results[0].status == "error"
        mock_supervise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mistral_ok_allows_tool(self):
        """Test OK via Mistral path allows the tool."""
        provider, mock_supervise = _mock_provider_with_supervise(MistralAI, _make_ok_answer())

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ):
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                        )

        assert len(results) == 1
        assert results[0].status == "ok"


# ==============================================================================
# 4. Fail-open when supervise() raises an exception
# ==============================================================================


class TestSuperviseFailOpen:
    """Tests for fail-open behavior when supervise() raises an exception."""

    @pytest.mark.asyncio
    async def test_openai_supervise_exception_returns_ok_error(self):
        """Test that OpenAI.supervise() returns OK+error on exception."""
        provider = OpenAI(token=TEST_TOKEN)
        with patch.object(provider, "_classify_with_text", new_callable=AsyncMock) as mock_classify:
            mock_classify.side_effect = RuntimeError("simulated failure")
            result = await provider.supervise(context="test context")

        assert result.verdict == SupervisorVerdict.OK
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_anthropic_supervise_exception_returns_ok_error(self):
        """Test that Anthropic.supervise() returns OK+error on exception."""
        provider = Anthropic(token=TEST_TOKEN)
        with patch.object(provider, "_classify_with_forced_tool", new_callable=AsyncMock) as mock_classify:
            mock_classify.side_effect = RuntimeError("simulated failure")
            result = await provider.supervise(context="test context")

        assert result.verdict == SupervisorVerdict.OK
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_call_functions_supervise_exception_still_allows_tool(self):
        """Test that when supervise() returns error status, the tool is still executed (fail-open)."""
        provider = OpenAI(token=TEST_TOKEN)
        # Simulate supervise() failing by returning error status
        error_answer = SupervisorAnswer(verdict=SupervisorVerdict.OK, status="error")
        mock_supervise = AsyncMock(return_value=error_answer)

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ) as mock_tool_call:
                        results = await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test")],
                            system_prompt="test prompt",
                        )

        # When supervise returns OK+error (fail-open), the tool is still executed
        assert len(results) == 1
        mock_tool_call.assert_awaited_once()


# ==============================================================================
# 5. Final-answer hook — get_chat_response() template method
# ==============================================================================


class TestFinalAnswerHook:
    """Tests for supervisor final-answer hook in Provider.get_chat_response()."""

    @pytest.mark.asyncio
    async def test_supervisor_disabled_final_answer_not_checked(self):
        """Test that when disabled, final answer is returned without supervision."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock()

        response = _make_chat_response("test answer")
        mock_impl = AsyncMock(return_value=(response, []))

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch.object(gpt_settings, "supervisor_enabled", False):
                    result, new_msgs = await provider.get_chat_response(
                        messages=[Message(role="user", content="hello")],
                        user=TEST_USER,
                    )

        assert result.answer == "test answer"
        mock_supervise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ok_verdict_returns_answer_unchanged(self):
        """Test that OK verdict returns the answer unchanged."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock(return_value=_make_ok_answer())

        response = _make_chat_response("approved answer")
        mock_impl = AsyncMock(return_value=(response, []))

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch(
                    "chibi.services.providers.provider.prepare_system_prompt", new_callable=AsyncMock
                ) as mock_prep:
                    mock_prep.return_value = "enriched system prompt"
                    with patch.object(gpt_settings, "supervisor_enabled", True):
                        with patch.object(
                            RegisteredProviders,
                            "first_supervisor_ready",
                            new_callable=PropertyMock,
                            return_value=provider,
                        ):
                            result, new_msgs = await provider.get_chat_response(
                                messages=[Message(role="user", content="hello")],
                                user=TEST_USER,
                            )

        assert result.answer == "approved answer"
        mock_supervise.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_intervene_then_ok_on_retry(self):
        """Test that INTERVENE triggers retry, and OK on retry returns the new answer."""
        provider = OpenAI(token=TEST_TOKEN)
        # First call: intervene. Second call: ok.
        mock_supervise = AsyncMock(side_effect=[_make_intervene_answer(), _make_ok_answer()])

        first_response = _make_chat_response("rejected answer")
        second_response = _make_chat_response("corrected answer")
        mock_impl = AsyncMock(side_effect=[(first_response, []), (second_response, [])])

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch(
                    "chibi.services.providers.provider.prepare_system_prompt", new_callable=AsyncMock
                ) as mock_prep:
                    mock_prep.return_value = "enriched system prompt"
                    with patch.object(gpt_settings, "supervisor_enabled", True):
                        with patch.object(gpt_settings, "max_supervisor_retries", 2):
                            with patch.object(
                                RegisteredProviders,
                                "first_supervisor_ready",
                                new_callable=PropertyMock,
                                return_value=provider,
                            ):
                                result, new_msgs = await provider.get_chat_response(
                                    messages=[Message(role="user", content="hello")],
                                    user=TEST_USER,
                                )

        assert result.answer == "corrected answer"
        # supervise called twice: first for original, second for retry
        assert mock_supervise.call_count == 2
        # impl called twice: original + retry
        assert mock_impl.call_count == 2

    @pytest.mark.asyncio
    async def test_intervene_max_retries_exhausted_fail_open(self):
        """Test that after max retries, the last answer is returned (fail-open)."""
        provider = OpenAI(token=TEST_TOKEN)
        # All supervise calls return intervene
        mock_supervise = AsyncMock(return_value=_make_intervene_answer())

        last_response = _make_chat_response("last rejected answer")
        impl_side_effect: list[tuple[ChatResponseSchema, list[Message]]] = [
            (_make_chat_response("rejected 1"), []),
            (_make_chat_response("rejected 2"), []),
            (last_response, []),
        ]
        mock_impl = AsyncMock(side_effect=impl_side_effect)

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch(
                    "chibi.services.providers.provider.prepare_system_prompt", new_callable=AsyncMock
                ) as mock_prep:
                    mock_prep.return_value = "enriched system prompt"
                    with patch.object(gpt_settings, "supervisor_enabled", True):
                        with patch.object(gpt_settings, "max_supervisor_retries", 2):
                            with patch.object(
                                RegisteredProviders,
                                "first_supervisor_ready",
                                new_callable=PropertyMock,
                                return_value=provider,
                            ):
                                result, new_msgs = await provider.get_chat_response(
                                    messages=[Message(role="user", content="hello")],
                                    user=TEST_USER,
                                )

        # After max retries (2), the last answer is returned
        assert result.answer == "last rejected answer"
        # supervise called: original + 2 retries = 3 calls
        assert mock_supervise.call_count == 3

    @pytest.mark.asyncio
    async def test_no_supervisor_provider_returns_unchanged(self):
        """Test that when no supervisor provider is resolved, answer is returned unchanged."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock()

        response = _make_chat_response("unchecked answer")
        mock_impl = AsyncMock(return_value=(response, []))

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch.object(gpt_settings, "supervisor_enabled", True):
                    with patch.object(
                        RegisteredProviders,
                        "first_supervisor_ready",
                        new_callable=PropertyMock,
                        return_value=None,
                    ):
                        result, new_msgs = await provider.get_chat_response(
                            messages=[Message(role="user", content="hello")],
                            user=TEST_USER,
                        )

        assert result.answer == "unchecked answer"
        mock_supervise.assert_not_awaited()


# ==============================================================================
# 6. supervisor_enabled=False — supervise() not called
# ==============================================================================


class TestSupervisorDisabled:
    """Tests verifying that supervise() is never called when supervisor_enabled=False."""

    @pytest.mark.asyncio
    async def test_get_chat_response_disabled_no_supervise(self):
        """Test get_chat_response with supervisor disabled never calls supervise()."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock()

        response = _make_chat_response("answer")
        mock_impl = AsyncMock(return_value=(response, []))

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(provider, "_get_chat_response_impl", mock_impl):
                with patch.object(gpt_settings, "supervisor_enabled", False):
                    await provider.get_chat_response(
                        messages=[Message(role="user", content="hello")],
                        user=TEST_USER,
                    )

        mock_supervise.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_call_functions_disabled_no_supervise(self):
        """Test call_functions with supervisor disabled never calls supervise()."""
        provider = OpenAI(token=TEST_TOKEN)
        mock_supervise = AsyncMock()

        calls = [ToolCallSchema(tool_name="test_tool", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", False):
                with patch(
                    "chibi.services.providers.provider.RegisteredChibiTools.call",
                    new_callable=AsyncMock,
                    return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                ):
                    await provider.call_functions(
                        calls=calls,
                        caller_model="test-model",
                        caller_provider="test-provider",
                        messages=[Message(role="user", content="test")],
                        system_prompt="test",
                    )

        mock_supervise.assert_not_awaited()


# ==============================================================================
# 7. first_supervisor_ready resolution
# ==============================================================================


class TestFirstSupervisorReady:
    """Tests for RegisteredProviders.first_supervisor_ready resolution."""

    def test_resolves_explicit_supervisor_provider(self):
        """Test that first_supervisor_ready uses explicit supervisor_provider when available."""
        with patch.object(gpt_settings, "supervisor_provider", "OpenAI"):
            with patch.object(gpt_settings, "public_mode", False):
                rp = RegisteredProviders()
                result = rp.first_supervisor_ready

        assert result is not None
        assert result.name == "OpenAI"

    def test_falls_back_to_first_moderation_ready(self):
        """Test that first_supervisor_ready falls back to first_moderation_ready."""
        with patch.object(gpt_settings, "supervisor_provider", None):
            with patch.object(gpt_settings, "moderation_provider", None):
                with patch.object(gpt_settings, "public_mode", False):
                    rp = RegisteredProviders()
                    result = rp.first_supervisor_ready

        # Should fall back to first_moderation_ready (some provider)
        assert result is not None


# ==============================================================================
# 8. Tool-call hook context verification
# ==============================================================================


class TestToolCallHookContext:
    """Tests verifying that correct context is passed to supervise()."""

    @pytest.mark.asyncio
    async def test_context_includes_system_prompt(self):
        """Test that the context built for supervise includes the system prompt."""
        provider = OpenAI(token=TEST_TOKEN)
        captured_contexts: list[str] = []

        async def capture_supervise(*, context: str, **kwargs: object) -> SupervisorAnswer:
            captured_contexts.append(context)
            return _make_ok_answer()

        mock_supervise = AsyncMock(side_effect=capture_supervise)

        calls = [ToolCallSchema(tool_name="test_tool", args={"key": "value"})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="test_tool", status="ok", result="done"),
                    ):
                        await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=[Message(role="user", content="test message")],
                            system_prompt="you are a test agent",
                            user_id=TEST_USER.id,
                        )

        assert len(captured_contexts) == 1
        assert "you are a test agent" in captured_contexts[0]
        assert "test_tool" in captured_contexts[0]
        assert "test message" in captured_contexts[0]

    @pytest.mark.asyncio
    async def test_context_includes_message_history(self):
        """Test that the context includes the full message history."""
        provider = OpenAI(token=TEST_TOKEN)
        captured_contexts: list[str] = []

        async def capture_supervise(*, context: str, **kwargs: object) -> SupervisorAnswer:
            captured_contexts.append(context)
            return _make_ok_answer()

        mock_supervise = AsyncMock(side_effect=capture_supervise)

        messages = [
            Message(role="user", content="user request"),
            Message(role="assistant", content="assistant response"),
        ]

        calls = [ToolCallSchema(tool_name="tool_a", args={})]

        with patch.object(provider, "supervise", mock_supervise):
            with patch.object(gpt_settings, "supervisor_enabled", True):
                with patch.object(
                    RegisteredProviders,
                    "first_supervisor_ready",
                    new_callable=PropertyMock,
                    return_value=provider,
                ):
                    with patch(
                        "chibi.services.providers.provider.RegisteredChibiTools.call",
                        new_callable=AsyncMock,
                        return_value=ToolResponseSchema(tool_name="tool_a", status="ok", result="done"),
                    ):
                        await provider.call_functions(
                            calls=calls,
                            caller_model="test-model",
                            caller_provider="test-provider",
                            messages=messages,
                            system_prompt="system prompt",
                        )

        assert len(captured_contexts) == 1
        assert "user request" in captured_contexts[0]
        assert "assistant response" in captured_contexts[0]
