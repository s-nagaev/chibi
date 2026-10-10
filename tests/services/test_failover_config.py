"""Unit tests for the parsing and resolution of failover chain settings.

Covers ``parse_failover_chain`` and ``resolve_failover_policy`` in
``chibi.services.failover`` (task: failover_config).
"""

from unittest.mock import AsyncMock

import pytest
from loguru import logger

from chibi.config.gpt import gpt_settings
from chibi.services.failover import (
    FailoverEngine,
    FailoverMode,
    FailoverPair,
    parse_failover_chain,
    reset_failover_policy_cache,
    resolve_failover_policy,
)


@pytest.fixture(autouse=True)
def _clean_policy_cache():
    reset_failover_policy_cache()
    yield
    reset_failover_policy_cache()


@pytest.fixture
def loguru_messages():
    """Capture loguru output via a temporary sink (loguru does not
    propagate to the standard logging handlers used by ``caplog``)."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    yield messages
    logger.remove(sink_id)


class TestParseFailoverChain:
    def test_unset_returns_auto(self):
        policy = parse_failover_chain(None, "master")
        assert policy.mode is FailoverMode.AUTO
        assert policy.chain == []

    def test_blank_returns_auto(self):
        for raw in ("", "   ", "\t"):
            policy = parse_failover_chain(raw, "master")
            assert policy.mode is FailoverMode.AUTO
            assert policy.chain == []

    def test_auto_returns_auto(self):
        policy = parse_failover_chain("auto", "master")
        assert policy.mode is FailoverMode.AUTO
        assert policy.chain == []

    def test_auto_case_insensitive_and_whitespace(self):
        policy = parse_failover_chain("  AUTO ", "subagent")
        assert policy.mode is FailoverMode.AUTO

    def test_disabled_returns_disabled(self):
        policy = parse_failover_chain("disabled", "master")
        assert policy.mode is FailoverMode.DISABLED
        assert policy.chain == []

    def test_disabled_case_insensitive(self):
        policy = parse_failover_chain("Disabled", "subagent")
        assert policy.mode is FailoverMode.DISABLED

    def test_single_pair_manual(self):
        policy = parse_failover_chain("openai/gpt-5.6-luna", "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [FailoverPair(provider="openai", model="gpt-5.6-luna")]

    def test_multiple_pairs_ordered(self):
        raw = "melious/glm-5.3-flash, lyceum/glm-5.3-flash, deepseek/deepseek-flash"
        policy = parse_failover_chain(raw, "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [
            FailoverPair(provider="melious", model="glm-5.3-flash"),
            FailoverPair(provider="lyceum", model="glm-5.3-flash"),
            FailoverPair(provider="deepseek", model="deepseek-flash"),
        ]

    def test_first_slash_split_model_may_contain_slash(self):
        policy = parse_failover_chain("openrouter/deepseek/deepseek-chat", "master")
        assert policy.chain == [FailoverPair(provider="openrouter", model="deepseek/deepseek-chat")]

    def test_whitespace_tolerated(self):
        policy = parse_failover_chain("  melious / glm-5.3-flash ,  gemini / gemini-2.5-flash  ", "subagent")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [
            FailoverPair(provider="melious", model="glm-5.3-flash"),
            FailoverPair(provider="gemini", model="gemini-2.5-flash"),
        ]

    def test_missing_slash_skipped_with_warning(self, loguru_messages):
        policy = parse_failover_chain("openai, gemini/gemini-2.5-flash", "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [FailoverPair(provider="gemini", model="gemini-2.5-flash")]
        assert any("expected 'provider/model'" in m for m in loguru_messages)

    def test_empty_model_part_skipped(self):
        policy = parse_failover_chain("openai/", "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == []

    def test_empty_provider_part_skipped(self):
        policy = parse_failover_chain("/gpt-5.6-luna", "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == []

    def test_empty_entries_between_commas_skipped(self):
        policy = parse_failover_chain("openai/gpt-5.6-luna,,gemini/gemini-2.5-flash", "master")
        assert policy.chain == [
            FailoverPair(provider="openai", model="gpt-5.6-luna"),
            FailoverPair(provider="gemini", model="gemini-2.5-flash"),
        ]

    def test_trailing_comma_tolerated(self):
        policy = parse_failover_chain("openai/gpt-5.6-luna,", "master")
        assert policy.chain == [FailoverPair(provider="openai", model="gpt-5.6-luna")]

    def test_all_entries_invalid_degrades_to_manual_with_empty_chain(self):
        policy = parse_failover_chain("no-slash-here, /missing-provider, openai/", "master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == []

    def test_duplicate_pairs_skipped(self):
        raw = "openai/gpt-5.6-luna, openai/gpt-5.6-luna, gemini/gemini-2.5-flash"
        policy = parse_failover_chain(raw, "subagent")
        assert policy.chain == [
            FailoverPair(provider="openai", model="gpt-5.6-luna"),
            FailoverPair(provider="gemini", model="gemini-2.5-flash"),
        ]

    def test_duplicate_provider_detected_case_insensitively(self):
        raw = "OpenAI/gpt-5.6-luna, openai/gpt-5.6-luna"
        policy = parse_failover_chain(raw, "master")
        assert policy.chain == [FailoverPair(provider="OpenAI", model="gpt-5.6-luna")]


class TestResolveFailoverPolicy:
    def test_default_is_auto_when_env_unset(self):
        policy = resolve_failover_policy("master")
        assert policy.mode is FailoverMode.AUTO
        assert policy.chain == []

    def test_master_reads_master_setting(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "openai/gpt-5.6-luna", raising=False)
        policy = resolve_failover_policy("master")
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [FailoverPair(provider="openai", model="gpt-5.6-luna")]

    def test_subagent_reads_subagent_setting(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_subagent_raw", "disabled", raising=False)
        policy = resolve_failover_policy("subagent")
        assert policy.mode is FailoverMode.DISABLED

    def test_roles_are_independent(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "disabled", raising=False)
        monkeypatch.setattr(gpt_settings, "failover_chain_subagent_raw", "openai/gpt-5.6-luna", raising=False)
        assert resolve_failover_policy("master").mode is FailoverMode.DISABLED
        subagent = resolve_failover_policy("subagent")
        assert subagent.mode is FailoverMode.MANUAL
        assert subagent.chain == [FailoverPair(provider="openai", model="gpt-5.6-luna")]

    def test_result_is_memoized_per_role(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
        first = resolve_failover_policy("master")
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "disabled", raising=False)
        assert resolve_failover_policy("master") is first
        reset_failover_policy_cache()
        assert resolve_failover_policy("master").mode is FailoverMode.DISABLED

    def test_invalid_entries_do_not_crash(self, monkeypatch, loguru_messages):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto, ???, openai/gpt-5.6-luna", raising=False)
        policy = resolve_failover_policy("master")  # must not raise
        assert policy.mode is FailoverMode.MANUAL
        assert policy.chain == [FailoverPair(provider="openai", model="gpt-5.6-luna")]
        assert any("???" in m for m in loguru_messages)


class TestManualChainHeadIsPrimary:
    async def test_engine_attempts_primary_before_configured_chain(self, monkeypatch):
        monkeypatch.setattr(
            gpt_settings,
            "failover_chain_master_raw",
            "openai/gpt-5.6-luna, gemini/gemini-2.5-flash",
            raising=False,
        )
        reset_failover_policy_cache()

        primary = AsyncMock()
        primary.name = "melious"

        calls: list[tuple[str, str | None]] = []

        async def call(provider, model):
            calls.append((provider.name, model))
            if provider.name == "melious":
                raise RuntimeError("primary down")
            return (AsyncMock(provider=provider.name, model=model), [])

        engine = FailoverEngine(user=AsyncMock())
        with pytest.raises(RuntimeError, match="primary down"):
            # The configured chain steps are not available to this (mock)
            # user, so the walk ends honestly on the primary's exception —
            # but the primary MUST have been attempted first, as chain head.
            await engine.run(role="master", primary_provider=primary, model=None, call=call)

        assert calls == [("melious", None)]

    async def test_engine_walks_configured_chain_after_primary_failure(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)
        reset_failover_policy_cache()

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"

        # Minimal provider registry double: one chat-ready provider class
        # ("lyceum") whose instance is the fallback mock above.
        lyceum_class = type("LyceumStub", (), {"name": "lyceum"})
        user = AsyncMock()
        user.providers.chat_ready = {"lyceum": lyceum_class}
        user.providers.get_instance = lambda provider_class: fallback

        calls: list[tuple[str, str | None]] = []

        async def call(provider, model):
            calls.append((provider.name, model))
            if provider.name == "melious":
                # ConnectionError is a failover-worthy trigger (network).
                raise ConnectionError("primary down")
            return (AsyncMock(provider=provider.name, model=model), [])

        engine = FailoverEngine(user=user)
        result = await engine.run(role="master", primary_provider=primary, model="glm-5.3-flash", call=call)

        assert calls == [("melious", "glm-5.3-flash"), ("lyceum", "glm-5.3-flash")]
        assert result.fallback_used is True
        assert result.serving_provider == "lyceum"
        assert result.serving_model == "glm-5.3-flash"
        assert result.attempts[-1].ok is True
