"""Unit tests for failover notification behaviour (failover_notification task).

Covers the user-facing warning/failure message builders in
``chibi.services.failover`` and the ``on_fallback`` / ``on_failure``
notification hooks of ``FailoverEngine`` / ``run_with_failover``:

- happy path (primary success) fires NO hooks;
- every real fallback attempt (auto ladder AND manual chain) fires
  ``on_fallback`` with the trigger and the pair about to be attempted;
- the final honest failure fires ``on_failure`` only when fallbacks were
  genuinely attempted;
- notification-hook failures never break the failover walk;
- failover semantics themselves (ladder order, cooldown, manual walk) are
  untouched by the notification layer.
"""

from unittest.mock import AsyncMock

import pytest
from loguru import logger

from chibi.config.gpt import gpt_settings
from chibi.services.failover import (
    CooldownStore,
    FailoverEngine,
    FailoverModelRegistry,
    FailoverPair,
    FailoverTrigger,
    failover_failure_message,
    failover_warning_message,
    format_failover_pair,
    reset_failover_policy_cache,
    run_with_failover,
)


@pytest.fixture(autouse=True)
def _clean_policy_cache():
    reset_failover_policy_cache()
    yield
    reset_failover_policy_cache()


@pytest.fixture(autouse=True)
def _clean_cooldown_and_registry_singletons():
    """CooldownStore / FailoverModelRegistry are process-wide singletons:
    drop their state between tests so cooldowns from one test never leak
    into the ladder of another."""
    from chibi.utils.app import SingletonMeta

    for cls in (CooldownStore, FailoverModelRegistry):
        SingletonMeta._instances.pop(cls, None)
    yield
    for cls in (CooldownStore, FailoverModelRegistry):
        SingletonMeta._instances.pop(cls, None)


def _make_trigger(kind: str = "rate_limit", provider: str = "openai", model: str | None = "gpt-5.6-luna"):
    return FailoverTrigger(kind=kind, provider=provider, model=model, original=RuntimeError("boom"))


class _Recorder:
    """Collect on_fallback / on_failure hook invocations."""

    def __init__(self) -> None:
        self.fallbacks: list[
            tuple[str, str, str, str | None]
        ] = []  # (kind, trigger pair, target provider, target model)
        self.failures: list[tuple[str, list[tuple[str, str | None]]]] = []

    def on_fallback(self, trigger, pair) -> None:
        self.fallbacks.append((trigger.kind, f"{trigger.provider}/{trigger.model}", pair.provider, pair.model))

    def on_failure(self, trigger, fallback_targets) -> None:
        self.failures.append((trigger.kind, [(p.provider, p.model) for p in fallback_targets]))


def _stub_user(*, chat_ready: dict | None = None, instances: dict | None = None):
    user = AsyncMock()
    if chat_ready is not None and instances is not None:
        user.providers.chat_ready = chat_ready
        user.providers.get_instance = lambda provider_class: instances[provider_class.name]
    else:
        user.providers.chat_ready = {}  # empty registry: no fallback candidates
    return user


class TestMessageBuilders:
    def test_format_pair_with_model(self):
        assert format_failover_pair("openai", "gpt-5.6-luna") == "openai/gpt-5.6-luna"

    def test_format_pair_default_model(self):
        assert format_failover_pair("gemini", None) == "gemini/<default>"

    def test_warning_message_matches_owner_confirmed_shape(self):
        message = failover_warning_message(trigger=_make_trigger(), fallback=FailoverPair("gemini", "gemini-2.5-flash"))
        assert message == "⚠️ openai/gpt-5.6-luna hit rate_limit — falling back to gemini/gemini-2.5-flash"

    def test_warning_message_with_default_models(self):
        message = failover_warning_message(
            trigger=_make_trigger(kind="timeout", provider="melious", model=None),
            fallback=FailoverPair("lyceum", None),
        )
        assert message == "⚠️ melious/<default> hit timeout — falling back to lyceum/<default>"

    def test_failure_message_lists_attempted_targets(self):
        message = failover_failure_message(
            trigger=_make_trigger(kind="server_error"),
            fallback_targets=[FailoverPair("gemini", "gemini-2.5-flash"), FailoverPair("lyceum", None)],
        )
        assert message == (
            "⚠️ openai/gpt-5.6-luna hit server_error — all fallback attempts failed "
            "(gemini/gemini-2.5-flash, lyceum/<default>). No fallback succeeded."
        )

    def test_failure_message_without_targets(self):
        message = failover_failure_message(trigger=_make_trigger(kind="network"), fallback_targets=[])
        assert message == "⚠️ openai/gpt-5.6-luna hit network — no fallback succeeded."


class TestEngineHooksAuto:
    async def test_happy_path_fires_no_hooks(self):
        primary = AsyncMock()
        primary.name = "openai"

        recorder = _Recorder()

        async def call(provider, model):
            return (AsyncMock(provider=provider.name, model=model), [])

        engine = FailoverEngine(user=_stub_user())
        result = await engine.run(
            role="master",
            primary_provider=primary,
            model="gpt-5.6-luna",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )
        assert result.fallback_used is False
        assert recorder.fallbacks == []
        assert recorder.failures == []

    async def test_auto_fallback_fires_on_fallback_with_trigger_and_target(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = AsyncMock()
        primary.name = "openai"
        fallback = AsyncMock()
        fallback.name = "gemini"
        fallback.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        lyceum_stub = type("GeminiStub", (), {"name": "gemini"})
        user = _stub_user(chat_ready={"gemini": lyceum_stub}, instances={"gemini": fallback})

        calls: list[tuple[str, str | None]] = []

        async def call(provider, model):
            calls.append((provider.name, model))
            if provider.name == "openai":
                # RateLimitError is not constructible without the SDK
                # request/response payload; use a trigger-classified
                # network error instead (ConnectionError -> "network").
                raise ConnectionError("primary down")
            return (AsyncMock(provider=provider.name, model=model), [])

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        result = await engine.run(
            role="master",
            primary_provider=primary,
            model="gpt-5.6-luna",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )

        assert result.fallback_used is True
        assert recorder.failures == []
        assert len(recorder.fallbacks) == 1
        kind, trigger_pair, target_provider, target_model = recorder.fallbacks[0]
        assert kind == "network"
        assert trigger_pair == "openai/gpt-5.6-luna"
        # Auto-ladder step 2: exact model match on the other provider.
        assert (target_provider, target_model) == ("gemini", "gpt-5.6-luna")
        # The fallback was actually attempted AFTER the warning fired.
        assert calls == [("openai", "gpt-5.6-luna"), ("gemini", "gpt-5.6-luna")]

    async def test_multiple_fallbacks_fire_one_warning_each(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = AsyncMock()
        primary.name = "openai"
        second = AsyncMock()
        second.name = "gemini"
        second.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])
        third = AsyncMock()
        third.name = "lyceum"
        third.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        user = _stub_user(
            chat_ready={
                "gemini": type("GeminiStub", (), {"name": "gemini"}),
                "lyceum": type("LyceumStub", (), {"name": "lyceum"}),
            },
            instances={"gemini": second, "lyceum": third},
        )

        async def call(provider, model):
            if provider.name in ("openai", "gemini"):
                raise ConnectionError(f"{provider.name} down")
            return (AsyncMock(provider=provider.name, model=model), [])

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        result = await engine.run(
            role="master",
            primary_provider=primary,
            model="gpt-5.6-luna",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )

        assert result.fallback_used is True
        assert len(recorder.fallbacks) == 2
        # First transition: openai (network) -> gemini.
        assert recorder.fallbacks[0][0] == "network"
        assert recorder.fallbacks[0][1] == "openai/gpt-5.6-luna"
        assert recorder.fallbacks[0][2] == "gemini"
        # Second transition: gemini (network) -> lyceum (step 2: the model
        # exists at lyceum too, so the ladder continues with the exact model).
        assert recorder.fallbacks[1][0] == "network"
        assert recorder.fallbacks[1][1] == "gemini/gpt-5.6-luna"
        assert recorder.fallbacks[1][2] == "lyceum"
        assert recorder.fallbacks[1][3] == "gpt-5.6-luna"
        assert recorder.failures == []

    async def test_all_fallbacks_failed_fires_on_failure_before_raise(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = AsyncMock()
        primary.name = "openai"
        fallback = AsyncMock()
        fallback.name = "gemini"
        fallback.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        user = _stub_user(
            chat_ready={"gemini": type("GeminiStub", (), {"name": "gemini"})}, instances={"gemini": fallback}
        )

        async def call(provider, model):
            raise ConnectionError(f"{provider.name} down")

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        with pytest.raises(ConnectionError):
            await engine.run(
                role="master",
                primary_provider=primary,
                model="gpt-5.6-luna",
                call=call,
                on_fallback=recorder.on_fallback,
                on_failure=recorder.on_failure,
            )

        assert len(recorder.fallbacks) == 1  # gemini was announced before its attempt
        assert len(recorder.failures) == 1
        kind, targets = recorder.failures[0]
        assert kind == "network"
        assert ("gemini", "gpt-5.6-luna") in targets

    async def test_single_attempt_failure_fires_no_failure_hook(self, monkeypatch):
        # AUTO with no other chat-ready providers: no fallback candidates
        # exist, so on_failure must stay silent (the standard error handler
        # reports the raw failure; "no fallback succeeded" would be noise).
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = AsyncMock()
        primary.name = "openai"
        user = _stub_user()  # empty registry -> no candidates

        async def call(provider, model):
            raise ConnectionError("down")

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        with pytest.raises(ConnectionError):
            await engine.run(
                role="master",
                primary_provider=primary,
                model=None,
                call=call,
                on_fallback=recorder.on_fallback,
                on_failure=recorder.on_failure,
            )

        assert recorder.fallbacks == []
        assert recorder.failures == []


class TestEngineHooksManual:
    async def test_manual_chain_steps_are_announced(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"

        user = _stub_user(
            chat_ready={"lyceum": type("LyceumStub", (), {"name": "lyceum"})},
            instances={"lyceum": fallback},
        )

        async def call(provider, model):
            if provider.name == "melious":
                raise TimeoutError("primary timed out")
            return (AsyncMock(provider=provider.name, model=model), [])

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        result = await engine.run(
            role="master",
            primary_provider=primary,
            model="glm-5.3-flash",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )

        assert result.fallback_used is True
        assert recorder.fallbacks == [("timeout", "melious/glm-5.3-flash", "lyceum", "glm-5.3-flash")]
        assert recorder.failures == []

    async def test_manual_chain_all_fail_announces_failure(self, monkeypatch):
        monkeypatch.setattr(
            gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash, gemini/gemini-2.5-flash", raising=False
        )

        primary = AsyncMock()
        primary.name = "melious"
        lyceum = AsyncMock()
        lyceum.name = "lyceum"
        gemini = AsyncMock()
        gemini.name = "gemini"

        user = _stub_user(
            chat_ready={
                "lyceum": type("LyceumStub", (), {"name": "lyceum"}),
                "gemini": type("GeminiStub", (), {"name": "gemini"}),
            },
            instances={"lyceum": lyceum, "gemini": gemini},
        )

        async def call(provider, model):
            raise TimeoutError(f"{provider.name} timed out")

        recorder = _Recorder()
        engine = FailoverEngine(user=user)
        with pytest.raises(TimeoutError):
            await engine.run(
                role="master",
                primary_provider=primary,
                model="glm-5.3-flash",
                call=call,
                on_fallback=recorder.on_fallback,
                on_failure=recorder.on_failure,
            )

        assert [(p, m) for _, _, p, m in recorder.fallbacks] == [
            ("lyceum", "glm-5.3-flash"),
            ("gemini", "gemini-2.5-flash"),
        ]
        assert len(recorder.failures) == 1
        kind, targets = recorder.failures[0]
        assert kind == "timeout"
        assert ("lyceum", "glm-5.3-flash") in targets
        assert ("gemini", "gemini-2.5-flash") in targets

    async def test_disabled_mode_fires_no_hooks(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "disabled", raising=False)

        primary = AsyncMock()
        primary.name = "openai"

        async def call(provider, model):
            return (AsyncMock(provider=provider.name, model=model), [])

        recorder = _Recorder()
        engine = FailoverEngine(user=_stub_user())
        await engine.run(
            role="master",
            primary_provider=primary,
            model="gpt-5.6-luna",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )
        assert recorder.fallbacks == []
        assert recorder.failures == []


class TestHookRobustness:
    async def test_raising_hook_does_not_break_the_walk(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"

        user = _stub_user(
            chat_ready={"lyceum": type("LyceumStub", (), {"name": "lyceum"})},
            instances={"lyceum": fallback},
        )

        async def call(provider, model):
            if provider.name == "melious":
                raise ConnectionError("down")
            return (AsyncMock(provider=provider.name, model=model), [])

        def bad_hook(trigger, pair):
            raise RuntimeError("notification channel exploded")

        engine = FailoverEngine(user=user)
        result = await engine.run(
            role="master",
            primary_provider=primary,
            model="glm-5.3-flash",
            call=call,
            on_fallback=bad_hook,
            on_failure=bad_hook,
        )
        assert result.fallback_used is True
        assert result.serving_provider == "lyceum"

    async def test_async_hook_is_awaited(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"

        user = _stub_user(
            chat_ready={"lyceum": type("LyceumStub", (), {"name": "lyceum"})},
            instances={"lyceum": fallback},
        )

        async def call(provider, model):
            if provider.name == "melious":
                raise ConnectionError("down")
            return (AsyncMock(provider=provider.name, model=model), [])

        seen: list[tuple[str, str]] = []

        async def async_hook(trigger, pair):
            seen.append((trigger.kind, pair.provider))

        engine = FailoverEngine(user=user)
        await engine.run(
            role="master",
            primary_provider=primary,
            model="glm-5.3-flash",
            call=call,
            on_fallback=async_hook,
        )
        assert seen == [("network", "lyceum")]

    async def test_failing_hook_is_logged_not_raised(self, monkeypatch, caplog):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)
        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"
        user = _stub_user(
            chat_ready={"lyceum": type("LyceumStub", (), {"name": "lyceum"})},
            instances={"lyceum": fallback},
        )

        async def call(provider, model):
            if provider.name == "melious":
                raise ConnectionError("down")
            return (AsyncMock(provider=provider.name, model=model), [])

        def bad_hook(trigger, pair):
            raise RuntimeError("hook down")

        try:
            engine = FailoverEngine(user=user)
            await engine.run(
                role="master", primary_provider=primary, model="glm-5.3-flash", call=call, on_fallback=bad_hook
            )
        finally:
            logger.remove(sink_id)
        assert any("on_fallback notification hook failed" in m for m in messages)


class TestRunWithFailoverForwardsHooks:
    async def test_hooks_are_forwarded(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "lyceum/glm-5.3-flash", raising=False)

        primary = AsyncMock()
        primary.name = "melious"
        fallback = AsyncMock()
        fallback.name = "lyceum"
        user = _stub_user(
            chat_ready={"lyceum": type("LyceumStub", (), {"name": "lyceum"})},
            instances={"lyceum": fallback},
        )

        async def call(provider, model):
            if provider.name == "melious":
                raise ConnectionError("down")
            return (AsyncMock(provider=provider.name, model=model), [])

        recorder = _Recorder()
        result = await run_with_failover(
            role="master",
            user=user,
            primary_provider=primary,
            model="glm-5.3-flash",
            call=call,
            on_fallback=recorder.on_fallback,
            on_failure=recorder.on_failure,
        )
        assert result.fallback_used is True
        assert recorder.fallbacks == [("network", "melious/glm-5.3-flash", "lyceum", "glm-5.3-flash")]
        assert recorder.failures == []
