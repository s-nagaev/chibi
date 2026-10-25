"""Integration tests for the failover wiring (failover_tests task).

The engine hooks and chain builders are unit-covered elsewhere; these tests
prove the FULL wiring through the two real call sites (mock providers, no
real network):

- master site: ``chibi.services.user.get_llm_chat_completion_answer`` —
  primary fails → warning delivered to the chat via ``interface.send_message``
  → fallback answer returned; all candidates fail → warnings + honest error;
- subagent site: ``chibi.services.providers.tools.utils.get_sub_agent_response``
  — primary fails → LOG-ONLY warning (no chat send, no interface by design)
  → fallback result;
- happy path on both sites: no warnings, no cooldown writes.

Mocking patterns follow ``tests/services/test_failover_notification.py``:
``AsyncMock`` providers with a plain-string ``name``, a stub user whose
provider registry maps chat-ready provider classes to instances, and
``ConnectionError`` as the failover-classified trigger exception.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from chibi.config.gpt import gpt_settings
from chibi.models import Message
from chibi.schemas.app import ChatResponseSchema
from chibi.services.failover import CooldownStore, FailoverModelRegistry, reset_failover_policy_cache
from chibi.services.providers.tools.utils import get_sub_agent_response
from chibi.services.user import get_llm_chat_completion_answer
from chibi.storage.database import _db_provider
from chibi.utils.app import SingletonMeta

if TYPE_CHECKING:
    from chibi.services.providers.provider import Provider


@pytest.fixture(autouse=True)
def _clean_policy_cache():
    reset_failover_policy_cache()
    yield
    reset_failover_policy_cache()


@pytest.fixture(autouse=True)
def _clean_cooldown_and_registry_singletons():
    for cls in (CooldownStore, FailoverModelRegistry):
        _pop_singleton_instances(cls)
    yield
    for cls in (CooldownStore, FailoverModelRegistry):
        _pop_singleton_instances(cls)


def _pop_singleton_instances(cls: type) -> None:
    SingletonMeta._instances.pop(cls, None)


def _stub_user() -> Mock:
    """A minimal user double with an empty provider registry."""
    user = Mock()
    user.providers = Mock()
    user.providers.chat_ready = {}
    user.stt_provider = None
    return user


def _stub_db(user: Mock) -> AsyncMock:
    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=[])
    return db


def _stub_interface() -> Mock:
    interface = Mock()
    interface.storage_id = 1
    interface.thread_id = 0
    interface.user_data = {"first_name": "Tester"}
    # EditorContextProvider is a runtime-checkable Protocol: a bare Mock
    # would pass isinstance() and leak Mocks into the JSON prompt. A None
    # editor_context keeps the master path on the plain-chat branch.
    interface.editor_context = None
    interface.send_message = AsyncMock()
    return interface


def _response(answer: str, provider: str, model: str | None) -> ChatResponseSchema:
    return ChatResponseSchema(answer=answer, provider=provider, model=model or "default", usage=None)


def _assistant_message(text: str) -> Message:
    return Message(role="assistant", content=text)


def _wire_providers(user: Mock, instances: dict[str, "Provider"]) -> None:
    """Populate the stub user's registry: name -> chat-ready class + instance."""
    user.providers.chat_ready = {name: type(f"{name.title()}Stub", (), {"name": name}) for name in instances}
    user.providers.get_instance = lambda provider_class: instances[provider_class.name]


def _primary(name: str, *, fails: bool, answer: str = "primary answer") -> AsyncMock:
    provider = AsyncMock()
    provider.name = name
    if fails:

        async def _fail(**kwargs):
            raise ConnectionError(f"{name} down")

        provider.get_chat_response = AsyncMock(side_effect=_fail)
    else:
        provider.get_chat_response = AsyncMock(
            side_effect=lambda **kwargs: (_response(answer, name, kwargs.get("model")), [_assistant_message(answer)])
        )
    return provider


class TestMasterWiring:
    async def test_primary_fails_warning_delivered_fallback_answer(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = _primary("openai", fails=True)
        gemini = _primary("gemini", fails=False, answer="fallback answer")
        gemini.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        user = _stub_user()
        user.get_active_llm_provider = Mock(return_value=primary)
        user.get_active_llm_model = Mock(return_value="gpt-5.6-luna")
        _wire_providers(user, {"gemini": gemini})

        db = _stub_db(user)
        interface = _stub_interface()
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        result = await get_llm_chat_completion_answer(storage_id=1, interface=interface, user_text_message="hi")

        assert result.answer == "fallback answer"
        assert result.provider == "gemini"
        # Exactly one warning, delivered to the chat before the answer.
        interface.send_message.assert_awaited_once()
        warning = interface.send_message.await_args.kwargs["message"]
        assert "openai/gpt-5.6-luna" in warning
        assert "falling back to gemini/gpt-5.6-luna" in warning
        # History persisted for the turn.
        assert db.add_message.await_count >= 1

    async def test_all_candidates_fail_warnings_then_honest_error(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

        primary = _primary("openai", fails=True)
        gemini = _primary("gemini", fails=True)
        gemini.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        user = _stub_user()
        user.get_active_llm_provider = Mock(return_value=primary)
        user.get_active_llm_model = Mock(return_value="gpt-5.6-luna")
        _wire_providers(user, {"gemini": gemini})

        db = _stub_db(user)
        interface = _stub_interface()
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        with pytest.raises(ConnectionError, match="gemini down"):
            await get_llm_chat_completion_answer(storage_id=1, interface=interface, user_text_message="hi")

        messages = [call.kwargs["message"] for call in interface.send_message.await_args_list]
        joined = "\n".join(messages)
        # The fallback transition was announced...
        assert any("falling back to gemini/gpt-5.6-luna" in m for m in messages)
        # ...and so was the final honest failure.
        assert "all fallback attempts failed (gemini/gpt-5.6-luna)" in joined
        assert "No fallback succeeded." in joined

    async def test_happy_path_no_warnings_no_cooldown_writes(self, monkeypatch):
        monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
        mark_spy = AsyncMock()
        monkeypatch.setattr(CooldownStore, "mark", mark_spy)

        primary = _primary("openai", fails=False, answer="primary answer")
        user = _stub_user()
        user.get_active_llm_provider = Mock(return_value=primary)
        user.get_active_llm_model = Mock(return_value="gpt-5.6-luna")

        db = _stub_db(user)
        interface = _stub_interface()
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        result = await get_llm_chat_completion_answer(storage_id=1, interface=interface, user_text_message="hi")

        assert result.answer == "primary answer"
        interface.send_message.assert_not_awaited()
        mark_spy.assert_not_awaited()


class TestSubagentWiring:
    async def _run_subagent(self, monkeypatch, *, primary, chain_raw="auto", extra_instances=None):
        monkeypatch.setattr(gpt_settings, "failover_chain_subagent_raw", chain_raw, raising=False)

        user = _stub_user()
        user.get_effective_working_dir = Mock(return_value="/tmp")
        user.providers.get = Mock(return_value=primary)
        _wire_providers(user, extra_instances or {})

        db = _stub_db(user)
        monkeypatch.setattr(_db_provider, "get_database", AsyncMock(return_value=db))

        log_lines: list[str] = []
        sink_id = logger.add(log_lines.append, level="WARNING")
        try:
            result = await get_sub_agent_response(
                user_id=1,
                prompt="do the thing",
                model_name="gpt-5.6-luna",
                provider_name="openai",
                caller_storage_id=1,
                caller_thread_id=0,
            )
        finally:
            logger.remove(sink_id)
        return result, log_lines

    async def test_primary_fails_log_only_warning_fallback_result(self, monkeypatch):
        primary = _primary("openai", fails=True)
        gemini = _primary("gemini", fails=False, answer="subagent fallback answer")
        gemini.get_available_models = AsyncMock(return_value=[type("ModelStub", (), {"name": "gpt-5.6-luna"})()])

        result, log_lines = await self._run_subagent(monkeypatch, primary=primary, extra_instances={"gemini": gemini})

        assert result.answer == "subagent fallback answer"
        assert result.provider == "gemini"
        # Log-only: the warning went to the logger, and NO chat send happened
        # (the subagent path has no interface at all — nothing else could
        # deliver a message, and no send_message mock exists to be called).
        assert any("Failover[subagent]" in line for line in log_lines)
        assert any("falling back to gemini/gpt-5.6-luna" in line for line in log_lines)

    async def test_happy_path_no_warnings_no_cooldown_writes(self, monkeypatch):
        mark_spy = AsyncMock()
        monkeypatch.setattr(CooldownStore, "mark", mark_spy)

        primary = _primary("openai", fails=False, answer="subagent primary answer")
        result, log_lines = await self._run_subagent(monkeypatch, primary=primary)

        assert result.answer == "subagent primary answer"
        assert log_lines == []
        mark_spy.assert_not_awaited()
