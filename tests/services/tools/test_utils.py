"""Tests for thread-context plumbing in provider tool utilities."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from chibi.models import User
from chibi.schemas.app import ChatResponseSchema, ModeratorsAnswer
from chibi.services.providers.provider import Provider
from chibi.services.providers.tools.cmd import RunCommandInTerminalTool
from chibi.services.providers.tools.memory import GetCurrentWorkingDirTool, SetWorkingDirTool
from chibi.services.providers.tools.schemas import ToolCallSchema
from chibi.services.providers.tools.tool import ChibiTool, RegisteredChibiTools
from chibi.services.providers.tools.utils import (
    get_sub_agent_response,
    resolve_session_context,
)
from chibi.services.user import set_thread_working_dir
from chibi.storage.local import LocalStorage


class _StubProviders:
    def __init__(self, provider: Any) -> None:
        self._provider = provider

    def get(self, provider_name: str) -> Any:
        return self._provider if provider_name == "OpenAI" else None


@pytest.fixture
def local_db(tmp_path: Path) -> LocalStorage:
    """Local storage backend pointed at a temp directory."""
    return LocalStorage(storage_path=str(tmp_path))


@pytest.fixture
def patched_db_provider(local_db: LocalStorage):
    """Route every @inject_database call to the temp LocalStorage."""
    fake_provider = SimpleNamespace(get_database=AsyncMock(return_value=local_db))
    with patch("chibi.storage.database._db_provider", fake_provider):
        yield local_db


class TestResolveSessionContext:
    def test_interface_takes_priority(self) -> None:
        interface = SimpleNamespace(storage_id=7, thread_id=9)

        storage_id, thread_id = resolve_session_context(
            interface=cast(Any, interface),
            caller_storage_id=1,
            caller_thread_id=2,
        )

        assert (storage_id, thread_id) == (7, 9)

    def test_caller_options_used_when_no_interface(self) -> None:
        assert resolve_session_context(caller_storage_id=11, caller_thread_id=42) == (11, 42)

    def test_raises_when_both_absent(self) -> None:
        with pytest.raises(ValueError, match="session context"):
            resolve_session_context()

    def test_raises_on_partial_identity(self) -> None:
        with pytest.raises(ValueError, match="session context"):
            resolve_session_context(caller_storage_id=11)


class TestSubAgentPropagation:
    @pytest.mark.asyncio
    async def test_parent_ids_and_effective_cwd_reach_sub_agent_request(
        self, patched_db_provider: LocalStorage
    ) -> None:
        await set_thread_working_dir(user_id=1, thread_id=9, new_wd="~/thread-override")
        real_user = await patched_db_provider.get_or_create_user(user_id=1)

        provider_mock = MagicMock()
        provider_mock.get_chat_response = AsyncMock(
            return_value=(ChatResponseSchema(answer="done", provider="OpenAI", model="m", usage=None), [])
        )

        async def fake_get_or_create_user(user_id: int) -> User:
            return real_user

        db_mock = MagicMock()
        db_mock.get_or_create_user = AsyncMock(side_effect=fake_get_or_create_user)

        with (
            patch.object(User, "providers", new_callable=PropertyMock) as providers_prop,
            patch(
                "chibi.storage.database._db_provider",
                SimpleNamespace(get_database=AsyncMock(return_value=db_mock)),
            ),
        ):
            providers_prop.return_value = _StubProviders(provider_mock)
            response = await get_sub_agent_response(
                user_id=1,
                prompt="do things",
                model_name="m",
                provider_name="OpenAI",
                caller_storage_id=77,
                caller_thread_id=9,
            )

        assert response.answer == "done"
        _, kwargs = provider_mock.get_chat_response.await_args
        assert kwargs["caller_storage_id"] == 77
        assert kwargs["caller_thread_id"] == 9

        payload = json.loads(kwargs["messages"][0].content)
        expected_override = str(Path("~/thread-override").expanduser())
        assert payload["current_working_dir"] == expected_override

    @pytest.mark.asyncio
    async def test_sub_agent_payload_falls_back_to_legacy_dir_without_thread(
        self, patched_db_provider: LocalStorage
    ) -> None:
        user = await patched_db_provider.get_or_create_user(user_id=5)
        legacy = user.working_dir

        provider_mock = MagicMock()
        provider_mock.get_chat_response = AsyncMock(
            return_value=(ChatResponseSchema(answer="ok", provider="OpenAI", model="m", usage=None), [])
        )

        with patch.object(User, "providers", new_callable=PropertyMock) as providers_prop:
            providers_prop.return_value = _StubProviders(provider_mock)
            await get_sub_agent_response(user_id=5, prompt="p", model_name="m", provider_name="OpenAI")

        payload = json.loads(provider_mock.get_chat_response.await_args.kwargs["messages"][0].content)
        assert payload["current_working_dir"] == legacy


class TestCallFunctionsContextInjection:
    @pytest.fixture
    def probe_tool(self):
        received: dict[str, Any] = {}

        class ProbeTool(ChibiTool):
            register = False
            name = "__test_probe_cwd"
            definition = {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
            }

            @classmethod
            async def function(cls, **kwargs: Any) -> dict[str, str]:
                received.update(kwargs)
                return {"status": "ok"}

        RegisteredChibiTools.tools_map[ProbeTool.name] = cast(Any, ProbeTool)
        yield received
        RegisteredChibiTools.tools_map.pop("__test_probe_cwd", None)

    @pytest.mark.asyncio
    async def test_caller_ids_injected_into_tool_payload(self, probe_tool: dict[str, Any]) -> None:
        calls = [ToolCallSchema(tool_name="__test_probe_cwd", args={"custom": "value"})]
        self_stub = SimpleNamespace()

        results = await Provider.call_functions(
            cast(Any, self_stub),
            calls=calls,
            caller_model="test-model",
            caller_provider="Test",
            user_id=5,
            interface=None,
            caller_storage_id=11,
            caller_thread_id=42,
        )

        assert results[0].status == "ok"
        assert probe_tool["caller_storage_id"] == 11
        assert probe_tool["caller_thread_id"] == 42
        assert probe_tool["user_id"] == 5
        assert probe_tool["custom"] == "value"

    @pytest.mark.asyncio
    async def test_interface_overrides_caller_ids(self, probe_tool: dict[str, Any]) -> None:
        interface = SimpleNamespace(storage_id=100, thread_id=200)
        calls = [ToolCallSchema(tool_name="__test_probe_cwd", args={})]

        await Provider.call_functions(
            cast(Any, SimpleNamespace()),
            calls=calls,
            caller_model="m",
            caller_provider="p",
            user_id=100,
            interface=cast(Any, interface),
            caller_storage_id=999,
            caller_thread_id=888,
        )

        assert probe_tool["caller_storage_id"] == 100
        assert probe_tool["caller_thread_id"] == 200


class TestSetWorkingDirTool:
    @pytest.mark.asyncio
    async def test_set_writes_current_thread_slot_only(self, patched_db_provider: LocalStorage) -> None:
        result = await SetWorkingDirTool.function(
            new_wd="~/project-x",
            user_id=3,
            caller_model="m",
            caller_provider="p",
            caller_storage_id=3,
            caller_thread_id=17,
        )

        assert result == {"status": "ok"}
        user = await patched_db_provider.get_or_create_user(user_id=3)
        expected = str(Path("~/project-x").expanduser())
        assert user.thread_working_dirs[17] == expected
        assert 18 not in user.thread_working_dirs

    @pytest.mark.asyncio
    async def test_set_raises_without_session_context(self) -> None:
        with pytest.raises(ValueError, match="session context"):
            await SetWorkingDirTool.function(new_wd="/tmp/x", user_id=3)


class TestGetCurrentWorkingDirTool:
    @pytest.mark.asyncio
    async def test_get_returns_effective_thread_cwd(self, patched_db_provider: LocalStorage) -> None:
        legacy = (await patched_db_provider.get_or_create_user(user_id=4)).working_dir
        await SetWorkingDirTool.function(new_wd="/thread/dir", user_id=4, caller_storage_id=4, caller_thread_id=2)

        cwd = await GetCurrentWorkingDirTool.function(user_id=4, caller_storage_id=4, caller_thread_id=2)
        other_thread = await GetCurrentWorkingDirTool.function(user_id=4, caller_storage_id=4, caller_thread_id=99)

        assert cwd == {"cwd": "/thread/dir"}
        assert other_thread == {"cwd": legacy}

    @pytest.mark.asyncio
    async def test_get_raises_without_session_context(self) -> None:
        with pytest.raises(ValueError, match="session context"):
            await GetCurrentWorkingDirTool.function(user_id=4)


class TestTerminalFallback:
    @pytest.mark.asyncio
    async def test_default_cwd_resolves_thread_override(self, patched_db_provider: LocalStorage) -> None:
        await set_thread_working_dir(user_id=6, thread_id=31, new_wd="/thread/terminal")

        moderator = MagicMock()
        moderator.name = "Moderator"
        moderator.moderate_command = AsyncMock(return_value=ModeratorsAnswer(verdict="approved", reason="ok"))

        process = MagicMock()
        process.pid = 123
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b"", b""))

        fake_moderation_provider = AsyncMock(return_value=moderator)
        with (
            patch("chibi.services.providers.tools.cmd.get_moderation_provider", fake_moderation_provider),
            patch("asyncio.create_subprocess_shell", AsyncMock(return_value=process)) as spawn_mock,
        ):
            result = await RunCommandInTerminalTool.function(
                cmd="true",
                user_id=6,
                caller_model="m",
                caller_provider="p",
                caller_storage_id=6,
                caller_thread_id=31,
            )

        assert result["return_code"] == 0
        assert spawn_mock.await_args is not None
        assert spawn_mock.await_args.kwargs["cwd"] == "/thread/terminal"


class TestThreadCloneCopiesDirs:
    @pytest.mark.asyncio
    async def test_clone_copies_thread_working_dirs(self, tmp_path: Path) -> None:
        from chibi.services.user import clone_thread_messages

        db = LocalStorage(storage_path=str(tmp_path))
        await set_thread_working_dir.__wrapped__(db, user_id=8, thread_id=10, new_wd="/clone-me")

        cloned_messages = await clone_thread_messages.__wrapped__(db, storage_id=8, old_thread_id=10, new_thread_id=20)

        user = await db.get_or_create_user(user_id=8)
        assert cloned_messages == 0
        assert user.thread_working_dirs.get(20) == "/clone-me"
