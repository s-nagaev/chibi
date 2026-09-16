"""Tests for thread-scoped notes: User field, set_thread_notes service, update_notes tool, prompt injection."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from chibi.models import User
from chibi.services.interface import UserInterface
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.memory import THREAD_NOTES_MAX_LENGTH, UpdateNotesTool
from chibi.services.providers.tools.tool import RegisteredChibiTools
from chibi.services.providers.utils import prepare_system_prompt
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import set_thread_notes
from chibi.storage.local import LocalStorage


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


def _reset_usage_cache() -> None:
    """Clear the UsageCacheStore singleton between tests."""
    UsageCacheStore()._data.clear()


class TestUserThreadNotesModel:
    def test_default_empty_dict(self) -> None:
        """A freshly created user has no thread notes."""
        user = User(id=1)

        assert user.thread_notes == {}

    def test_old_persisted_blob_without_field_loads(self) -> None:
        """Users restored from records persisted before the field existed get the default dict."""
        blob: dict[str, Any] = User(id=1, info="person facts").model_dump()
        blob.pop("thread_notes")

        user = User(**blob)

        assert user.thread_notes == {}
        assert user.info == "person facts"


class TestSetThreadNotesService:
    @pytest.mark.asyncio
    async def test_full_replace_per_thread(self, patched_db_provider: LocalStorage) -> None:
        """Setting notes twice for the same thread fully replaces the stored text."""
        await set_thread_notes(user_id=1, thread_id=7, notes="first version\n- point A")
        await set_thread_notes(user_id=1, thread_id=7, notes="second version")

        user = await patched_db_provider.get_or_create_user(user_id=1)
        assert user.thread_notes == {7: "second version"}

    @pytest.mark.asyncio
    async def test_two_threads_isolated(self, patched_db_provider: LocalStorage) -> None:
        """Notes of different threads are stored independently."""
        await set_thread_notes(user_id=1, thread_id=1, notes="thread one notes")
        await set_thread_notes(user_id=1, thread_id=2, notes="thread two notes")
        await set_thread_notes(user_id=1, thread_id=1, notes="thread one updated")

        user = await patched_db_provider.get_or_create_user(user_id=1)
        assert user.thread_notes == {1: "thread one updated", 2: "thread two notes"}

    @pytest.mark.asyncio
    async def test_empty_string_clears_notes(self, patched_db_provider: LocalStorage) -> None:
        """An empty string is a valid replacement that clears the thread's notes."""
        await set_thread_notes(user_id=1, thread_id=9, notes="some notes")
        await set_thread_notes(user_id=1, thread_id=9, notes="")

        user = await patched_db_provider.get_or_create_user(user_id=1)
        assert user.thread_notes[9] == ""

    @pytest.mark.asyncio
    async def test_notes_persist_via_whole_user_save(self, patched_db_provider: LocalStorage) -> None:
        """Notes survive a fresh reload from storage (whole-User serialization)."""
        await set_thread_notes(user_id=42, thread_id=5, notes="durable notes")

        refreshed = await patched_db_provider.get_or_create_user(user_id=42)
        assert refreshed.thread_notes == {5: "durable notes"}


class TestUpdateNotesTool:
    @pytest.mark.asyncio
    async def test_sets_notes_for_caller_thread(self, patched_db_provider: LocalStorage) -> None:
        """The tool stores notes under the thread resolved from caller context."""
        result = await UpdateNotesTool.function(
            new_notes="project state: tests green",
            user_id=3,
            caller_model="m",
            caller_provider="p",
            caller_storage_id=3,
            caller_thread_id=17,
        )

        assert result == {"status": "ok"}
        user = await patched_db_provider.get_or_create_user(user_id=3)
        assert user.thread_notes == {17: "project state: tests green"}

    @pytest.mark.asyncio
    async def test_never_trusts_model_supplied_thread_id(self, patched_db_provider: LocalStorage) -> None:
        """A spoofed thread_id among model args is ignored — caller context decides the slot."""
        result = await cast(Any, UpdateNotesTool.function)(
            new_notes="attempt to hijack another thread",
            user_id=3,
            caller_storage_id=3,
            caller_thread_id=17,
            thread_id=999,  # model-supplied impostor
        )

        assert result == {"status": "ok"}
        user = await patched_db_provider.get_or_create_user(user_id=3)
        assert 999 not in user.thread_notes
        assert user.thread_notes[17] == "attempt to hijack another thread"

    @pytest.mark.asyncio
    async def test_interface_context_wins_over_caller_ids(self, patched_db_provider: LocalStorage) -> None:
        """When an interface travels with the call, its thread id is authoritative."""
        interface = cast(Any, SimpleNamespace(storage_id=100, thread_id=200))

        await UpdateNotesTool.function(
            new_notes="from interface",
            user_id=100,
            interface=interface,
            caller_storage_id=100,
            caller_thread_id=888,
        )

        user = await patched_db_provider.get_or_create_user(user_id=100)
        assert 888 not in user.thread_notes
        assert user.thread_notes[200] == "from interface"

    @pytest.mark.asyncio
    async def test_sub_agent_scopes_notes_to_parent_thread(self, patched_db_provider: LocalStorage) -> None:
        """Sub-agents receive the parent caller_thread_id and their notes land on the parent thread."""
        await UpdateNotesTool.function(
            new_notes="sub-agent findings",
            user_id=7,
            caller_storage_id=7,
            caller_thread_id=42,
        )

        user = await patched_db_provider.get_or_create_user(user_id=7)
        assert user.thread_notes == {42: "sub-agent findings"}

    @pytest.mark.asyncio
    async def test_oversized_notes_raise_tool_exception(self, patched_db_provider: LocalStorage) -> None:
        """Notes above the size cap are refused with a clear error naming the limit."""
        oversized = "x" * (THREAD_NOTES_MAX_LENGTH + 1)

        with pytest.raises(ToolException) as exc_info:
            await UpdateNotesTool.function(
                new_notes=oversized,
                user_id=3,
                caller_storage_id=3,
                caller_thread_id=17,
            )

        assert str(THREAD_NOTES_MAX_LENGTH) in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_oversized_notes_are_not_saved(self, patched_db_provider: LocalStorage) -> None:
        """A refused oversized call leaves the stored notes untouched."""
        with pytest.raises(ToolException):
            await UpdateNotesTool.function(
                new_notes="x" * (THREAD_NOTES_MAX_LENGTH + 1),
                user_id=3,
                caller_storage_id=3,
                caller_thread_id=17,
            )

        user = await patched_db_provider.get_or_create_user(user_id=3)
        assert user.thread_notes == {}

    @pytest.mark.asyncio
    async def test_notes_at_exact_limit_are_accepted(self, patched_db_provider: LocalStorage) -> None:
        """The cap is a strict upper bound: exactly 16000 characters pass."""
        result = await UpdateNotesTool.function(
            new_notes="y" * THREAD_NOTES_MAX_LENGTH,
            user_id=3,
            caller_storage_id=3,
            caller_thread_id=17,
        )

        assert result == {"status": "ok"}
        user = await patched_db_provider.get_or_create_user(user_id=3)
        assert user.thread_notes[17] == "y" * THREAD_NOTES_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_raises_without_session_context(self) -> None:
        """Without an interface or caller identity the tool cannot resolve a thread."""
        with pytest.raises(ValueError, match="session context"):
            await UpdateNotesTool.function(new_notes="orphan notes", user_id=3)

    @pytest.mark.asyncio
    async def test_raises_without_user_id(self) -> None:
        """user_id is injected server-side; its absence is an error."""
        with pytest.raises(ValueError, match="user_id"):
            await UpdateNotesTool.function(new_notes="notes", caller_storage_id=3, caller_thread_id=17)

    def test_registration_and_model_facing_definition(self) -> None:
        """The tool is registered, and its model-facing schema never mentions threads."""
        assert UpdateNotesTool.name == "update_notes"
        assert RegisteredChibiTools.tools_map.get("update_notes") is UpdateNotesTool

        definition = UpdateNotesTool.definition
        assert definition["type"] == "function"
        function = definition["function"]
        assert function["name"] == "update_notes"

        description = function["description"]
        assert "thread" not in description.lower()
        assert "parent" not in description.lower()
        assert "override" in description.lower()  # full-replace semantics are explicit

        params = function["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert "new_notes" in params["properties"]
        assert params["required"] == ["new_notes"]


class TestPromptThreadNotesInjection:
    @pytest.mark.asyncio
    async def test_notes_injected_as_last_payload_key(self) -> None:
        """Notes of the current thread are injected as the LAST key of the JSON payload."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="person facts", llm_skills={})
        user.thread_notes[1] = "my working notes"
        interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, interface)

        prompt = json.loads(prompt_json)
        assert prompt["thread_notes"] == "my working notes"
        assert list(prompt.keys())[-1] == "thread_notes"

    @pytest.mark.asyncio
    async def test_notes_key_absent_when_thread_has_no_notes(self) -> None:
        """No notes for the thread — the key is omitted entirely."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="", llm_skills={})
        interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, interface)

        prompt = json.loads(prompt_json)
        assert "thread_notes" not in prompt

    @pytest.mark.asyncio
    async def test_notes_key_absent_when_notes_are_empty_string(self) -> None:
        """An empty notes string is treated the same as absent — the key is omitted."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="", llm_skills={})
        user.thread_notes[1] = ""
        interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, interface)

        prompt = json.loads(prompt_json)
        assert "thread_notes" not in prompt

    @pytest.mark.asyncio
    async def test_other_thread_notes_are_not_leaked(self) -> None:
        """Notes belonging to a different thread never reach this thread's prompt."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="", llm_skills={})
        user.thread_notes[2] = "other thread secret"
        interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, interface)

        prompt = json.loads(prompt_json)
        assert "thread_notes" not in prompt
        assert "other thread secret" not in prompt_json

    @pytest.mark.asyncio
    async def test_user_info_still_present_alongside_notes(self) -> None:
        """Global user_info stays in the payload when thread notes are injected."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="person facts", llm_skills={})
        user.thread_notes[1] = "my working notes"
        interface = cast(UserInterface, SimpleNamespace(thread_id=1, uses_uploaded_file_storage=False))

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, interface)

        prompt = json.loads(prompt_json)
        assert prompt["user_info"] == "person facts"
        assert prompt["thread_notes"] == "my working notes"

    @pytest.mark.asyncio
    async def test_thread_id_param_resolves_notes_without_interface(self) -> None:
        """Sub-agent path: with no interface, the thread_id argument selects the notes."""
        _reset_usage_cache()
        user = User(id=1, working_dir="/tmp", info="", llm_skills={})
        user.thread_notes[9] = "parent thread notes"

        with (
            patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
            patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        ):
            prompt_json = await prepare_system_prompt("base", 1, None, thread_id=9)

        prompt = json.loads(prompt_json)
        assert prompt["thread_notes"] == "parent thread notes"
        assert list(prompt.keys())[-1] == "thread_notes"
