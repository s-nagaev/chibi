"""Archive permanence: every compaction path leaves long-term memory untouched.

Compaction shrinks the working context; the ChromaDB archive is the assistant's
long-term memory and must survive it. These tests drive the four compaction
paths (``/reset``, ``summarize_history``, ``clear_tool_call_history`` and the
automatic state machine) against a real local-storage backend, the real
archival wrapper and a real embedded ChromaDB client whose embedding function
is a deterministic content hash, so no model is downloaded and semantic search
is reproducible.

They assert three properties: nothing pre-existing in the archive was deleted
or modified, messages archived before a compaction are still returned by the
model-facing search tool afterwards, and a search result is ordinary tool
traffic rather than a special case.
"""

import asyncio
import hashlib
import itertools
import json
from collections.abc import Coroutine, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from chromadb import EmbeddingFunction
from chromadb.api.types import Documents, Embeddings

from chibi.config import application_settings
from chibi.memory.abstract import MemorySearchResult
from chibi.memory.chroma import InternalChromaLongConversationMemory, with_chroma_archival
from chibi.models import FunctionSchema, Message, ToolSchema, User
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.providers.tools.memory import (
    ClearToolCallHistoryTool,
    SearchInConversationHistoryTool,
    SummarizeHistoryTool,
)
from chibi.services.providers.tools.schemas import ToolResponseSchema
from chibi.services.user import (
    COMPACTION_SEED_SOURCE,
    CompactionResult,
    emergency_summarization,
    reset_chat_history,
)
from chibi.storage.abstract import Database
from chibi.storage.local import LocalStorage
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

SEARCH_QUERY = "QUESTION-ALPHA about bicycles"
OLD_ASSISTANT_MESSAGE = "ANSWER-BETA about bicycles"
OLD_LATE_USER_MESSAGE = "QUESTION-GAMMA about bicycles"
TOOL_CALL_CARRIER = "Calling the file reader"
TOOL_RESULT_PAYLOAD = "TOOL-RESULT-PAYLOAD"
BACKGROUND_TOOL_RESULT = "BACKGROUND-RESULT"
SUMMARY_TEXT = "COLLAPSED SUMMARY OF THE THREAD"
COMPACTION_QUESTION = "What we were talking about?"
ACTIVE_MODEL = "gpt-4.1"

ARCHIVED_DIALOGUE_COUNT = 3
BATCH_TOKEN_LIMIT = 5
EMBEDDING_DIMENSION = 16


class HashEmbeddingFunction(EmbeddingFunction):
    """Deterministic content-hash embeddings: no model download, stable ordering.

    An exact-document query embeds to the identical vector as the archived
    document, so the semantic top hit of a query is the message carrying that
    text — which is what makes the "old memory is still reachable" assertion
    sharp rather than incidental.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        """Store the embedding dimension.

        Args:
            dimension: Number of bytes of the digest used as the vector.
        """
        self._dimension = dimension

    @staticmethod
    def name() -> str:
        """Return the name chromadb persists for this embedding function.

        Returns:
            The embedding-function name.
        """
        return "test-content-hash"

    def get_config(self) -> dict[str, Any]:
        """Return the serializable configuration of this embedding function.

        Returns:
            The dimension the vectors are truncated to.
        """
        return {"dimension": self._dimension}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbeddingFunction":
        """Rebuild the embedding function from its persisted configuration.

        Args:
            config: Configuration produced by :meth:`get_config`.

        Returns:
            An equivalent embedding function.
        """
        return HashEmbeddingFunction(dimension=int(config["dimension"]))

    def __call__(self, input: Documents) -> Embeddings:
        """Embed documents as normalized digest bytes.

        Args:
            input: Documents to embed.

        Returns:
            One vector per document. The values are plain Python lists, which
            chromadb accepts at runtime; the cast only reconciles the declared
            array-based vector type with that runtime shape.
        """
        vectors = [
            [byte / 255.0 for byte in hashlib.sha256(document.encode()).digest()[: self._dimension]]
            for document in input
        ]
        return cast(Embeddings, vectors)


class RecordingTaskManager:
    """Stand-in for the fire-and-forget background task manager.

    The real manager schedules the archival coroutine on the running loop and
    returns immediately. This one does the same through ``asyncio.create_task``
    and keeps the handles, so a test can await the archival deterministically
    instead of racing it.
    """

    def __init__(self) -> None:
        """Initialize an empty task list."""
        self.tasks: list[asyncio.Task[None]] = []

    def run_task(
        self,
        coro: Coroutine[Any, Any, None],
        user_id: int,
        thread_id: int = 0,
        timeout: float | None = None,
    ) -> asyncio.Task[None]:
        """Schedule a coroutine and remember it.

        Args:
            coro: The coroutine to schedule.
            user_id: The user the task is linked to.
            thread_id: The thread the task is linked to.
            timeout: Unused here; mirrors the real signature.

        Returns:
            The scheduled task.
        """
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def drain(self) -> None:
        """Await every scheduled task, so the archive reflects the writes."""
        pending = list(self.tasks)
        self.tasks.clear()
        if pending:
            await asyncio.gather(*pending)


class ToolInterface:
    """The interface surface the three memory tools under test actually read."""

    def __init__(self, storage_id: int, thread_id: int) -> None:
        """Bind the session identity the tools resolve their thread from.

        Args:
            storage_id: The storage key of the conversation.
            thread_id: The message thread identifier.
        """
        self.storage_id = storage_id
        self.thread_id = thread_id


class ArchiveHarness:
    """Real local storage wrapped by the archival decorator, plus its real archive.

    Every write goes through the production path: the storage backend persists
    the message, the decorator schedules its archival, and the archive is a real
    embedded ChromaDB client. Nothing about archiving is mocked, so an assertion
    about the archive is an assertion about production behaviour.
    """

    def __init__(self, memory: InternalChromaLongConversationMemory, tasks: RecordingTaskManager, path: Path) -> None:
        """Build the wrapped storage for one test.

        Args:
            memory: The real archive the decorator will write to.
            tasks: The recording task manager standing in for the real one.
            path: Directory the local storage backend keeps its files in.
        """
        self.memory = memory
        self.tasks = tasks
        inner = LocalStorage(storage_path=str(path))
        self.db: Database = cast(Database, with_chroma_archival(cast(Any, memory))(cast(Any, inner)))

    async def seed(self, user_id: int, thread_id: int, messages: list[Message]) -> User:
        """Persist messages through the wrapped storage and await their archival.

        Args:
            user_id: The storage key of the conversation.
            thread_id: The message thread identifier.
            messages: Messages to append to the thread.

        Returns:
            The user the messages were written for.
        """
        user = await self.db.get_or_create_user(user_id=user_id)
        for message in messages:
            await self.db.add_message(user=user, message=message, thread_id=thread_id)
        await self.tasks.drain()
        return user

    async def stored(self, user_id: int, thread_id: int) -> list[Message]:
        """Read the thread's working history straight from storage.

        Args:
            user_id: The storage key of the conversation.
            thread_id: The message thread identifier.

        Returns:
            The stored conversation messages of the thread.
        """
        user = await self.db.get_or_create_user(user_id=user_id)
        return await self.db.get_conversation_messages(user=user, thread_id=thread_id)

    async def entries(self, user_id: int, thread_id: int) -> dict[str, tuple[str, str]]:
        """Snapshot every archived entry of a thread, keyed by vector id.

        The metadata is fingerprinted as sorted JSON so a changed timestamp,
        batch id or message position is visible as a difference, not hidden by
        dict ordering.

        Args:
            user_id: The storage key of the conversation.
            thread_id: The message thread identifier.

        Returns:
            Mapping of vector id to ``(document, metadata fingerprint)``.
        """
        collection = await self.memory._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return {
            str(entry_id): (document, json.dumps(metadata, sort_keys=True))
            for entry_id, document, metadata in zip(ids, documents, metadatas)
        }

    async def search(self, user_id: int, thread_id: int, query: str) -> list[MemorySearchResult]:
        """Search the thread's archive through the model-facing tool.

        Args:
            user_id: The storage key of the conversation.
            thread_id: The message thread identifier.
            query: The natural-language search query.

        Returns:
            The results the tool would hand to the model.
        """
        interface = ToolInterface(storage_id=user_id, thread_id=thread_id)
        with patch("chibi.services.providers.tools.memory.memory", self.memory):
            payload = await SearchInConversationHistoryTool.function(
                query=query, user_id=user_id, interface=cast(Any, interface)
            )
        return list(payload.get("results", []))


def build_history() -> list[Message]:
    """Build a stored history covering every message shape the archiver sees.

    Returns:
        Plain dialogue, an in-loop tool-call pair and a background
        tool-response blob. Only the plain dialogue is archived, on purpose:
        the shapes that are excluded stay excluded, and the test's pre-existing
        archive is therefore exactly the dialogue.
    """
    tool_call = ToolSchema(id="call-1", type="function", function=FunctionSchema(name="read_file", arguments="{}"))
    background_blob = json.dumps({"type": "tool response", "tool_name": "read_file", "result": BACKGROUND_TOOL_RESULT})
    return [
        Message(role="user", content=SEARCH_QUERY),
        Message(role="assistant", content=OLD_ASSISTANT_MESSAGE),
        Message(role="assistant", content=TOOL_CALL_CARRIER, tool_calls=[tool_call]),
        Message(role="tool", content=TOOL_RESULT_PAYLOAD, tool_call_id="call-1"),
        Message(role="user", content=background_blob),
        Message(role="user", content=OLD_LATE_USER_MESSAGE),
    ]


def summarizer_provider() -> Mock:
    """Build the fake active provider the automatic path summarises with.

    Returns:
        A provider whose ``get_chat_response`` is an AsyncMock returning a
        non-empty summary; its unresolvable ``_get_max_tokens_value`` makes the
        output reservation fall back to the configured default, exactly as a
        provider without a concrete cap does.
    """
    provider = Mock()
    provider.get_chat_response = AsyncMock(
        return_value=(
            ChatResponseSchema(answer=SUMMARY_TEXT, provider="test", model=ACTIVE_MODEL, usage=UsageSchema()),
            [],
        )
    )
    return provider


async def assert_entries_survived(
    harness: ArchiveHarness, user_id: int, thread_id: int, before: dict[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """Assert every pre-existing entry is still present and byte-identical.

    Args:
        harness: The test harness holding the real archive.
        user_id: The storage key of the conversation.
        thread_id: The message thread identifier.
        before: The snapshot taken before the compaction ran.

    Returns:
        The snapshot taken after the compaction ran.
    """
    after = await harness.entries(user_id=user_id, thread_id=thread_id)
    deleted = sorted(set(before) - set(after))
    modified = {
        entry_id: (before[entry_id], after[entry_id])
        for entry_id in set(before) & set(after)
        if after[entry_id] != before[entry_id]
    }
    assert deleted == [], "the compaction path deleted archived vectors"
    assert modified == {}, "the compaction path modified archived vectors"
    return after


def assert_old_memory_is_searchable(results: list[MemorySearchResult], before: dict[str, tuple[str, str]]) -> None:
    """Assert the search returned pre-compaction memory and nothing else.

    Args:
        results: The results the search tool returned.
        before: The archived entries that existed before the compaction.
    """
    contents = [result.content for result in results]
    assert SEARCH_QUERY in contents, "the pre-compaction message is no longer reachable by search"
    assert OLD_ASSISTANT_MESSAGE in contents, "the neighbouring pre-compaction batch is no longer reachable"
    returned_ids = {result.message_id for result in results}
    assert returned_ids <= set(before), "search returned archive entries that did not exist before the compaction"


@pytest.fixture(autouse=True)
def distinct_message_ids(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give every Message a distinct, strictly increasing id for the test's duration.

    The production ``Message.id`` default (``time.time_ns``, ``chibi/models.py:101``)
    has ~microsecond resolution on macOS, so two messages built back-to-back,
    including the compaction seed pair built internally by the code under test,
    often receive the SAME id, and ChromaDB then silently drops the second
    ``add`` carrying a duplicate id. That collision is a pre-existing production
    defect, out of scope for this suite; the archive-permanence properties under
    test only need ids that are distinct by construction, so the field's default
    factory is replaced with a deterministic counter and the model schema is
    rebuilt (the factory is compiled into the pydantic core schema, so patching
    the FieldInfo alone has no effect). This deliberately stops exercising the
    real id-generation path; the archive write path itself is untouched.
    """
    field = Message.model_fields["id"]
    counter = itertools.count(start=1)
    monkeypatch.setattr(field, "default_factory", lambda: next(counter))
    Message.model_rebuild(force=True)
    try:
        yield
    finally:
        monkeypatch.undo()
        Message.model_rebuild(force=True)


@pytest.fixture
def chroma_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the embedded archive at the test's temp directory.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture restoring the patched settings afterwards.
    """
    monkeypatch.setattr(application_settings, "local_data_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(application_settings, "batch_token_limit", BATCH_TOKEN_LIMIT)


@pytest.fixture
def archive(chroma_settings: None, tmp_path: Path) -> Iterator[InternalChromaLongConversationMemory]:
    """Yield a real embedded archive with a fresh client per test.

    The memory class is a singleton, so the registry is cleared around the test;
    otherwise a later test would silently reuse the first test's client, which
    points at a deleted directory.

    Args:
        chroma_settings: Ensures the storage path is patched before construction.
        tmp_path: Per-test temporary directory.

    Yields:
        The archive instance under test.
    """
    SingletonMeta._instances.pop(InternalChromaLongConversationMemory, None)
    memory = InternalChromaLongConversationMemory(embedding_function=HashEmbeddingFunction())
    yield memory
    SingletonMeta._instances.pop(InternalChromaLongConversationMemory, None)


@pytest.fixture
def harness(
    archive: InternalChromaLongConversationMemory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ArchiveHarness:
    """Yield the wrapped real storage together with a recording task manager.

    Args:
        archive: The real archive.
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture restoring the patched task manager afterwards.

    Returns:
        The harness the tests drive.
    """
    tasks = RecordingTaskManager()
    monkeypatch.setattr("chibi.memory.chroma.task_manager", tasks)
    return ArchiveHarness(memory=archive, tasks=tasks, path=tmp_path / "storage")


@pytest.fixture
def injected_db(harness: ArchiveHarness, monkeypatch: pytest.MonkeyPatch) -> Database:
    """Route every ``@inject_database`` call to the wrapped test storage.

    Without this the injected helpers would resolve the global DatabaseCache
    singleton, which builds a backend from the environment instead of the test's
    temporary directory.

    Args:
        harness: The harness holding the wrapped storage.
        monkeypatch: Fixture restoring the patched provider afterwards.

    Returns:
        The wrapped storage the injected helpers now see.
    """
    provider = SimpleNamespace(get_database=AsyncMock(return_value=harness.db))
    monkeypatch.setattr("chibi.storage.database._db_provider", provider)
    return harness.db


async def test_reset_drops_the_working_history_and_keeps_the_archive(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """``/reset`` empties the thread but leaves long-term memory intact."""
    user_id, thread_id = 701, 7
    await harness.seed(user_id, thread_id, build_history())
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == ARCHIVED_DIALOGUE_COUNT, "seeding did not archive the dialogue"

    await reset_chat_history.__wrapped__(injected_db, storage_id=user_id, thread_id=thread_id)

    assert await harness.stored(user_id=user_id, thread_id=thread_id) == [], "the reset did not clear the thread"
    await assert_entries_survived(harness, user_id=user_id, thread_id=thread_id, before=before)
    results = await harness.search(user_id=user_id, thread_id=thread_id, query=SEARCH_QUERY)
    assert_old_memory_is_searchable(results, before)


async def test_summarize_history_tool_replaces_the_thread_and_keeps_the_archive(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """The ``summarize_history`` tool's compaction is an addition to the archive."""
    user_id, thread_id = 702, 7
    await harness.seed(user_id, thread_id, build_history())
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == ARCHIVED_DIALOGUE_COUNT, "seeding did not archive the dialogue"
    interface = ToolInterface(storage_id=user_id, thread_id=thread_id)

    outcome = await SummarizeHistoryTool.function(summary=SUMMARY_TEXT, user_id=user_id, interface=cast(Any, interface))
    await harness.tasks.drain()

    assert outcome == {"status": "ok"}
    stored = await harness.stored(user_id=user_id, thread_id=thread_id)
    assert [(message.role, message.content) for message in stored] == [
        ("user", COMPACTION_QUESTION),
        ("assistant", SUMMARY_TEXT),
    ], "the tool did not replace the thread history with the seed"
    after = await assert_entries_survived(harness, user_id=user_id, thread_id=thread_id, before=before)
    added = sorted(after[entry_id][0] for entry_id in set(after) - set(before))
    assert added == sorted([COMPACTION_QUESTION, SUMMARY_TEXT]), "the archive changed by more than the seed"
    results = await harness.search(user_id=user_id, thread_id=thread_id, query=SEARCH_QUERY)
    assert_old_memory_is_searchable(results, before)


async def test_clear_tool_call_history_tool_keeps_the_archive(harness: ArchiveHarness, injected_db: Database) -> None:
    """The ``clear_tool_call_history`` tool's compaction is an addition to the archive."""
    user_id, thread_id = 703, 7
    await harness.seed(user_id, thread_id, build_history())
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == ARCHIVED_DIALOGUE_COUNT, "seeding did not archive the dialogue"
    interface = ToolInterface(storage_id=user_id, thread_id=thread_id)

    outcome = await ClearToolCallHistoryTool.function(user_id=user_id, interface=cast(Any, interface))
    await harness.tasks.drain()

    assert outcome == {"status": "ok"}
    stored = await harness.stored(user_id=user_id, thread_id=thread_id)
    assert [message.role for message in stored] == ["user", "assistant", "assistant", "user", "user"], (
        "the tool did not drop the tool-call traffic from the thread"
    )
    assert all(not message.tool_calls and not message.tool_call_id for message in stored)
    after = await assert_entries_survived(harness, user_id=user_id, thread_id=thread_id, before=before)
    assert set(before) < set(after), "the re-written history was not archived at all"
    results = await harness.search(user_id=user_id, thread_id=thread_id, query=SEARCH_QUERY)
    assert_old_memory_is_searchable(results, before)


async def test_automatic_compaction_replaces_the_thread_and_keeps_the_archive(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """The automatic state machine's compaction is an addition to the archive."""
    user_id, thread_id = 704, 7
    await harness.seed(user_id, thread_id, build_history())
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == ARCHIVED_DIALOGUE_COUNT, "seeding did not archive the dialogue"

    with patch("chibi.models.User.get_active_llm_provider", Mock(return_value=summarizer_provider())):
        result = await emergency_summarization.__wrapped__(injected_db, storage_id=user_id, thread_id=thread_id)
    await harness.tasks.drain()

    assert isinstance(result, CompactionResult)
    stored = await harness.stored(user_id=user_id, thread_id=thread_id)
    assert [(message.role, message.content) for message in stored] == [
        ("user", COMPACTION_QUESTION),
        ("assistant", SUMMARY_TEXT),
    ], "the state machine did not replace the thread history with the seed"
    assert {message.source for message in stored} == {COMPACTION_SEED_SOURCE}
    after = await assert_entries_survived(harness, user_id=user_id, thread_id=thread_id, before=before)
    added = sorted(after[entry_id][0] for entry_id in set(after) - set(before))
    assert added == sorted([COMPACTION_QUESTION, SUMMARY_TEXT]), "the archive changed by more than the seed"
    results = await harness.search(user_id=user_id, thread_id=thread_id, query=SEARCH_QUERY)
    assert_old_memory_is_searchable(results, before)


async def test_rearchiving_an_existing_id_neither_duplicates_nor_modifies(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """Re-writing a message that is already archived leaves its entry untouched.

    This is the mechanism behind property (a) on the backends that keep message
    ids: ChromaDB skips an ``add`` whose id already exists, so a compaction path
    that re-writes an existing message can neither duplicate its vector nor
    refresh its stored document or metadata.
    """
    user_id, thread_id = 705, 7
    message = Message(role="user", content=SEARCH_QUERY)
    await harness.seed(user_id, thread_id, [message])
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == 1

    await harness.seed(user_id, thread_id, [message])

    after = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert after == before


async def test_retention_window_is_the_only_deleter_and_the_check_detects_it(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """The integrity check can fail, the window governs retention, and search reads the archive.

    Retention cleanup is the only code path in the archive layer that removes
    entries, and it removes exactly what its window covers. Running it tight
    shows two things the compaction tests depend on: the assertion they rest on
    does report a deletion, and a search over an empty archive returns nothing —
    so the pre-compaction results they observe come from the archive itself and
    not from the live history.
    """
    user_id, thread_id = 707, 7
    await harness.seed(user_id, thread_id, build_history())
    before = await harness.entries(user_id=user_id, thread_id=thread_id)
    assert len(before) == ARCHIVED_DIALOGUE_COUNT

    await harness.memory.delete_old(retention_days=1)
    assert await harness.entries(user_id=user_id, thread_id=thread_id) == before

    await harness.memory.delete_old(retention_days=0)
    with pytest.raises(AssertionError, match="deleted archived vectors"):
        await assert_entries_survived(harness, user_id=user_id, thread_id=thread_id, before=before)
    assert await harness.search(user_id=user_id, thread_id=thread_id, query=SEARCH_QUERY) == []


async def test_search_never_leaves_the_thread_it_was_archived_in(
    harness: ArchiveHarness, injected_db: Database
) -> None:
    """Collections are per (user, thread), so another thread cannot reach this memory.

    This is the accepted scope of the archive as it stands, and the test pins it
    so a later reader does not mistake it for a regression of the compaction
    work: the thread-scoped collection is the whole reason a search can never
    cross threads.
    """
    user_id = 708
    await harness.seed(user_id, 7, build_history())

    assert await harness.entries(user_id=user_id, thread_id=8) == {}
    assert await harness.search(user_id=user_id, thread_id=8, query=SEARCH_QUERY) == []
    assert await harness.search(user_id=user_id, thread_id=7, query=SEARCH_QUERY) != []


async def test_search_result_is_ordinary_tool_traffic(harness: ArchiveHarness, injected_db: Database) -> None:
    """The search payload is a plain JSON tool result, like every other tool's."""
    user_id, thread_id = 706, 7
    await harness.seed(user_id, thread_id, build_history())
    interface = ToolInterface(storage_id=user_id, thread_id=thread_id)

    with patch("chibi.services.providers.tools.memory.memory", harness.memory):
        response = await SearchInConversationHistoryTool._get_tool_call_result(
            query=SEARCH_QUERY, user_id=user_id, interface=cast(Any, interface)
        )

    assert isinstance(response, ToolResponseSchema)
    assert response.tool_name == "search_in_conversation_history"
    assert response.status == "ok"
    result = cast(dict[str, Any], response.result)
    results = cast(list[MemorySearchResult], result["results"])
    contents = [entry.content for entry in results]
    assert result["count"] == len(contents)
    assert set(contents) == {SEARCH_QUERY, OLD_ASSISTANT_MESSAGE}
    serialized = json.loads(response.model_dump_json())
    assert serialized["result"]["count"] == result["count"]
    assert serialized["result"]["results"][0]["content"]
