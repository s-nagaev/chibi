"""Concurrency and racing proof of the compaction chain.

Per-task suites cover the single-threaded behaviour; this module pins what no
single task owns: compactions never overlap inside one thread, repeated
replacement is idempotent, the volatile note is never duplicated across a
retry, the guard and the recovery never both compact one turn, a concurrent
history reset cannot destroy a compaction, and the two history-forking paths
invalidate the cached prompt size. Storage is the real local backend; the
provider SDK clients are mocked at the wire boundary.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Coroutine, cast
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.config.gpt import gpt_settings
from chibi.exceptions import ContextLengthExceededError
from chibi.models import Message, User
from chibi.runners.ide_transport import IDE_STORAGE_ID, IDEStdioRunner
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.bot import handle_clone_thread
from chibi.services.context_counters import ContextDecisionCounters
from chibi.services.context_measurement import SizeProvenance, estimate_messages_tokens
from chibi.services.context_note import CONTEXT_NOTE_OPEN
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import (
    COMPACTION_SEED_SOURCE,
    CompactionOutcome,
    CompactionResult,
    emergency_summarization,
    get_llm_chat_completion_answer,
    replace_thread_history,
    reset_chat_history,
)
from chibi.storage.local import LocalStorage
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

TEST_TOKEN = "test-token-for-compaction-concurrency"
WIRE_MODEL = "kimi-k2.6"
WINDOW_MODEL = "gpt-4o"
RESERVATION = 32000
SUMMARY_TEXT = "COLLAPSED SUMMARY OF THE THREAD"
SEED_QUESTION = "What we were talking about?"
COMPRESSED_SUFFIX = "_(Context was compressed to stay within limits.)_"
UID_SINGLE_FLIGHT = 921
UID_REPLACE = 922
UID_GUARD_TRUNCATION = 923
UID_RESET_RACE = 924
UID_TELEGRAM_CLONE = 925


def _dialogue(content_chars: int, messages: int) -> list[Message]:
    """Build a plain alternating history of a controlled heuristic size.

    Args:
        content_chars: Characters per message content.
        messages: Number of messages to build.

    Returns:
        The message list, oldest first.
    """
    roles = ["user", "assistant"]
    return [Message(role=roles[index % 2], content=f"turn {index} " + "x" * content_chars) for index in range(messages)]


async def _seed_history(storage: LocalStorage, user_id: int, thread_id: int, history: list[Message]) -> None:
    """Persist a whole history into real storage in one save.

    Args:
        storage: The real storage backend.
        user_id: The storage key of the conversation.
        thread_id: The message thread identifier (0 for the main thread).
        history: The messages to store, oldest first.
    """
    user = await storage.get_or_create_user(user_id=user_id)
    if thread_id:
        user.thread_messages_map[thread_id] = list(history)
    else:
        user.messages = list(history)
    await storage.save_user(user)


async def _stored(storage: LocalStorage, user_id: int, thread_id: int = 0) -> list[Message]:
    """Read a thread's working history straight from real storage.

    Args:
        storage: The real storage backend.
        user_id: The storage key of the conversation.
        thread_id: The message thread identifier.

    Returns:
        The stored conversation messages.
    """
    user = await storage.get_or_create_user(user_id=user_id)
    return await storage.get_conversation_messages(user=user, thread_id=thread_id)


def _stub_user(provider: Any, model: str) -> Mock:
    """Build the user double the full answer path resolves its turn from.

    Args:
        provider: The provider the user's active LLM selection resolves to.
        model: The active model name the user reports.

    Returns:
        The user double.
    """
    user = Mock()
    user.id = 1
    user.providers = Mock()
    user.providers.chat_ready = {}
    user.stt_provider = None
    user.thread_usage = {}
    user.get_active_llm_provider = Mock(return_value=provider)
    user.get_active_llm_model = Mock(return_value=model)
    return user


def _stub_db(user: Mock, history: list[Message]) -> AsyncMock:
    """Build the database double the full answer path reads history from.

    Args:
        user: The user the database resolves.
        history: The conversation history the database returns.

    Returns:
        The database double.
    """
    db = AsyncMock()
    db.get_or_create_user = AsyncMock(return_value=user)
    db.get_conversation_messages = AsyncMock(return_value=list(history))
    db.add_message = AsyncMock()
    return db


def _full_path_interface(storage_id: int, thread_id: int = 0) -> Any:
    """Build the minimal interface the full answer path needs.

    Args:
        storage_id: The storage key of the conversation.
        thread_id: The message thread identifier.

    Returns:
        The interface double, cast past the protocol type for the call.
    """
    interface = Mock()
    interface.storage_id = storage_id
    interface.thread_id = thread_id
    interface.user_data = {"first_name": "Tester"}
    interface.editor_context = None
    return interface


def _completion(answer: str, prompt_tokens: int | None = None) -> ChatCompletion:
    """Build a ChatCompletion carrying a final answer and optional usage.

    Args:
        answer: Assistant message content.
        prompt_tokens: Reported prompt tokens, or None for a usage-less response.

    Returns:
        The SDK response object the mocked client returns.
    """
    usage = None
    if prompt_tokens is not None:
        usage = CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=5, total_tokens=prompt_tokens + 5)
    return ChatCompletion(
        id="test-id",
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content=answer), finish_reason="stop")
        ],
        created=1234567890,
        model=WIRE_MODEL,
        object="chat.completion",
        usage=usage,
    )


class SingleFlightProbeProvider:
    """Summarizer provider double that detects two compactions in flight.

    The fake provider call yields to the event loop several times, so a second
    compaction whose lock is missing overlaps deterministically and is counted.
    """

    def __init__(self) -> None:
        """Initialize the empty flight state."""
        self.in_flight = False
        self.calls = 0
        self.overlaps = 0
        self._get_max_tokens_value = Mock(return_value=RESERVATION)

    async def get_chat_response(self, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        """Serve one summarizer call and count any concurrent second call.

        Args:
            kwargs: The summarizer call arguments (ignored).

        Returns:
            A canned summary response.
        """
        if self.in_flight:
            self.overlaps += 1
        self.in_flight = True
        try:
            for _ in range(5):
                await asyncio.sleep(0)
        finally:
            self.in_flight = False
        self.calls += 1
        usage = UsageSchema(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return (ChatResponseSchema(answer=SUMMARY_TEXT, provider="fake", model=WINDOW_MODEL, usage=usage), [])


class OverflowThenSummarizeProvider:
    """Active provider double serving the recovery sequence with a reset gate.

    The first turn call raises the overflow, the summarizer call gates on an
    event so a concurrent coroutine deterministically runs while the compaction
    holds the lock, and the retry call answers. The summarizer call is
    recognized by its dedicated system prompt, exactly as the machine sends it.
    """

    def __init__(self, reset_now: asyncio.Event, reset_done: asyncio.Event) -> None:
        """Bind the gate events the reset coroutine synchronizes on.

        Args:
            reset_now: Set when the summarizer call begins.
            reset_done: The summarizer call waits until this is set.
        """
        self._reset_now = reset_now
        self._reset_done = reset_done
        self._overflowed = False
        self.calls: list[str] = []
        self.name = "fake"
        self._get_max_tokens_value = Mock(return_value=RESERVATION)

    async def get_chat_response(self, **kwargs: Any) -> tuple[ChatResponseSchema, list[Message]]:
        """Serve one call of the overflow -> summarize -> retry sequence.

        Args:
            kwargs: The provider call arguments; ``system_prompt`` tells the summarizer call apart.

        Returns:
            The canned response for the recognized phase.

        Raises:
            ContextLengthExceededError: On the first non-summarizer call.
        """
        system_prompt = kwargs.get("system_prompt") or ""
        if system_prompt.startswith("Summarize"):
            self.calls.append("summarize")
            self._reset_now.set()
            await self._reset_done.wait()
            usage = UsageSchema(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            return (ChatResponseSchema(answer=SUMMARY_TEXT, provider="fake", model=WINDOW_MODEL, usage=usage), [])
        if not self._overflowed:
            self._overflowed = True
            self.calls.append("overflow")
            raise ContextLengthExceededError(provider="fake", model=WINDOW_MODEL)
        self.calls.append("retry")
        answer = Message(role="assistant", content="recovered answer")
        usage = UsageSchema(prompt_tokens=30, completion_tokens=5, total_tokens=35)
        return (ChatResponseSchema(answer="recovered", provider="fake", model=WINDOW_MODEL, usage=usage), [answer])


def _pop_singletons(*classes: type) -> None:
    """Drop singleton instances so a test never reuses another test's state.

    Args:
        classes: The singleton classes to clear.
    """
    for cls in classes:
        SingletonMeta._instances.pop(cls, None)


@pytest.fixture(autouse=True)
def _clean_process_singletons() -> Any:
    """Isolate the process-lifetime usage cache and decision counters per test."""
    UsageCacheStore()._data.clear()
    ContextDecisionCounters().reset()
    yield
    UsageCacheStore()._data.clear()
    ContextDecisionCounters().reset()


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """Provide a real local storage backend rooted in the test's temp directory.

    Args:
        tmp_path: Per-test temporary directory.

    Returns:
        The storage instance.
    """
    return LocalStorage(storage_path=str(tmp_path / "storage"))


@pytest.fixture
def local_db(storage: LocalStorage, monkeypatch: pytest.MonkeyPatch) -> LocalStorage:
    """Route every ``@inject_database`` call to the test's real storage.

    Args:
        storage: The real storage instance.
        monkeypatch: Fixture restoring the patched provider afterwards.

    Returns:
        The storage the injected helpers now see.
    """
    provider = SimpleNamespace(get_database=AsyncMock(return_value=storage))
    monkeypatch.setattr("chibi.storage.database._db_provider", provider)
    return storage


async def test_concurrent_compactions_are_single_flight(local_db: LocalStorage, monkeypatch) -> None:
    """Two overlapping compactions of one thread serialize; the history ends as ONE seed.

    Both compactions run concurrently against real storage; the fake summarizer yields to
    the event loop mid-call, so a missing per-thread lock would deterministically interleave
    the two calls and be counted. The final history must be a single seed pair, never a
    doubled one.
    """
    monkeypatch.setattr(gpt_settings, "max_tokens", RESERVATION)
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)
    await _seed_history(local_db, UID_SINGLE_FLIGHT, 0, _dialogue(200, 4))
    probe = SingleFlightProbeProvider()
    with (
        patch("chibi.models.User.get_active_llm_provider", Mock(return_value=probe)),
        patch("chibi.models.User.get_active_llm_model", Mock(return_value=WINDOW_MODEL)),
    ):
        results = await asyncio.gather(
            emergency_summarization(storage_id=UID_SINGLE_FLIGHT, thread_id=0),
            emergency_summarization(storage_id=UID_SINGLE_FLIGHT, thread_id=0),
        )

    assert all(isinstance(result, CompactionResult) for result in results)
    assert probe.calls == 2
    assert probe.overlaps == 0
    stored = await _stored(local_db, UID_SINGLE_FLIGHT)
    assert len(stored) == 2
    assert [(message.role, message.source) for message in stored] == [
        ("user", COMPACTION_SEED_SOURCE),
        ("assistant", COMPACTION_SEED_SOURCE),
    ]


async def test_repeated_replacement_leaves_exactly_one_seed(local_db: LocalStorage, monkeypatch) -> None:
    """Applying replacement twice leaves exactly one seed, through both paths.

    The replacement helper applied twice to the same seed stores exactly the seed; the
    end-to-end compaction machine run twice in a row (the shape that raced in the old
    implementation) ends with a single seed pair, not four messages.
    """
    seed = [
        Message(id=501, role="user", content=SEED_QUESTION, source=COMPACTION_SEED_SOURCE),
        Message(id=502, role="assistant", content="seed summary", source=COMPACTION_SEED_SOURCE),
    ]
    for _ in range(2):
        await replace_thread_history.__wrapped__(local_db, storage_id=UID_REPLACE, thread_id=0, seed=seed)
    stored = await _stored(local_db, UID_REPLACE)
    # LocalStorage regenerates Message ids on read (the wire format excludes them), so the
    # single-seed property is asserted on content and source, not on identity.
    assert [message.content for message in stored] == [SEED_QUESTION, "seed summary"]
    assert all(message.source == COMPACTION_SEED_SOURCE for message in stored)

    monkeypatch.setattr(gpt_settings, "max_tokens", RESERVATION)
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)
    await _seed_history(local_db, UID_REPLACE + 1, 0, _dialogue(200, 4))
    provider = SingleFlightProbeProvider()
    with (
        patch("chibi.models.User.get_active_llm_provider", Mock(return_value=provider)),
        patch("chibi.models.User.get_active_llm_model", Mock(return_value=WINDOW_MODEL)),
    ):
        for _ in range(2):
            result = await emergency_summarization(storage_id=UID_REPLACE + 1, thread_id=0)
    assert isinstance(result, CompactionResult)
    stored_after = await _stored(local_db, UID_REPLACE + 1)
    assert len(stored_after) == 2
    assert all(message.source == COMPACTION_SEED_SOURCE for message in stored_after)
    assert provider.calls == 2


async def test_retry_does_not_duplicate_the_volatile_block(local_db: LocalStorage, monkeypatch) -> None:
    """The retry of an overflowed turn carries exactly one note, and never inherits one.

    The full answer path runs to the real wire provider: the first attempt is rejected as
    oversized, the reactive recovery compacts and retries once. Both captured wire payloads
    must carry exactly one context note, and the Message objects both attempts converted
    from must be note-free, so the second attempt cannot inherit the first's block.
    """
    create_mock = AsyncMock(
        side_effect=[
            ContextLengthExceededError(provider="MoonshotAI", model=WIRE_MODEL),
            _completion("recovered answer", prompt_tokens=100),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    history = [Message(role="user", content="old turn"), Message(role="assistant", content="older answer")]
    provider = MoonshotAI(token=TEST_TOKEN)
    user = _stub_user(provider, WIRE_MODEL)
    db = _stub_db(user, history)
    compaction = AsyncMock(return_value=CompactionResult(outcome=CompactionOutcome.PREDICTED_FITS))
    monkeypatch.setattr("chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=db)))
    monkeypatch.setattr("chibi.services.providers.provider.prepare_system_prompt", AsyncMock(return_value="{}"))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=None))
    monkeypatch.setattr("chibi.services.user.emergency_summarization", compaction)
    monkeypatch.setattr(
        "chibi.services.user.measure_request_size_with_provenance",
        AsyncMock(side_effect=[(1000, SizeProvenance.ESTIMATED), (100, SizeProvenance.ESTIMATED)]),
    )
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
    monkeypatch.setattr(gpt_settings, "reactive_context_recovery", True)
    with patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=client):
        result = await get_llm_chat_completion_answer(
            storage_id=1, interface=_full_path_interface(1), user_text_message="hello"
        )

    assert result.answer.endswith(COMPRESSED_SUFFIX)
    assert create_mock.await_count == 2
    for call in create_mock.await_args_list:
        payload = json.dumps(call.kwargs["messages"], default=str)
        assert payload.count(CONTEXT_NOTE_OPEN) == 1
    assert all(CONTEXT_NOTE_OPEN not in message.content for message in history)


async def test_overflow_turn_runs_exactly_one_compaction(local_db: LocalStorage, monkeypatch) -> None:
    """An overflowed turn compacts at most once: the recovery owns the only attempt.

    The guard never compacts (its window is unknown here), the recovery attempts exactly one
    compaction, and a failed recovery surfaces the original error instead of compacting
    again. The wire provider is called exactly once: no doomed second attempt.
    """
    create_mock = AsyncMock(side_effect=ContextLengthExceededError(provider="MoonshotAI", model=WIRE_MODEL))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock)))
    history = [Message(role="user", content="old turn"), Message(role="assistant", content="older answer")]
    provider = MoonshotAI(token=TEST_TOKEN)
    user = _stub_user(provider, WIRE_MODEL)
    db = _stub_db(user, history)
    compaction = AsyncMock(return_value=None)
    monkeypatch.setattr("chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=db)))
    monkeypatch.setattr("chibi.services.providers.provider.prepare_system_prompt", AsyncMock(return_value="{}"))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=None))
    monkeypatch.setattr("chibi.services.user.emergency_summarization", compaction)
    monkeypatch.setattr(
        "chibi.services.user.measure_request_size_with_provenance",
        AsyncMock(side_effect=[(1000, SizeProvenance.ESTIMATED), (1000, SizeProvenance.ESTIMATED)]),
    )
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
    monkeypatch.setattr(gpt_settings, "reactive_context_recovery", True)
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=client),
        pytest.raises(ContextLengthExceededError),
    ):
        await get_llm_chat_completion_answer(storage_id=1, interface=_full_path_interface(1), user_text_message="hello")

    assert create_mock.await_count == 1
    compaction.assert_awaited_once()
    counters = ContextDecisionCounters().snapshot()
    assert counters.get("recovery_overflow_caught") == 1
    assert counters.get("recovery_failed_not_reduced") == 1


async def test_guard_truncation_triggers_no_compaction(local_db: LocalStorage, monkeypatch) -> None:
    """A turn the pre-call guard can shrink never reaches the compaction machine.

    The window resolves and the request is oversized, so the guard truncates transiently and
    the turn is served: zero compactions, the shrunk request strictly smaller than the full
    one, and the guard's decision counted as a truncation. The truncation mutates the
    transient request objects in place by design (the sent list keeps them); the test pins
    the transience where it is observable: a fresh read of the stored history afterwards
    returns the seeded messages byte-identical. LocalStorage re-materializes every read
    from disk, which is what makes the truncation transient; a backend that re-served the
    same Message objects across reads would leak the truncation into this assertion.
    """
    history = _dialogue(2000, 2)
    seeded_contents = [message.content for message in history]
    await _seed_history(local_db, UID_GUARD_TRUNCATION, 0, history)
    provider = AsyncMock()
    provider.name = "fake"
    provider._get_max_tokens_value = Mock(return_value=200)
    usage = UsageSchema(prompt_tokens=50, completion_tokens=5, total_tokens=55)
    provider.get_chat_response = AsyncMock(
        return_value=(ChatResponseSchema(answer="plain answer", provider="fake", model=WINDOW_MODEL, usage=usage), [])
    )
    monkeypatch.setattr(User, "get_active_llm_provider", Mock(return_value=provider))
    monkeypatch.setattr(User, "get_active_llm_model", Mock(return_value=WINDOW_MODEL))
    compaction = AsyncMock(return_value=None)
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=1000))
    monkeypatch.setattr("chibi.services.user.emergency_summarization", compaction)
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

    result = await get_llm_chat_completion_answer(
        storage_id=UID_GUARD_TRUNCATION, interface=_full_path_interface(UID_GUARD_TRUNCATION), user_text_message="hello"
    )

    assert result.answer == "plain answer"
    provider.get_chat_response.assert_awaited_once()
    sent = provider.get_chat_response.await_args.kwargs["messages"]
    full_request = [*history, sent[-1]]
    assert estimate_messages_tokens(sent) < estimate_messages_tokens(full_request)
    stored = await _stored(local_db, UID_GUARD_TRUNCATION)
    assert [message.content for message in stored[: len(history)]] == seeded_contents
    compaction.assert_not_awaited()
    assert ContextDecisionCounters().snapshot().get("context_guard_truncated") == 1


async def test_concurrent_turn_and_reset_keeps_the_seed(local_db: LocalStorage, monkeypatch) -> None:
    """A history reset racing a recovering turn cannot leave the thread without its seed.

    The reset fires deterministically while the compaction holds the thread lock (between
    the summarizer call and the replacement): the compaction's whole-thread replacement wins,
    so the thread ends holding the seed plus the recovered turn, never empty and never a
    half-seed.
    """
    reset_now = asyncio.Event()
    reset_done = asyncio.Event()
    provider = OverflowThenSummarizeProvider(reset_now, reset_done)
    monkeypatch.setattr(User, "get_active_llm_provider", Mock(return_value=provider))
    monkeypatch.setattr(User, "get_active_llm_model", Mock(return_value=WINDOW_MODEL))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=None))
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
    monkeypatch.setattr(gpt_settings, "reactive_context_recovery", True)
    await _seed_history(local_db, UID_RESET_RACE, 0, _dialogue(200, 2))

    async def _do_reset() -> None:
        """Run the concurrent history reset the moment the compaction starts writing."""
        await reset_now.wait()
        await reset_chat_history.__wrapped__(local_db, storage_id=UID_RESET_RACE, thread_id=0)
        reset_done.set()

    turn_coro = cast(
        "Coroutine[Any, Any, ChatResponseSchema]",
        get_llm_chat_completion_answer(
            storage_id=UID_RESET_RACE, interface=_full_path_interface(UID_RESET_RACE), user_text_message="hello"
        ),
    )
    turn_task = asyncio.create_task(turn_coro)
    reset_task = asyncio.create_task(_do_reset())
    result = await asyncio.wait_for(turn_task, 20)
    await reset_task

    assert provider.calls == ["overflow", "summarize", "retry"]
    assert result.answer.endswith(COMPRESSED_SUFFIX)
    stored = await _stored(local_db, UID_RESET_RACE)
    assert len(stored) == 4
    assert stored[0].content == SEED_QUESTION and stored[0].source == COMPACTION_SEED_SOURCE
    assert stored[1].content == SUMMARY_TEXT and stored[1].source == COMPACTION_SEED_SOURCE
    assert stored[3].content == "recovered answer"


async def test_telegram_thread_fork_invalidates_the_cached_prompt_size(local_db: LocalStorage, monkeypatch) -> None:
    """Cloning a Telegram thread invalidates the destination's cached prompt size."""
    store = UsageCacheStore()
    store._data[store._make_key(user_id=UID_TELEGRAM_CLONE, thread_id=555)] = 99999
    interface = Mock()
    interface.storage_id = UID_TELEGRAM_CLONE
    interface.thread_id = 0
    interface.create_thread = AsyncMock(return_value=555)
    interface.send_message = AsyncMock()

    await handle_clone_thread(interface=interface)

    interface.create_thread.assert_awaited_once()
    assert store.get(user_id=UID_TELEGRAM_CLONE, thread_id=555) is None
    sent = interface.send_message.await_args_list[0].kwargs["message"]
    assert "Thread cloned" in sent


async def test_ide_thread_clone_invalidates_the_cached_prompt_size(
    local_db: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloning a thread on the IDE stdio path invalidates the destination's cached prompt size."""
    monkeypatch.setattr("chibi.memory.chroma.task_manager", Mock())
    # The IDE runner owns its storage identity: the clone reads and writes
    # under IDE_STORAGE_ID, not under a Telegram user id.
    await _seed_history(local_db, IDE_STORAGE_ID, 7, _dialogue(50, 2))
    store = UsageCacheStore()
    store._data[store._make_key(user_id=IDE_STORAGE_ID, thread_id=555)] = 4242
    runner = IDEStdioRunner()
    interface = SimpleNamespace(thread_id=555)
    responses: list[str] = []

    await runner._handle_new_thread_with_current_context(cast(Any, interface), "7 cloned-thread", responses)

    assert store.get(user_id=IDE_STORAGE_ID, thread_id=555) is None
    assert len(responses) == 1
    assert "Thread cloned" in responses[0]
    cloned = await _stored(local_db, IDE_STORAGE_ID, 555)
    assert len(cloned) == 2
