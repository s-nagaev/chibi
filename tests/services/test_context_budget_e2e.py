"""Cross-cutting end-to-end proof of the context-budget chain.

The per-task suites cover their local cases in isolation; this module pins the
whole chain the block promised at the boundaries a user actually crosses: the
compaction machine over real storage, the pre-call budget guard, the provider
entry point's volatile note, and the ChromaDB archive. Storage is the real
local backend (a fresh instance per test); the archive is a real embedded
ChromaDB with a deterministic content-hash embedding; only the provider SDK
clients are mocked at the wire boundary.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.config import application_settings
from chibi.config.gpt import gpt_settings
from chibi.exceptions import ContextLengthExceededError
from chibi.memory.abstract import MemorySearchResult
from chibi.memory.chroma import InternalChromaLongConversationMemory
from chibi.models import FunctionSchema, Message, ThreadUsageRecord, ToolSchema, User
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.context_counters import ContextDecisionCounters
from chibi.services.context_measurement import (
    SizeProvenance,
    estimate_messages_tokens,
    measure_request_size_with_provenance,
)
from chibi.services.context_note import CONTEXT_NOTE_OPEN, format_context_note
from chibi.services.failover import CooldownStore, FailoverModelRegistry
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.usage_cache import UsageCacheStore
from chibi.services.user import (
    COMPACTION_SEED_SOURCE,
    EMERGENCY_SUMMARIZATION_INPUT_BUDGET,
    CompactionOutcome,
    CompactionResult,
    _enforce_context_budget,
    emergency_summarization,
    get_llm_chat_completion_answer,
)
from chibi.storage.local import LocalStorage
from chibi.utils.app import SingletonMeta
from tests.test_chroma_archive_permanence import ArchiveHarness, HashEmbeddingFunction, RecordingTaskManager

pytestmark = pytest.mark.asyncio

TEST_TOKEN = "test-token-for-context-e2e"
WIRE_MODEL = "kimi-k2.6"
WINDOW_MODEL = "gpt-4o"
WINDOW = 128000
RESERVATION = 32000
SUMMARY_TEXT = "COLLAPSED SUMMARY OF THE THREAD"
UID = 901
CHROMA_UID = 905
CHROMA_THREAD = 7


def _pop_singletons(*classes: type) -> None:
    """Drop singleton instances so a test never reuses another test's state.

    Args:
        classes: The singleton classes to clear.
    """
    for cls in classes:
        SingletonMeta._instances.pop(cls, None)


def _dialogue(turn_chars: int, turns: int) -> list[Message]:
    """Build a plain user/assistant history of a controlled heuristic size.

    Args:
        turn_chars: Characters per message content.
        turns: Number of user/assistant turn pairs.

    Returns:
        The message list, oldest first.
    """
    messages: list[Message] = []
    for index in range(turns):
        messages.append(Message(role="user", content=f"turn {index} " + "u" * turn_chars))
        messages.append(Message(role="assistant", content=f"answer {index} " + "a" * turn_chars))
    return messages


async def _seed_history(storage: LocalStorage, user_id: int, thread_id: int, history: list[Message]) -> User:
    """Persist a whole history into real storage in one save.

    Args:
        storage: The real storage backend.
        user_id: The storage key of the conversation.
        thread_id: The message thread identifier (0 for the main thread).
        history: The messages to store, oldest first.

    Returns:
        The persisted user.
    """
    user = await storage.get_or_create_user(user_id=user_id)
    if thread_id:
        user.thread_messages_map[thread_id] = list(history)
    else:
        user.messages = list(history)
    await storage.save_user(user)
    return user


def _fake_chat_provider(
    answer: str = "answer", prompt_tokens: int | None = None, reservation: int = RESERVATION
) -> Any:
    """Build the fake active provider the compaction machine summarises with.

    Args:
        answer: Answer text the fake provider returns.
        prompt_tokens: Provider-reported prompt tokens in the returned usage, or None for no usage.
        reservation: The value the failed request's reservation path yields.

    Returns:
        The provider double (its ``get_chat_response`` records every call).
    """
    provider = AsyncMock()
    provider.name = "fake"
    provider._get_max_tokens_value = Mock(return_value=reservation)
    usage: UsageSchema | None = None
    if prompt_tokens is not None:
        usage = UsageSchema(prompt_tokens=prompt_tokens, completion_tokens=5, total_tokens=prompt_tokens + 5)
    provider.get_chat_response = AsyncMock(
        return_value=(ChatResponseSchema(answer=answer, provider="fake", model=WINDOW_MODEL, usage=usage), [])
    )
    return provider


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


@contextmanager
def _moonshot_wire(create_mock: AsyncMock, client: Any) -> Iterator[MoonshotAI]:
    """Run the real OpenAI-friendly entry point against a mocked SDK client.

    Args:
        create_mock: The AsyncMock standing in for ``client.chat.completions.create``.
        client: The fake client namespace the provider must resolve.

    Yields:
        The real provider instance whose wire payload the test inspects.
    """
    mock_settings = MagicMock(wraps=application_settings)
    mock_settings.is_influx_configured = False
    with (
        patch.object(MoonshotAI, "client", new_callable=PropertyMock, return_value=client),
        patch("chibi.services.providers.provider.application_settings", mock_settings),
        patch("chibi.services.providers.provider.prepare_system_prompt", new=AsyncMock(return_value="{}")),
    ):
        yield MoonshotAI(token=TEST_TOKEN)


def _wire_client(answer: str, prompt_tokens: int | None = None) -> tuple[AsyncMock, Any]:
    """Build the mocked SDK client for the wire-level entry point.

    Args:
        answer: The completion answer the mocked client returns.
        prompt_tokens: Reported prompt tokens, or None for a usage-less response.

    Returns:
        The create mock and the client namespace carrying it.
    """
    create_mock = AsyncMock(return_value=_completion(answer, prompt_tokens))
    client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))))
    return create_mock, client


def _wire_interface(thread_id: int = 0) -> Any:
    """Build the minimal interface the wire-level entry point needs.

    Args:
        thread_id: The message thread identifier the interface reports.

    Returns:
        The interface double, cast past the protocol type for the call.
    """
    return cast(Any, SimpleNamespace(thread_id=thread_id, uses_uploaded_file_storage=False, delta_emitted=False))


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


def _stub_user(provider: Any, model: str, thread_usage: dict[int, Any] | None = None) -> Mock:
    """Build the user double the full answer path resolves its turn from.

    Args:
        provider: The provider the user's active LLM selection resolves to.
        model: The active model name the user reports.
        thread_usage: The durable per-thread usage records the user carries.

    Returns:
        The user double.
    """
    user = Mock()
    user.id = UID
    user.providers = Mock()
    user.providers.chat_ready = {}
    user.stt_provider = None
    user.thread_usage = thread_usage if thread_usage is not None else {}
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


@pytest.fixture(autouse=True)
def _clean_process_singletons() -> Iterator[None]:
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


@pytest.fixture
def chroma_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the embedded archive at the test's temp directory.

    Args:
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture restoring the patched settings afterwards.
    """
    monkeypatch.setattr(application_settings, "local_data_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(application_settings, "batch_token_limit", 5)


@pytest.fixture
def chroma_archive(chroma_settings: None) -> Iterator[InternalChromaLongConversationMemory]:
    """Yield a real embedded archive with a fresh client per test.

    Args:
        chroma_settings: Ensures the storage path is patched before construction.

    Yields:
        The archive instance under test.
    """
    _pop_singletons(InternalChromaLongConversationMemory)
    memory = InternalChromaLongConversationMemory(embedding_function=HashEmbeddingFunction())
    yield memory
    _pop_singletons(InternalChromaLongConversationMemory)


@pytest.fixture
def chroma_harness(
    chroma_archive: InternalChromaLongConversationMemory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ArchiveHarness:
    """Yield real local storage wrapped by the real archival decorator.

    Args:
        chroma_archive: The real archive.
        tmp_path: Per-test temporary directory.
        monkeypatch: Fixture restoring the patched task manager afterwards.

    Returns:
        The harness holding the wrapped storage and the recording task manager.
    """
    tasks = RecordingTaskManager()
    monkeypatch.setattr("chibi.memory.chroma.task_manager", tasks)
    return ArchiveHarness(memory=chroma_archive, tasks=tasks, path=tmp_path / "storage")


@pytest.fixture
def chroma_db(chroma_harness: ArchiveHarness, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Route every ``@inject_database`` call to the archival-wrapped storage.

    Args:
        chroma_harness: The harness holding the wrapped storage.
        monkeypatch: Fixture restoring the patched provider afterwards.

    Returns:
        The wrapped storage the injected helpers now see.
    """
    provider = SimpleNamespace(get_database=AsyncMock(return_value=chroma_harness.db))
    monkeypatch.setattr("chibi.storage.database._db_provider", provider)
    return chroma_harness.db


async def test_summarizer_request_fits_the_model_window_end_to_end(local_db: LocalStorage, monkeypatch) -> None:
    """A history past the fit budget still yields a summarizer call that fits the real window.

    The measured request (~192k counted tokens) exceeds the fit budget (the 128000-token
    window of gpt-4o minus the 32000-token output reservation), so the machine must trim the
    input: the request the summarizer actually receives, plus its own output reservation,
    must fit the window. Asserted on the wire input the fake provider captured, not on the
    fit function in isolation.
    """
    history = _dialogue(turn_chars=6400, turns=60)
    await _seed_history(local_db, UID, 0, history)
    provider = _fake_chat_provider(answer=SUMMARY_TEXT)
    monkeypatch.setattr(User, "get_active_llm_provider", Mock(return_value=provider))
    monkeypatch.setattr(User, "get_active_llm_model", Mock(return_value=WINDOW_MODEL))
    monkeypatch.setattr(gpt_settings, "max_tokens", RESERVATION)
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 200000)

    result = await emergency_summarization(storage_id=UID, thread_id=0)

    assert isinstance(result, CompactionResult)
    assert result.outcome is CompactionOutcome.PREDICTED_FITS
    provider.get_chat_response.assert_awaited_once()
    sent = provider.get_chat_response.await_args.kwargs["messages"]
    assert len(sent) == 1 and sent[0].role == "user"
    input_tokens = estimate_messages_tokens(sent)
    assert input_tokens <= WINDOW - RESERVATION
    assert input_tokens + RESERVATION <= WINDOW
    assert input_tokens < estimate_messages_tokens(history)
    assert history[-1].content in sent[0].content
    assert provider.get_chat_response.await_args.kwargs["max_tokens"] == RESERVATION


async def test_fit_budget_uses_the_real_window_scale_not_the_fallback_share(
    local_db: LocalStorage, monkeypatch
) -> None:
    """A history fitting the real window input space reaches the summarizer whole.

    The history (~80k counted tokens) exceeds the documented fallback share of
    ``max_history_tokens`` (0.6 x 100000 = 60000) but fits the real scale (the 128000-token
    window minus the 32000-token reservation). A fit still sized on the old fallback scale
    would trim here; nothing is dropped, which is the relationship that distinguishes the
    two scales.
    """
    history = _dialogue(turn_chars=8000, turns=20)
    counted = estimate_messages_tokens(history)
    monkeypatch.setattr(gpt_settings, "max_history_tokens", 100000)
    assert counted > int(100000 * EMERGENCY_SUMMARIZATION_INPUT_BUDGET)
    assert counted <= WINDOW - RESERVATION
    await _seed_history(local_db, UID, 0, history)
    provider = _fake_chat_provider(answer=SUMMARY_TEXT)
    monkeypatch.setattr(User, "get_active_llm_provider", Mock(return_value=provider))
    monkeypatch.setattr(User, "get_active_llm_model", Mock(return_value=WINDOW_MODEL))
    monkeypatch.setattr(gpt_settings, "max_tokens", RESERVATION)

    result = await emergency_summarization(storage_id=UID, thread_id=0)

    assert isinstance(result, CompactionResult)
    sent_text = provider.get_chat_response.await_args.kwargs["messages"][0].content
    for message in history:
        assert message.content in sent_text
    assert provider.get_chat_response.await_args.kwargs["max_tokens"] == RESERVATION


async def test_measured_size_survives_a_restart_and_the_prediction_is_close(storage: LocalStorage, monkeypatch) -> None:
    """A durable per-thread measurement outlives the process and anchors a close prediction.

    The measurement is taken through the real provider entry point (a wire-level response
    reporting 100000 prompt tokens), persisted through the real storage, and then resolved
    by a FRESH storage instance over the same directory with the in-process usage cache
    cleared, which is the restart shape. Closeness: no reference tokenizer ships with the
    project, so the provider's report for the FOLLOW-UP turn cannot be observed locally.
    The prediction's appended part is therefore compared against a count computed here in
    the test from the raw appended strings (the same documented ``len // 4`` rule, applied
    by this code, not by calling the production estimator), on top of the provider-reported
    100000 base. The tolerance is 1 token per appended message, the exact divergence the
    shared rule permits (floor-division splitting of content and role counts), and it is
    load-bearing: any change to the estimator's arithmetic moves the prediction by more
    than that and fails this assertion, because the reference does not move with it. The
    estimate-only fallback is pinned below as the contrast: it sits orders of magnitude
    away, so the tolerance cannot be satisfied by a degraded figure.
    """
    provider = SimpleNamespace(get_database=AsyncMock(return_value=storage))
    monkeypatch.setattr("chibi.storage.database._db_provider", provider)
    measured_history = _dialogue(turn_chars=1000, turns=5)
    turn = Message(role="user", content="FRESH-USER-TURN after the measured request")
    create_mock, client = _wire_client("ok", prompt_tokens=100000)

    with _moonshot_wire(create_mock, client) as wire_provider:
        await wire_provider.get_chat_response(
            messages=list(measured_history),
            user=User(id=UID),
            caller_storage_id=UID,
            caller_thread_id=0,
            model=WIRE_MODEL,
            system_prompt="",
            interface=_wire_interface(),
            track_prompt_size=True,
        )
    assert create_mock.await_count == 1

    UsageCacheStore()._data.clear()
    restarted_storage = LocalStorage(storage_path=storage.storage_path)
    monkeypatch.setattr(
        "chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=restarted_storage))
    )
    restarted_user = await restarted_storage.get_or_create_user(user_id=UID)
    record = restarted_user.thread_usage[0]
    assert record.prompt_tokens == 100000
    assert record.message_count == len(measured_history)

    after_turn = [*measured_history, turn]
    predicted, provenance = await measure_request_size_with_provenance(
        user_id=UID, thread_id=0, model=WIRE_MODEL, messages=after_turn
    )
    assert provenance is SizeProvenance.PROVIDER_MEASURED
    appended_counted = estimate_messages_tokens([turn])
    assert predicted == 100000 + appended_counted
    # Independent reference for the closeness check: the appended delta counted HERE, from
    # the raw strings the test authored, under the same documented rule, never through the
    # production estimator, so a change in the estimator's arithmetic moves only one side.
    appended = after_turn[len(measured_history) :]
    independent_delta = sum((len(message.content) + len(message.role)) // 4 for message in appended)
    provider_report = 100000 + independent_delta
    tolerance = len(appended)  # one token per appended message of permitted floor-division divergence
    assert abs(predicted - provider_report) <= tolerance
    estimate_only = estimate_messages_tokens(after_turn)
    assert estimate_only < 0.3 * predicted


async def test_missing_entry_degrades_to_the_estimate_and_the_request_is_sent(
    local_db: LocalStorage, monkeypatch
) -> None:
    """No durable record: the request is measured on the estimate and sent unchanged.

    The guard must not block or trim on the absence of a measurement: the full answer path
    sends the request once, the guard counts a no-op, and the provenance of the size figure
    it acted on is the estimate.
    """
    history = _dialogue(turn_chars=200, turns=2)
    provider = _fake_chat_provider(answer="plain answer", prompt_tokens=50)
    user = _stub_user(provider, WINDOW_MODEL)
    db = _stub_db(user, history)
    monkeypatch.setattr("chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=db)))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=WINDOW))
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

    result = await get_llm_chat_completion_answer(
        storage_id=UID, interface=_full_path_interface(UID), user_text_message="hello"
    )

    assert result.answer == "plain answer"
    provider.get_chat_response.assert_awaited_once()
    sent = provider.get_chat_response.await_args.kwargs["messages"]
    assert len(sent) == len(history) + 1
    _, provenance = await measure_request_size_with_provenance(
        user_id=UID, thread_id=0, model=WINDOW_MODEL, messages=sent
    )
    assert provenance is SizeProvenance.ESTIMATED
    assert ContextDecisionCounters().snapshot().get("context_guard_noop") == 1


async def test_stale_entry_degrades_instead_of_mis_triggering(local_db: LocalStorage, monkeypatch) -> None:
    """A model-mismatched record degrades to the estimate and mis-triggers nothing.

    The stale record claims a 10M-token request measured on ANOTHER model: a wrong scale and
    a wrong tokenizer at once. The behaviour must match the missing-record case: the request
    is sent whole, nothing is trimmed, and no compaction is triggered by the stale value
    alone.
    """
    history = _dialogue(turn_chars=200, turns=2)
    stale = ThreadUsageRecord(prompt_tokens=10_000_000, model="another-model", message_count=len(history))
    provider = _fake_chat_provider(answer="plain answer", prompt_tokens=50)
    user = _stub_user(provider, WINDOW_MODEL, thread_usage={0: stale})
    db = _stub_db(user, history)
    compaction = AsyncMock(return_value=None)
    monkeypatch.setattr("chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=db)))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=WINDOW))
    monkeypatch.setattr("chibi.services.user.emergency_summarization", compaction)
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)

    result = await get_llm_chat_completion_answer(
        storage_id=UID, interface=_full_path_interface(UID), user_text_message="hello"
    )

    assert result.answer == "plain answer"
    provider.get_chat_response.assert_awaited_once()
    sent = provider.get_chat_response.await_args.kwargs["messages"]
    assert len(sent) == len(history) + 1
    assert estimate_messages_tokens(sent) < 10_000_000
    compaction.assert_not_awaited()
    _, provenance = await measure_request_size_with_provenance(
        user_id=UID, thread_id=0, model=WINDOW_MODEL, messages=sent
    )
    assert provenance is SizeProvenance.ESTIMATED
    assert ContextDecisionCounters().snapshot().get("context_guard_noop") == 1


async def test_overflow_is_not_failed_over_and_not_retried(local_db: LocalStorage, monkeypatch) -> None:
    """An overflow is neither failed over to a ready backup nor attempted twice.

    Both halves asserted: a backup provider sits in the user's registry (it would be
    consulted the moment the overflow were misclassified as a failover trigger) and is never
    awaited, and the primary receives exactly one attempt, so no retry of any kind ran. The
    error surfaces honestly as ContextLengthExceededError.
    """
    _pop_singletons(CooldownStore, FailoverModelRegistry)
    primary = AsyncMock()
    primary.name = "openai"
    primary._get_max_tokens_value = Mock(return_value=RESERVATION)

    async def _fail(**kwargs: Any) -> Any:
        raise ContextLengthExceededError(provider="openai", model=WINDOW_MODEL)

    primary.get_chat_response = AsyncMock(side_effect=_fail)
    backup = AsyncMock()
    backup.name = "gemini"
    backup._get_max_tokens_value = Mock(return_value=RESERVATION)
    backup.get_chat_response = AsyncMock(
        return_value=(ChatResponseSchema(answer="backup answer", provider="gemini", model=WINDOW_MODEL, usage=None), [])
    )
    user = _stub_user(primary, WINDOW_MODEL)
    user.providers.chat_ready = {
        "openai": type("OpenAIStub", (), {"name": "openai"}),
        "gemini": type("GeminiStub", (), {"name": "gemini"}),
    }
    user.providers.get_instance = lambda provider_class: {"openai": primary, "gemini": backup}[provider_class.name]
    db = _stub_db(user, _dialogue(turn_chars=200, turns=2))
    compaction = AsyncMock(return_value=None)
    mark_spy = AsyncMock()
    monkeypatch.setattr(CooldownStore, "mark", mark_spy)
    monkeypatch.setattr("chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=db)))
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=None))
    monkeypatch.setattr("chibi.services.user.emergency_summarization", compaction)
    monkeypatch.setattr(gpt_settings, "failover_chain_master_raw", "auto", raising=False)
    monkeypatch.setattr(gpt_settings, "reactive_context_recovery", True)

    with pytest.raises(ContextLengthExceededError):
        await get_llm_chat_completion_answer(
            storage_id=UID, interface=_full_path_interface(UID), user_text_message="hello"
        )

    assert primary.get_chat_response.await_count == 1
    backup.get_chat_response.assert_not_awaited()
    mark_spy.assert_not_awaited()
    compaction.assert_awaited_once()
    _pop_singletons(CooldownStore, FailoverModelRegistry)


async def test_truncation_keeps_tool_pairs_intact(local_db: LocalStorage, monkeypatch) -> None:
    """The last-resort shrink never separates an assistant tool call from its results.

    Driven through the pre-call guard end to end (real storage, real measurement, a real
    resolvable window scaled down via the window lookup). The scenario is sized so the
    oldest-unit drop tier and the content-truncation tier BOTH act on tool traffic: the
    opening user unit is dropped whole, and a tool result is content-truncated in place
    while its owning assistant message stays, so the pair rule is exercised by the shrink
    itself, not incidental to the scenario. The resulting list is asserted directly: every
    assistant ``tool_calls`` message keeps all of its ``role="tool"`` results, no orphan
    tool result remains, the most recent message survives, ids are never renumbered, and
    the shrink actually happened.
    """

    def _pair(call_id: str, result_text: str) -> list[Message]:
        assistant = Message(
            role="assistant",
            content=f"Calling tool {call_id}",
            tool_calls=[
                ToolSchema(id=call_id, type="function", function=FunctionSchema(name="read_file", arguments="{}"))
            ],
        )
        return [assistant, Message(role="tool", content=result_text, tool_call_id=call_id)]

    # Sizing (heuristic counts, window 2400 / reservation 600 / budget 1800 / excess ~520):
    # the opening unit (~254 tokens) is droppable whole, while each tool unit
    # (assistant carrier ~30 + result ~1000) overshoots the remaining excess and is left
    # for the truncation tier, which halves a tool result in place. If the pair rule were
    # broken into per-message units, the assistant carriers (~30 tokens each) would be
    # dropped in the same tier while their oversized results stayed behind, leaving
    # orphaned tool results that the assertions below reject.
    messages: list[Message] = [
        Message(role="user", content="opening turn " + "o" * 1000),
        *_pair("call_1", "r" * 4000),
        *_pair("call_2", "s" * 4000),
        Message(role="user", content="the current turn"),
    ]
    counted_before = estimate_messages_tokens(messages)
    provider = _fake_chat_provider(reservation=600)
    monkeypatch.setattr(
        "chibi.storage.database._db_provider", SimpleNamespace(get_database=AsyncMock(return_value=local_db))
    )
    monkeypatch.setattr("chibi.services.user.get_model_context_window", Mock(return_value=2400))

    result = await _enforce_context_budget(
        provider=provider, model=WINDOW_MODEL, storage_id=UID, thread_id=0, messages=messages
    )

    assert estimate_messages_tokens(result) < counted_before
    assert result[-1] is messages[-1]
    assert {message.id for message in result} <= {message.id for message in messages}
    assistants = {call.id: message for message in result if message.tool_calls for call in message.tool_calls}
    tool_results = [message for message in result if message.role == "tool"]
    assert tool_results, "the shrink removed all tool traffic: pair integrity cannot be asserted"
    for tool_result in tool_results:
        assert tool_result.tool_call_id in assistants
        assert assistants[tool_result.tool_call_id] in result
    for assistant in assistants.values():
        for call in assistant.tool_calls or []:
            assert any(tool_result.tool_call_id == call.id for tool_result in tool_results)
    # A truncated tool result must stay under its owner, not orphaned: the carrier whose
    # result was halved is still in the sent list.
    assert len(assistants) == 2


async def test_volatile_note_reaches_the_wire_but_never_storage_or_archive(
    chroma_harness: ArchiveHarness, chroma_db: Any
) -> None:
    """The note is attached to the wire payload and absent from stored history and the archive.

    A real provider entry point attaches the note (observed on the captured wire payload).
    The persistence half then writes the shape the strip-on-write boundary exists for: a
    caller holding the ATTACHED converted envelope (the re-persist/retry shape) writes the
    fresh turn back through the archival-wrapped real storage with the note still on its
    content. Stored history and the archive must end up byte-identical to the note-free
    turn: under a disabled strip the marker survives on both surfaces and these assertions
    fail. Message ids are explicit so the archival assertions never depend on Message.id
    distinctness.
    """
    harness = chroma_harness
    seeded = [
        Message(id=1, role="user", content="SEEDED QUESTION about bicycles"),
        Message(id=2, role="assistant", content="SEEDED ANSWER about bicycles"),
    ]
    await harness.seed(CHROMA_UID, CHROMA_THREAD, seeded)
    turn = Message(id=3, role="user", content="FRESH-USER-TURN about bicycles")
    create_mock, client = _wire_client("WIRE ANSWER", prompt_tokens=50)
    turn_messages = [*seeded, turn]

    with _moonshot_wire(create_mock, client) as provider:
        chat_response, new_messages = await provider.get_chat_response(
            messages=list(turn_messages),
            user=User(id=CHROMA_UID),
            caller_storage_id=CHROMA_UID,
            caller_thread_id=CHROMA_THREAD,
            model=WIRE_MODEL,
            system_prompt="base",
            interface=_wire_interface(CHROMA_THREAD),
        )
    assert chat_response.answer == "WIRE ANSWER"
    assert create_mock.await_args is not None
    wire_payload = create_mock.await_args.kwargs["messages"]
    assert json.dumps(wire_payload, default=str).count(CONTEXT_NOTE_OPEN) == 1
    assert all(CONTEXT_NOTE_OPEN not in message.content for message in turn_messages)

    # The attached converted envelope of the fresh turn, exactly as the entry point
    # produced it for the wire: a caller persisting that envelope carries the note.
    attached_envelope = wire_payload[-1]["content"]
    assert isinstance(attached_envelope, str)
    assert CONTEXT_NOTE_OPEN in attached_envelope
    carried = Message(id=3, role="user", content=attached_envelope)

    user = await chroma_db.get_or_create_user(user_id=CHROMA_UID)
    for message in [carried, *new_messages]:
        await chroma_db.add_message(user=user, message=message, thread_id=CHROMA_THREAD)
    await harness.tasks.drain()

    stored = await harness.stored(user_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    # The carried envelope is stored stripped: byte-identical to the note-free turn.
    assert [message.content for message in stored] == [
        seeded[0].content,
        seeded[1].content,
        turn.content,
        *[message.content for message in new_messages],
    ]
    assert all(CONTEXT_NOTE_OPEN not in message.content for message in stored)
    entries = await harness.entries(user_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    assert all(CONTEXT_NOTE_OPEN not in document for document, _metadata in entries.values())
    assert turn.content in [document for document, _metadata in entries.values()]


async def test_strip_on_write_keeps_a_carried_note_out_of_storage_and_the_archive(
    chroma_harness: ArchiveHarness, chroma_db: Any
) -> None:
    """The persistence boundary strips a note a caller carries on a Message.

    A message whose content carries the attached note (the shape a caller holding an
    attached converted envelope would persist) is written through the real wrapped storage:
    both the stored history and the archived document end up byte-identical to the
    pre-attach content. Content equality is the assertion; without the strip the marker
    would survive on both surfaces.
    """
    harness = chroma_harness
    original = "TURN CONTENT that a caller persisted together with an attached note"
    note = format_context_note(["approximate_context_size: 1 tokens"])
    carried = Message(id=11, role="user", content=original + "\n\n" + note)

    user = await chroma_db.get_or_create_user(user_id=CHROMA_UID + 1)
    await chroma_db.add_message(user=user, message=carried, thread_id=CHROMA_THREAD)
    await harness.tasks.drain()

    stored = await harness.stored(user_id=CHROMA_UID + 1, thread_id=CHROMA_THREAD)
    assert [message.content for message in stored] == [original]
    entries = await harness.entries(user_id=CHROMA_UID + 1, thread_id=CHROMA_THREAD)
    assert [document for document, _metadata in entries.values()] == [original]


async def test_seed_turn_still_receives_the_block() -> None:
    """The first turn after a compaction still gets the note, and the seed never does.

    The request history is the compaction seed pair plus the fresh turn: the real entry
    point must attach exactly one note, on the fresh user message only, leaving the
    persisted seed messages untouched.
    """
    seed_user = Message(role="user", content="What we were talking about?", source=COMPACTION_SEED_SOURCE)
    seed_assistant = Message(role="assistant", content=SUMMARY_TEXT, source=COMPACTION_SEED_SOURCE)
    fresh_turn = Message(role="user", content="And what happened next?")
    messages = [seed_user, seed_assistant, fresh_turn]
    create_mock, client = _wire_client("ok")

    with _moonshot_wire(create_mock, client) as provider:
        await provider.get_chat_response(
            messages=messages,
            user=User(id=UID),
            caller_storage_id=UID,
            caller_thread_id=0,
            model=WIRE_MODEL,
            system_prompt="base",
            interface=_wire_interface(),
        )

    assert create_mock.await_args is not None
    payload = create_mock.await_args.kwargs["messages"]
    serialized = json.dumps(payload, default=str)
    assert serialized.count(CONTEXT_NOTE_OPEN) == 1
    fresh_content = payload[-1]["content"]
    assert isinstance(fresh_content, str)
    assert CONTEXT_NOTE_OPEN in fresh_content
    assert CONTEXT_NOTE_OPEN not in payload[0]["content"]
    assert all(CONTEXT_NOTE_OPEN not in message.content for message in messages)


async def test_compaction_leaves_the_archive_searchable(
    chroma_harness: ArchiveHarness, chroma_db: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The automatic compaction machine leaves pre-compaction memory intact and searchable.

    The machine runs over real archival-wrapped storage; every entry archived before the
    compaction survives byte-identically and the pre-compaction messages are still returned
    by the model-facing search afterwards. Seed additions are deliberately not asserted: the
    compaction constructs its two seed messages back-to-back, their Message.id values can
    collide (macOS clock resolution) and the archive silently deduplicates, so asserting on
    them would make this test flaky in exactly the way the block's own archive suite is.
    """
    harness = chroma_harness
    seeded = [
        Message(id=1, role="user", content="QUESTION-ALPHA about bicycles"),
        Message(id=2, role="assistant", content="ANSWER-BETA about bicycles"),
        Message(id=3, role="user", content="QUESTION-GAMMA about bicycles"),
    ]
    await harness.seed(CHROMA_UID, CHROMA_THREAD, seeded)
    before = await harness.entries(user_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    assert len(before) == 3

    provider = _fake_chat_provider(answer=SUMMARY_TEXT)
    with (
        patch("chibi.models.User.get_active_llm_provider", Mock(return_value=provider)),
        patch("chibi.models.User.get_active_llm_model", Mock(return_value=WINDOW_MODEL)),
        patch("chibi.services.user.gpt_settings.max_tokens", RESERVATION),
    ):
        result = await emergency_summarization(storage_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    await harness.tasks.drain()

    assert isinstance(result, CompactionResult)
    assert result.outcome is CompactionOutcome.PREDICTED_FITS
    stored = await harness.stored(user_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    assert len(stored) == 2
    assert all(message.source == COMPACTION_SEED_SOURCE for message in stored)
    after = await harness.entries(user_id=CHROMA_UID, thread_id=CHROMA_THREAD)
    assert set(before) <= set(after)
    assert all(after[entry_id] == before[entry_id] for entry_id in before)
    results: list[MemorySearchResult] = await harness.search(
        user_id=CHROMA_UID, thread_id=CHROMA_THREAD, query="QUESTION-ALPHA about bicycles"
    )
    contents = [entry.content for entry in results]
    assert "QUESTION-ALPHA about bicycles" in contents
    assert "ANSWER-BETA about bicycles" in contents
    assert all(str(entry.message_id) in {"1", "2", "3"} for entry in results)
