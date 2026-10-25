"""Tests for the durable per-thread context measurement and the single size entry point."""

from collections.abc import AsyncIterator

import boto3
import pytest
from fakeredis import FakeAsyncRedis
from moto import mock_aws

from chibi.models import FunctionSchema, Message, ThreadUsageRecord, ToolSchema, User
from chibi.schemas.app import UsageSchema
from chibi.services.context_measurement import (
    HeuristicTokenizer,
    SizeProvenance,
    estimate_messages_tokens,
    get_last_measured_usage,
    get_tokenizer,
    measure_request_size,
    measure_request_size_with_provenance,
    persist_measured_usage,
    predict_request_size,
    predict_request_size_with_provenance,
)
from chibi.services.usage_cache import UsageCacheStore
from chibi.storage.database import Database
from chibi.storage.dynamodb import DynamoDBStorage
from chibi.storage.local import LocalStorage
from chibi.storage.redis import RedisStorage

TABLE_USERS = "TestUsers"
TABLE_MESSAGES = "TestMessages"
REGION = "us-east-1"
MODEL = "test-model"
OTHER_MODEL = "other-model"


@pytest.fixture(params=["local", "redis", "dynamodb"])
async def storage(request: pytest.FixtureRequest, tmp_path, monkeypatch) -> AsyncIterator[Database]:
    """Provide instances of different Database implementations."""
    if request.param == "local":
        yield LocalStorage(storage_path=str(tmp_path))

    if request.param == "redis":
        fake = FakeAsyncRedis()

        async def fake_create(cls, url: str, password=None, db=0) -> RedisStorage:
            inst = cls(url, password, db)
            inst.redis = fake
            return inst

        monkeypatch.setattr(RedisStorage, "create", classmethod(fake_create), raising=True)
        redis_storage = await RedisStorage.create("redis://localhost", None, 0)
        await redis_storage.redis.flushdb()
        yield redis_storage
        await redis_storage.close()

    if request.param == "dynamodb":
        with mock_aws(config={"core": {"service_whitelist": ["dynamodb"]}}):
            resource = boto3.resource("dynamodb", region_name=REGION)
            resource.create_table(
                TableName=TABLE_USERS,
                KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            ).wait_until_exists()
            resource.create_table(
                TableName=TABLE_MESSAGES,
                KeySchema=[
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "message_id", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "user_id", "AttributeType": "S"},
                    {"AttributeName": "message_id", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            ).wait_until_exists()
            yield await DynamoDBStorage.create(
                region=REGION,
                access_key=None,
                secret_access_key=None,
                users_table=TABLE_USERS,
                messages_table=TABLE_MESSAGES,
            )


def _record(prompt_tokens: int = 1000, model: str = MODEL, message_count: int = 2) -> ThreadUsageRecord:
    return ThreadUsageRecord(prompt_tokens=prompt_tokens, model=model, message_count=message_count)


def _messages(count: int = 4) -> list[Message]:
    return [Message(role="user", content=f"message number {i}") for i in range(count)]


async def test_persist_and_read_round_trip(storage: Database) -> None:
    """A measurement persists per (user, thread) and reads back with its provenance."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=150),
        provider="OpenAI",
        model=MODEL,
        message_count=3,
    )

    record = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL)
    assert record is not None
    assert record.prompt_tokens == 150
    assert record.model == MODEL
    assert record.message_count == 3


async def test_measurement_is_per_thread_not_per_user(storage: Database) -> None:
    """Two threads of one user keep their measurements independent."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=1,
        usage=UsageSchema(prompt_tokens=100),
        provider="OpenAI",
        model=MODEL,
        message_count=1,
    )
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=2,
        usage=UsageSchema(prompt_tokens=9000),
        provider="OpenAI",
        model=MODEL,
        message_count=9,
    )

    first = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=1, model=MODEL)
    second = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=2, model=MODEL)
    assert first is not None and first.prompt_tokens == 100
    assert second is not None and second.prompt_tokens == 9000


async def test_record_survives_process_restart(tmp_path) -> None:
    """A fresh storage instance (new process) still resolves the last measured size."""
    first_instance = LocalStorage(storage_path=str(tmp_path))
    await persist_measured_usage.__wrapped__(
        first_instance,
        user_id=42,
        thread_id=0,
        usage=UsageSchema(prompt_tokens=777),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )
    UsageCacheStore()._data.clear()

    second_instance = LocalStorage(storage_path=str(tmp_path))
    record = await get_last_measured_usage.__wrapped__(second_instance, user_id=42, thread_id=0, model=MODEL)
    assert record is not None
    assert record.prompt_tokens == 777


async def test_unreported_usage_never_zeroes_a_good_value(storage: Database) -> None:
    """A response without reported usage (raw size 0) leaves the good record untouched."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=500),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(),
        provider="OpenAI",
        model=MODEL,
        message_count=3,
    )

    record = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL)
    assert record is not None
    assert record.prompt_tokens == 500
    assert record.message_count == 2


async def test_missing_entry_degrades_to_estimate_and_raises_nothing(storage: Database) -> None:
    """A thread without a record sizes the request from the heuristic estimate alone."""
    messages = _messages(4)
    size = await measure_request_size.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL, messages=messages)
    assert size > 0
    fresh_user = await storage.get_or_create_user(user_id=42)
    assert isinstance(fresh_user, User)
    assert 7 not in fresh_user.thread_usage


async def test_prediction_is_measured_plus_appended(storage: Database) -> None:
    """The prediction equals measured tokens plus the heuristic count of appended messages."""
    messages = _messages(4)
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=1000),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )

    size = await measure_request_size.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL, messages=messages)
    appended_estimate = predict_request_size(record=None, model=MODEL, messages=messages[2:])
    assert size == 1000 + appended_estimate
    assert size != appended_estimate


def test_prediction_with_no_record_counts_whole_history() -> None:
    """Without a record the prediction is the full heuristic count of the history."""
    messages = _messages(3)
    assert predict_request_size(record=None, model=MODEL, messages=messages) > 0


async def test_model_mismatch_degrades_to_estimate(storage: Database) -> None:
    """A record measured on a different model is treated as unknown."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=1000),
        provider="OpenAI",
        model=OTHER_MODEL,
        message_count=1,
    )

    record = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL)
    assert record is None
    messages = _messages(4)
    size = await measure_request_size.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL, messages=messages)
    assert size == predict_request_size(record=None, model=MODEL, messages=messages)


async def test_shrunken_history_degrades_to_estimate() -> None:
    """A record whose message count exceeds the current history is stale and never dictates."""
    record = _record(prompt_tokens=1000, message_count=10)
    messages = _messages(3)
    size = predict_request_size(record=record, model=MODEL, messages=messages)
    assert size == predict_request_size(record=None, model=MODEL, messages=messages)


def test_degraded_prediction_never_raises_on_absent_record() -> None:
    """The prediction tolerates a completely absent record and empty history."""
    assert predict_request_size(record=None, model=MODEL, messages=[]) == 0


def test_zero_message_count_is_unknown_not_zero_measured() -> None:
    """A record with an unknown (0) count never stacks the whole history on the measured prompt.

    Regression for the reviewer's executed scenario: ``message_count=0`` used to
    be read as "the measured prompt contained zero messages", so all four
    messages were counted on top of a ``prompt_tokens`` that already included
    them (1000 + 404 instead of 404). A non-positive count is a wrong-sized
    entry and must degrade to the pure heuristic estimate.
    """
    messages = [Message(role="user", content="x" * 400) for _ in range(4)]
    record = _record(prompt_tokens=1000, message_count=0)
    estimate = predict_request_size(record=None, model=MODEL, messages=messages)
    size = predict_request_size(record=record, model=MODEL, messages=messages)
    assert size == estimate
    assert size < 1000


def test_real_message_count_adds_exactly_the_messages_after_the_count() -> None:
    """A record with a real count adds exactly the messages measured after it."""
    messages = [Message(role="user", content="x" * 400) for _ in range(4)]
    record = _record(prompt_tokens=1000, message_count=2)
    appended = predict_request_size(record=None, model=MODEL, messages=messages[2:])
    whole = predict_request_size(record=None, model=MODEL, messages=messages)
    assert predict_request_size(record=record, model=MODEL, messages=messages) == 1000 + appended
    assert 1000 + appended != 1000 + whole


async def test_unknown_count_is_never_persisted(storage: Database) -> None:
    """A response with an unknown message count leaves the durable record untouched."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=500),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=900),
        provider="OpenAI",
        model=MODEL,
        message_count=None,
    )

    record = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL)
    assert record is not None
    assert record.prompt_tokens == 500
    assert record.message_count == 2

    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=9,
        usage=UsageSchema(prompt_tokens=900),
        provider="OpenAI",
        model=MODEL,
        message_count=None,
    )
    fresh_user = await storage.get_or_create_user(user_id=42)
    assert isinstance(fresh_user, User)
    assert 9 not in fresh_user.thread_usage


async def test_accessor_tolerates_user_without_thread_usage_field(storage: Database) -> None:
    """A database serving a user object without the field degrades instead of raising."""
    from types import SimpleNamespace
    from typing import cast

    class _StubUserDatabase:
        def __init__(self, inner: Database) -> None:
            self._inner = inner

        async def get_or_create_user(self, user_id: int) -> SimpleNamespace:
            user = await self._inner.get_or_create_user(user_id=user_id)
            return SimpleNamespace(id=user.id)

    messages = _messages(4)
    stub_db = _StubUserDatabase(storage)
    size = await measure_request_size.__wrapped__(
        cast(Database, stub_db), user_id=42, thread_id=7, model=MODEL, messages=messages
    )
    assert size == predict_request_size(record=None, model=MODEL, messages=messages)
    assert size > 0


def test_heuristic_counts_tool_call_arguments() -> None:
    """The heuristic includes tool_calls argument payloads."""
    plain = Message(role="assistant", content="ok")
    with_tools = Message(
        role="assistant",
        content="ok",
        tool_calls=[
            ToolSchema(
                id="1",
                function=FunctionSchema(name="big_tool", arguments="x" * 4000),
            )
        ],
    )
    assert HeuristicTokenizer().count_tokens("") == 0
    assert predict_request_size(record=None, model=MODEL, messages=[with_tools]) > (
        predict_request_size(record=None, model=MODEL, messages=[plain])
    )


def test_tokenizer_is_optional_at_runtime() -> None:
    """The heuristic tokenizer is always available; no tokenizer import can crash sizing."""
    tokenizer = get_tokenizer(MODEL)
    assert isinstance(tokenizer, HeuristicTokenizer)
    assert tokenizer.count_tokens("abcd") == 1


def test_estimate_matches_utility_helper() -> None:
    """The delegated utility helper and the module estimate agree exactly."""
    from chibi.services.providers.utils import _estimate_history_size

    messages = [_messages(2)[0], _messages(3)[1]]
    assert _estimate_history_size(messages) == predict_request_size(record=None, model=MODEL, messages=messages)


async def test_reset_chat_history_removes_durable_record(storage: Database) -> None:
    """A history reset also removes the durable measurement for that thread only."""
    from chibi.services.user import reset_chat_history

    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=500),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=8,
        usage=UsageSchema(prompt_tokens=300),
        provider="OpenAI",
        model=MODEL,
        message_count=1,
    )
    user = await storage.get_or_create_user(user_id=42)
    await storage.add_message(user=user, message=Message(role="user", content="hi"), thread_id=7)

    await reset_chat_history.__wrapped__(storage, storage_id=42, thread_id=7)

    removed = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=7, model=MODEL)
    assert removed is None
    kept = await get_last_measured_usage.__wrapped__(storage, user_id=42, thread_id=8, model=MODEL)
    assert kept is not None
    assert kept.prompt_tokens == 300
    messages = await storage.get_messages(user=user, thread_id=7)
    assert messages == []


async def test_provenance_is_provider_measured_when_a_valid_record_anchors(storage: Database) -> None:
    """A valid durable record anchors the figure: the provenance says provider-measured."""
    messages = _messages(4)
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=1000),
        provider="OpenAI",
        model=MODEL,
        message_count=2,
    )

    size, provenance = await measure_request_size_with_provenance.__wrapped__(
        storage, user_id=42, thread_id=7, model=MODEL, messages=messages
    )

    assert provenance is SizeProvenance.PROVIDER_MEASURED
    assert size == 1000 + predict_request_size(record=None, model=MODEL, messages=messages[2:])


def test_provenance_is_estimated_without_a_record() -> None:
    """Without a record the whole history is a heuristic estimate and the provenance says so."""
    messages = _messages(3)
    size, provenance = predict_request_size_with_provenance(record=None, model=MODEL, messages=messages)
    assert provenance is SizeProvenance.ESTIMATED
    assert size == estimate_messages_tokens(messages)


async def test_provenance_degrades_to_estimated_on_model_mismatch(storage: Database) -> None:
    """A record measured on another model is unknown, and the provenance says estimated."""
    await persist_measured_usage.__wrapped__(
        storage,
        user_id=42,
        thread_id=7,
        usage=UsageSchema(prompt_tokens=1000),
        provider="OpenAI",
        model=OTHER_MODEL,
        message_count=1,
    )
    messages = _messages(4)

    size, provenance = await measure_request_size_with_provenance.__wrapped__(
        storage, user_id=42, thread_id=7, model=MODEL, messages=messages
    )

    assert provenance is SizeProvenance.ESTIMATED
    assert size == estimate_messages_tokens(messages)


def test_provenance_wrapper_keeps_the_int_contract() -> None:
    """The plain prediction function still returns the bare int its callers expect."""
    messages = _messages(3)
    size = predict_request_size(record=None, model=MODEL, messages=messages)
    assert size == predict_request_size_with_provenance(record=None, model=MODEL, messages=messages)[0]
