from typing import AsyncGenerator

import pytest
from fakeredis import FakeAsyncRedis
from freezegun import freeze_time

from chibi.models import Message
from chibi.storage.redis import RedisStorage


@pytest.fixture
async def redis_storage(monkeypatch) -> AsyncGenerator[RedisStorage, None]:
    """
    Create RedisStorage instance with FakeAsyncRedis backend.
    """
    fake = FakeAsyncRedis()

    async def fake_create(cls, url: str, password=None, db=0) -> RedisStorage:
        inst = cls(url, password, db)
        inst.redis = fake
        return inst

    monkeypatch.setattr(RedisStorage, "create", classmethod(fake_create), raising=True)
    storage = await RedisStorage.create("redis://localhost", None, 0)
    await storage.redis.flushdb()
    yield storage
    await storage.close()


@pytest.mark.asyncio
@freeze_time("2025-01-01 00:00:00")
async def test_ttl(redis_storage: RedisStorage) -> None:
    """
    Test that messages expire correctly without using actual sleep.
    """
    user = await redis_storage.get_or_create_user(1)
    m = Message(role="user", content="temp")
    await redis_storage.add_message(user, m, ttl=5)

    msgs = await redis_storage.get_messages(user)
    assert any(msg["content"] == "temp" for msg in msgs)

    # Travel beyond TTL
    with freeze_time("2025-01-01 00:00:06"):
        msgs2 = await redis_storage.get_messages(user)
        assert not any(msg["content"] == "temp" for msg in msgs2)


@pytest.mark.asyncio
async def test_message_order(redis_storage: RedisStorage) -> None:
    """
    Test that messages are returned in the order they were added.
    """
    user = await redis_storage.get_or_create_user(2)
    m1 = Message(role="user", content="first")
    m2 = Message(role="assistant", content="second")
    await redis_storage.add_message(user, m1)
    await redis_storage.add_message(user, m2)

    msgs = await redis_storage.get_messages(user)
    assert [msg["content"] for msg in msgs] == ["first", "second"]


@pytest.mark.asyncio
async def test_skip_bad_message(redis_storage: RedisStorage) -> None:
    """
    Test that invalid JSON entries are skipped.
    """
    user = await redis_storage.get_or_create_user(3)
    m = Message(role="user", content="good")
    await redis_storage.add_message(user, m)

    bad_key = f"user:{user.id}:message:999"
    await redis_storage.redis.set(bad_key, "not a json")

    msgs = await redis_storage.get_messages(user)
    assert [msg["content"] for msg in msgs] == ["good"]


@pytest.mark.asyncio
@freeze_time("2025-01-01 00:00:00")
async def test_drop_messages_with_ttl(redis_storage: RedisStorage) -> None:
    """
    Test that drop_messages removes all messages including expired ones.
    """
    user = await redis_storage.get_or_create_user(4)
    m1 = Message(role="user", content="a")
    m2 = Message(role="assistant", content="b")
    await redis_storage.add_message(user, m1, ttl=5)
    await redis_storage.add_message(user, m2)

    with freeze_time("2025-01-01 00:00:06"):
        await redis_storage.drop_messages(user)
        msgs = await redis_storage.get_messages(user)
        assert msgs == []


def test_combine_redis_dsn_no_password() -> None:
    """
    Test DSN remains unchanged when password is embedded in URL.
    """
    inst = RedisStorage("redis://user:pass@localhost", None, 0)
    dsn = inst._combine_redis_dsn("redis://user:pass@localhost", None)
    assert dsn == "redis://user:pass@localhost"


def test_combine_redis_dsn_with_password() -> None:
    """
    Test DSN contains provided password when not present in URL.
    """
    inst = RedisStorage("redis://localhost", "secret", 0)
    dsn = inst._combine_redis_dsn("redis://localhost", "secret")
    assert "secret" in dsn


def test_combine_redis_dsn_error() -> None:
    """
    Test invalid DSN raises ValueError.
    """
    inst = RedisStorage("redis://", "p", 0)
    with pytest.raises(ValueError):
        inst._combine_redis_dsn("redis://", "p")


@pytest.mark.asyncio
async def test_close_called(redis_storage: RedisStorage) -> None:
    """
    Test that close() is called properly.
    """
    closed: list[bool] = []

    async def fake_close() -> None:
        closed.append(True)

    redis_storage.close = fake_close  # type: ignore
    await redis_storage.close()
    assert closed
