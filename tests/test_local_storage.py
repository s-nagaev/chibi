import time

import pytest

from chibi.models import Message, User
from chibi.storage.local import LocalStorage


@pytest.fixture
async def storage(tmp_path):
    """
    Fixture for LocalStorage with a temporary directory.
    """
    storage = LocalStorage(storage_path=str(tmp_path))
    return storage


@pytest.mark.asyncio
async def test_get_or_create_user(storage):
    """
    Test that get_or_create_user returns a new User when none exists
    and retrieves the same User on subsequent calls.
    """
    user = await storage.get_or_create_user(user_id=1)
    assert isinstance(user, User)
    assert user.id == 1

    # Should retrieve existing user
    user2 = await storage.get_or_create_user(user_id=1)
    assert user2.id == 1
    assert user2.messages == []


@pytest.mark.asyncio
async def test_add_and_get_messages(storage):
    """
    Test that messages are added and retrieved correctly and respect TTL.
    """
    user = await storage.get_or_create_user(user_id=1)
    message = Message(role="user", content="hello")

    # Add message with TTL = 1 second
    await storage.add_message(user, message, ttl=1)
    msgs = await storage.get_messages(user)
    assert msgs == [{"role": "user", "content": "hello"}]

    # Wait until TTL expires
    time.sleep(1.1)
    expired_msgs = await storage.get_messages(user)
    assert expired_msgs == []


@pytest.mark.asyncio
async def test_drop_messages(storage):
    """
    Test that drop_messages removes all messages for a user.
    """
    user = await storage.get_or_create_user(user_id=2)
    message = Message(role="assistant", content="reply")

    await storage.add_message(user, message)
    await storage.drop_messages(user)
    msgs = await storage.get_messages(user)
    assert msgs == []
