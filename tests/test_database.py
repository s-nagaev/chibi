import boto3  # Import boto3
import pytest
from fakeredis import FakeAsyncRedis
from freezegun import freeze_time
from moto import mock_aws

from chibi.models import Message, SelectedModel, User
from chibi.storage.database import Database
from chibi.storage.dynamodb import DynamoDBStorage
from chibi.storage.local import LocalStorage
from chibi.storage.redis import RedisStorage

TABLE_USERS = "TestUsers"
TABLE_MESSAGES = "TestMessages"
REGION = "us-east-1"


# Fixture to provide different storage instances
@pytest.fixture(params=["local", "redis", "dynamodb"])
async def storage(request, tmp_path, monkeypatch):
    """Provides instances of different Database implementations."""
    if request.param == "local":
        # LocalStorage setup
        local_storage = LocalStorage(storage_path=str(tmp_path))
        yield local_storage

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
            # Create users table (PK = user_id)
            resource.create_table(
                TableName=TABLE_USERS,
                KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            ).wait_until_exists()
            # Create messages table (PK = user_id, SK = message_id)
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
            # client = boto3.client("dynamodb", region_name=REGION)
            # client.update_time_to_live(
            #     TableName=TABLE_MESSAGES,
            #     TimeToLiveSpecification={
            #         "Enabled": True,
            #         "AttributeName": "expire_at",
            #     }
            # )
            dynamo_storage = await DynamoDBStorage.create(
                region=REGION,
                access_key=None,
                secret_access_key=None,
                users_table=TABLE_USERS,
                messages_table=TABLE_MESSAGES,
            )
            yield dynamo_storage


async def test_create_user(storage: Database) -> None:
    """Tests getting or creating a user."""
    user_id = 123

    user = await storage.get_or_create_user(user_id)
    assert user is not None

    assert isinstance(user, User)
    assert user.id == user_id


async def test_save_and_get_existent_user(storage: Database) -> None:
    user_id = 123
    active_model = "test_model"
    active_provider = "test_provider"
    user = await storage.get_or_create_user(user_id)
    user.thread_selected_llm[0] = SelectedModel(name=active_model, provider_name=active_provider)
    await storage.save_user(user)

    refreshed_user = await storage.get_or_create_user(user_id)
    assert refreshed_user is not None
    assert isinstance(refreshed_user, User)
    assert refreshed_user.get_active_llm_model(thread_id=0) == active_model
    assert refreshed_user.thread_selected_llm[0].provider_name == active_provider


async def test_add_and_get_messages(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1, thread_id=0)
    await storage.add_message(user=user, message=message2, thread_id=0)

    messages = await storage.get_messages(user=user, thread_id=0)
    assert messages
    assert len(messages) == 2
    assert messages[0]["role"] == message1.role
    assert messages[0]["content"] == message1.content
    assert messages[1]["role"] == message2.role
    assert messages[1]["content"] == message2.content


async def test_drop_messages(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1, thread_id=0)
    await storage.add_message(user=user, message=message2, thread_id=0)

    await storage.drop_messages(user=user, thread_id=0)

    messages = await storage.get_messages(user=user, thread_id=0)
    assert not messages


@freeze_time("2025-01-01 00:00:00")
async def test_messages_with_ttl(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1, ttl=10, thread_id=0)
    await storage.add_message(user=user, message=message2, ttl=15, thread_id=0)

    with freeze_time("2025-01-01 00:00:11"):
        messages = await storage.get_messages(user=user, thread_id=0)
        assert messages
        assert len(messages) == 1
        assert messages[0]["role"] == message2.role
        assert messages[0]["content"] == message2.content


async def test_thread_isolation(storage: Database) -> None:
    """Verify messages with different thread_id don't mix."""
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    # Add messages to thread 0 (global)
    message_global = Message(role="user", content="Global message")
    await storage.add_message(user=user, message=message_global, thread_id=0)

    # Add messages to thread 1
    message_thread1 = Message(role="user", content="Thread 1 message")
    await storage.add_message(user=user, message=message_thread1, thread_id=1)

    # Add messages to thread 2
    message_thread2 = Message(role="user", content="Thread 2 message")
    await storage.add_message(user=user, message=message_thread2, thread_id=2)

    # Get messages for each thread
    global_messages = await storage.get_messages(user=user, thread_id=0)
    thread1_messages = await storage.get_messages(user=user, thread_id=1)
    thread2_messages = await storage.get_messages(user=user, thread_id=2)

    # Verify thread isolation
    assert len(global_messages) == 1
    assert global_messages[0]["content"] == "Global message"

    assert len(thread1_messages) == 1
    assert thread1_messages[0]["content"] == "Thread 1 message"

    assert len(thread2_messages) == 1
    assert thread2_messages[0]["content"] == "Thread 2 message"


async def test_drop_messages_specific_thread(storage: Database) -> None:
    """Verify drop only affects specified thread."""
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    # Add messages to thread 0 (global)
    msg_global1 = Message(role="user", content="Global 1")
    msg_global2 = Message(role="assistant", content="Global 2")
    await storage.add_message(user=user, message=msg_global1, thread_id=0)
    await storage.add_message(user=user, message=msg_global2, thread_id=0)

    # Add messages to thread 1
    msg_thread1 = Message(role="user", content="Thread 1 message")
    await storage.add_message(user=user, message=msg_thread1, thread_id=1)

    # Drop only thread 1 messages
    await storage.drop_messages(user=user, thread_id=1)

    # Verify thread 0 messages still exist
    global_messages = await storage.get_messages(user=user, thread_id=0)
    assert len(global_messages) == 2

    # Verify thread 1 messages are gone
    thread1_messages = await storage.get_messages(user=user, thread_id=1)
    assert len(thread1_messages) == 0
