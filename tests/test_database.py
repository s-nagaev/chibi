import boto3  # Import boto3
import pytest
from fakeredis import FakeAsyncRedis
from freezegun import freeze_time
from moto import mock_aws

from chibi.models import Message, User
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
    user = await storage.get_or_create_user(user_id)
    user.selected_gpt_model_name = active_model
    await storage.save_user(user)

    refreshed_user = await storage.get_or_create_user(user_id)
    assert refreshed_user is not None
    assert isinstance(refreshed_user, User)
    assert refreshed_user.selected_gpt_model_name == active_model


async def test_add_and_get_messages(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1)
    await storage.add_message(user=user, message=message2)

    refreshed_user = await storage.get_or_create_user(user_id)
    assert refreshed_user.messages
    assert len(refreshed_user.messages) == 2
    assert refreshed_user.messages[0].role == message1.role
    assert refreshed_user.messages[0].content == message1.content
    assert refreshed_user.messages[1].role == message2.role
    assert refreshed_user.messages[1].content == message2.content

    messages = await storage.get_messages(user=user)
    assert messages
    assert len(messages) == 2
    assert messages[0] == message1.model_dump(exclude={"expire_at", "id"})
    assert messages[1] == message2.model_dump(exclude={"expire_at", "id"})


async def test_drop_messages(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1)
    await storage.add_message(user=user, message=message2)

    await storage.drop_messages(user=user)

    messages = await storage.get_messages(user=user)
    assert not messages

    refreshed_user = await storage.get_or_create_user(user_id)
    assert not refreshed_user.messages


@freeze_time("2025-01-01 00:00:00")
async def test_messages_with_ttl(storage: Database) -> None:
    user_id = 123
    user = await storage.get_or_create_user(user_id)

    message1 = Message(role="user", content="Hello")
    message2 = Message(role="assistant", content="Hi there!")

    await storage.add_message(user=user, message=message1, ttl=10)
    await storage.add_message(user=user, message=message2, ttl=15)

    with freeze_time("2025-01-01 00:00:11"):
        messages = await storage.get_messages(user=user)
        assert messages
        assert len(messages) == 1
        assert messages[0]["role"] == message2.role
        assert messages[0]["content"] == message2.content
