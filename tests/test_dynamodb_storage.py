import time

import boto3
import pytest
from moto import mock_aws

from chibi.models import Message, User
from chibi.storage.dynamodb import DynamoDBStorage

TABLE_USERS = "TestUsers"
TABLE_MESSAGES = "TestMessages"
REGION = "us-east-1"


@pytest.fixture(autouse=True)
def aws_dynamodb_mock():
    """
    Moto mock for AWS DynamoDB. Creates tables for users and messages.
    """
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
        yield


@pytest.mark.asyncio
async def test_get_or_create_user_new():
    storage = await DynamoDBStorage.create(
        region=REGION,
        access_key=None,
        secret_access_key=None,
        users_table=TABLE_USERS,
        messages_table=TABLE_MESSAGES,
    )
    # New user should be created
    user = await storage.get_or_create_user(user_id=123)
    assert isinstance(user, User)
    assert user.id == 123
    # Ensure raw item exists in table
    tbl = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_USERS)
    resp = tbl.get_item(Key={"user_id": "123"})
    assert "Item" in resp
    assert resp["Item"]["data"]


@pytest.mark.asyncio
async def test_save_user_and_fetch():
    storage = await DynamoDBStorage.create(
        region=REGION,
        access_key=None,
        secret_access_key=None,
        users_table=TABLE_USERS,
        messages_table=TABLE_MESSAGES,
    )
    user = User(id=321)
    user.tokens = {"openai": "key123"}
    await storage.save_user(user)
    tbl = boto3.resource("dynamodb", region_name=REGION).Table(TABLE_USERS)
    resp = tbl.get_item(Key={"user_id": "321"})
    raw = resp["Item"]["data"]
    fetched = User.model_validate_json(raw)
    assert fetched.id == 321
    assert fetched.tokens == {"openai": "key123"}


@pytest.mark.asyncio
async def test_add_and_get_messages_and_drop():
    storage = await DynamoDBStorage.create(
        region=REGION,
        access_key=None,
        secret_access_key=None,
        users_table=TABLE_USERS,
        messages_table=TABLE_MESSAGES,
    )
    user = await storage.get_or_create_user(user_id=111)
    m1 = Message(role="user", content="Hello")
    m2 = Message(role="assistant", content="World")
    await storage.add_message(user, m1)
    await storage.add_message(user, m2)
    msgs = await storage.get_messages(user)
    contents = [m["content"] for m in msgs]
    assert contents == ["Hello", "World"]
    await storage.drop_messages(user)
    msgs_after = await storage.get_messages(user)
    assert msgs_after == []


@pytest.mark.asyncio
async def test_message_ttl_expiry(monkeypatch):
    storage = await DynamoDBStorage.create(
        region=REGION,
        access_key=None,
        secret_access_key=None,
        users_table=TABLE_USERS,
        messages_table=TABLE_MESSAGES,
    )
    user = await storage.get_or_create_user(user_id=222)
    base = time.time()
    monkeypatch.setattr(time, "time", lambda: base)
    m1 = Message(role="user", content="Temp")
    await storage.add_message(user, m1, ttl=60)
    monkeypatch.setattr(time, "time", lambda: base + 61)
    msgs = await storage.get_messages(user)
    assert msgs == []
