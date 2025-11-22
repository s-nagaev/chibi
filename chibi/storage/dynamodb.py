import asyncio
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from chibi.models import Message, User
from chibi.storage.abstract import Database


class DynamoDBStorage(Database):
    """DynamoDB storage backend implementing Database interface.

    Uses two DynamoDB tables:
      - users table (PK=user_id)
      - messages table (PK=user_id, SK=message_id)
    """

    def __init__(
        self,
        users_table_name: str,
        messages_table_name: str,
        aws_region: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        """Initialize the DynamoDBStorage.

        Args:
            users_table_name: Name of the DynamoDB table for users.
            messages_table_name: Name of the DynamoDB table for messages.
            aws_region: AWS region for the DynamoDB tables.
            aws_access_key_id: AWS access key ID. Defaults to None.
            aws_secret_access_key: AWS secret access key. Defaults to None.
        """
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
        self.dynamodb = session.resource("dynamodb")
        self.users_table = self.dynamodb.Table(users_table_name)
        self.messages_table = self.dynamodb.Table(messages_table_name)

    @classmethod
    async def create(
        cls,
        region: str,
        access_key: str | None,
        secret_access_key: str | None,
        users_table: str,
        messages_table: str,
    ) -> "DynamoDBStorage":
        """Create and initializes an instance of DynamoDBStorage.

        Args:
            region: AWS region.
            access_key: AWS access key ID.
            secret_access_key: AWS secret access key.
            users_table: Name of the users table.
            messages_table: Name of the messages table.

        Returns:
            DynamoDBStorage: An instance of the DynamoDBStorage class.
        """
        instance = cls(
            users_table_name=users_table,
            messages_table_name=messages_table,
            aws_region=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )
        await instance.connect()
        return instance

    async def connect(self) -> None:
        """Ensure users and messages tables exist (creates if missing).

        Raises:
            ClientError: If there's an issue with DynamoDB operations other than
                ResourceNotFoundException when checking/creating tables.
        """
        client = self.dynamodb.meta.client
        # users table
        try:
            client.describe_table(TableName=self.users_table.table_name)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "ResourceNotFoundException":
                client.create_table(
                    TableName=self.users_table.table_name,
                    KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST",
                )
                client.get_waiter("table_exists").wait(TableName=self.users_table.table_name)
            else:
                raise
        # messages table
        try:
            client.describe_table(TableName=self.messages_table.table_name)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "ResourceNotFoundException":
                client.create_table(
                    TableName=self.messages_table.table_name,
                    KeySchema=[
                        {"AttributeName": "user_id", "KeyType": "HASH"},
                        {"AttributeName": "message_id", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "user_id", "AttributeType": "S"},
                        {"AttributeName": "message_id", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                client.get_waiter("table_exists").wait(TableName=self.messages_table.table_name)
            else:
                raise

    async def get_user(self, user_id: int) -> User | None:
        """Retrieve a User by ID.

        Args:
            user_id: The ID of the user to retrieve.

        Returns:
            The User object if found, otherwise None.
        """

        def _sync() -> User | None:
            try:
                resp = self.users_table.get_item(Key={"user_id": str(user_id)})
                item = resp.get("Item")
                if not item or "data" not in item:
                    return None
                return User.model_validate_json(item["data"])
            except ClientError:
                return None

        return await asyncio.to_thread(_sync)

    async def create_user(self, user_id: int) -> User:
        """Create a new User record.

        Args:
            user_id: The ID for the new user.

        Returns:
            The newly created User object.
        """
        user = User(id=user_id)
        await self.save_user(user)
        return user

    async def save_user(self, user: User) -> None:
        """Persist the User object to DynamoDB.

        Args:
            user: The User object to persist.
        """
        await asyncio.to_thread(
            self.users_table.put_item,
            Item={"user_id": str(user.id), "data": user.model_dump_json()},
        )

    async def get_or_create_user(self, user_id: int) -> User:
        """Retrieve a User by ID or creates if missing; loads messages.

        Args:
            user_id: The ID of the user to retrieve or create.

        Returns:
            The retrieved or newly created User object, with messages loaded.
        """
        user = await self.get_user(user_id)
        if not user:
            user = await self.create_user(user_id)

        # load messages
        def _load() -> list[dict[str, Any]]:
            resp = self.messages_table.query(
                KeyConditionExpression="user_id = :u",
                ExpressionAttributeValues={":u": str(user_id)},
            )
            return resp.get("Items", [])

        items = await asyncio.to_thread(_load)
        now_ts = int(time.time())
        msgs: list[Message] = []
        for it in items:
            exp = it.get("expire_at")
            if exp is None or exp >= now_ts:
                msgs.append(
                    Message(
                        id=int(it["message_id"]),
                        role=it.get("role", ""),
                        content=it.get("content", ""),
                        expire_at=exp,
                    )
                )
        msgs.sort(key=lambda m: m.id)
        user.messages = msgs
        return user

    async def add_message(self, user: User, message: Message, ttl: int | None = None) -> None:
        """Add a Message record with optional TTL in seconds.

        Args:
            user: The user to whom the message belongs.
            message: The message object to add.
            ttl: Optional Time To Live for the message in seconds.
        """
        expire_at: int | None = None
        if ttl is not None:
            expire_at = int(time.time()) + ttl

        item: dict[str, Any] = {
            "user_id": str(user.id),
            "message_id": str(message.id),
            "role": message.role,
            "content": message.content,
        }
        if expire_at is not None:
            item["expire_at"] = expire_at

        await asyncio.to_thread(self.messages_table.put_item, Item=item)

    async def get_messages(self, user: User) -> list[dict[str, str]]:
        """Retrieve non-expired messages as simple dicts.

        Args:
            user: The user whose messages are to be retrieved.

        Returns:
            A list of non-expired messages, where each message is a dictionary (excluding 'expire_at' and 'id').
        """
        user_ref = await self.get_or_create_user(user.id)
        return [msg.model_dump(exclude={"expire_at", "id"}) for msg in user_ref.messages]

    async def drop_messages(self, user: User) -> None:
        """Delete all messages for a user.

        Args:
            user: The user whose messages are to be deleted.
        """

        def _sync() -> None:
            resp = self.messages_table.query(
                KeyConditionExpression="user_id = :u",
                ExpressionAttributeValues={":u": str(user.id)},
            )
            for it in resp.get("Items", []):
                self.messages_table.delete_item(Key={"user_id": it["user_id"], "message_id": it["message_id"]})

        await asyncio.to_thread(_sync)
