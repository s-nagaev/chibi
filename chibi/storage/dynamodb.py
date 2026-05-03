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
        return user

    async def add_message(self, user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
        """Add a Message record with optional TTL in seconds.

        Args:
            user: The user to whom the message belongs.
            message: The message object to add.
            ttl: Optional Time To Live for the message in seconds.
            thread_id: Thread identifier (0 for global messages).
        """
        expire_at: int | None = None
        if ttl is not None:
            expire_at = int(time.time()) + ttl

        item: dict[str, Any] = {
            "user_id": str(user.id),
            "message_id": str(message.id),
            "thread_id": thread_id,
            "data": message.model_dump_json(exclude={"expire_at"}),
        }
        if expire_at is not None:
            item["expire_at"] = expire_at

        await asyncio.to_thread(self.messages_table.put_item, Item=item)

    async def get_messages(self, user: User, thread_id: int = 0) -> list[dict[str, Any]]:
        """Retrieve non-expired messages as simple dicts.

        Args:
            user: The user whose messages are to be retrieved.
            thread_id: Thread identifier (0 for global messages).

        Returns:
            A list of non-expired messages, where each message is a dictionary (excluding 'expire_at' and 'id').
        """
        now_ts = int(time.time())

        def _sync() -> list[dict[str, Any]]:
            query_params: dict[str, Any] = {
                "KeyConditionExpression": "user_id = :u",
                "ExpressionAttributeValues": {":u": str(user.id)},
            }
            # thread_id=0 must explicitly filter for global messages (backward compatibility)
            if thread_id == 0:
                query_params["FilterExpression"] = "thread_id = :t OR attribute_not_exists(thread_id)"
                query_params["ExpressionAttributeValues"][":t"] = 0
            else:
                query_params["FilterExpression"] = "thread_id = :t"
                query_params["ExpressionAttributeValues"][":t"] = thread_id

            resp = self.messages_table.query(**query_params)
            items = resp.get("Items", [])

            # Filter expired messages
            result = []
            for it in items:
                exp = it.get("expire_at")
                if exp is None or exp >= now_ts:
                    # Use "data" field for full message serialization (new format)
                    if "data" in it:
                        msg = Message.model_validate_json(it["data"])
                        result.append(msg.model_dump(exclude={"expire_at", "id"}))
                    else:
                        # Backward compatibility: reconstruct from individual fields (old format)
                        result.append(
                            {
                                "role": it.get("role", ""),
                                "content": it.get("content", ""),
                            }
                        )
            return result

        items = await asyncio.to_thread(_sync)
        return items

    async def drop_messages(self, user: User, thread_id: int = 0) -> None:
        """Delete messages for a user, optionally filtered by thread_id.

        Args:
            user: The user whose messages are to be deleted.
            thread_id: Thread identifier (0 for global messages, >0 for specific thread).
        """

        def _sync() -> None:
            query_params: dict[str, Any] = {
                "KeyConditionExpression": "user_id = :u",
                "ExpressionAttributeValues": {":u": str(user.id)},
            }
            # thread_id=0 must explicitly filter for global messages (backward compatibility)
            if thread_id == 0:
                query_params["FilterExpression"] = "thread_id = :t OR attribute_not_exists(thread_id)"
                query_params["ExpressionAttributeValues"][":t"] = 0
            else:
                query_params["FilterExpression"] = "thread_id = :t"
                query_params["ExpressionAttributeValues"][":t"] = thread_id

            resp = self.messages_table.query(**query_params)
            for it in resp.get("Items", []):
                self.messages_table.delete_item(Key={"user_id": it["user_id"], "message_id": it["message_id"]})

        await asyncio.to_thread(_sync)
