import asyncio
import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from chibi.models import Message, User
from chibi.storage.abstract import Database


class DynamoDBStorage(Database):
    """
    DynamoDB-based storage backend implementing the Database interface.
    Uses two DynamoDB tables: one for users and one for messages.
    """

    def __init__(
        self,
        users_table_name: str,
        messages_table_name: str,
        aws_region: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> None:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )
        resource = session.resource("dynamodb")
        self.users_table = resource.Table(users_table_name)
        self.messages_table = resource.Table(messages_table_name)

    @classmethod
    async def create(
        cls,
        region: str,
        access_key: Optional[str],
        secret_access_key: Optional[str],
        users_table: str,
        messages_table: str,
    ) -> "DynamoDBStorage":
        """
        Factory method to create an instance of DynamoDBStorage.

        Args:
            region: AWS region name.
            access_key: AWS access key ID.
            secret_access_key: AWS secret access key.
            users_table: Name of the DynamoDB table for users.
            messages_table: Name of the DynamoDB table for messages.

        Returns:
            An initialized DynamoDBStorage instance.
        """
        return cls(
            users_table_name=users_table,
            messages_table_name=messages_table,
            aws_region=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )

    async def get_or_create_user(self, user_id: int) -> User:
        """
        Retrieve a User by ID or create a new one if it does not exist.

        Args:
            user_id: The user identifier.

        Returns:
            The User instance.
        """

        def _sync_get_or_create() -> User:
            try:
                resp = self.users_table.get_item(Key={"user_id": str(user_id)})
                item = resp.get("Item")
                if item and "data" in item:
                    return User.parse_raw(item["data"])
            except ClientError:
                pass
            # Create new user
            user = User(id=user_id)
            self.users_table.put_item(Item={"user_id": str(user_id), "data": user.model_dump_json()})
            return user

        return await asyncio.to_thread(_sync_get_or_create)

    async def save_user(self, user: User) -> None:
        """
        Persist the User object in the users table.

        Args:
            user: The User instance to save.
        """
        await asyncio.to_thread(
            self.users_table.put_item,
            Item={"user_id": str(user.id), "data": user.model_dump_json()},
        )

    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None:
        """
        Add a Message record for a user with optional TTL.

        Args:
            user: The User instance.
            message: The Message instance.
            ttl: Time-to-live in seconds for this message.
        """
        expire_at = None
        if ttl is not None:
            expire_at = time.time() + ttl
        item: Dict[str, Any] = {
            "user_id": str(user.id),
            "message_id": str(message.id),
            "role": message.role,
            "content": message.content,
        }
        if expire_at is not None:
            item["expire_at"] = int(expire_at)

        await asyncio.to_thread(self.messages_table.put_item, Item=item)

    async def get_messages(self, user: User) -> List[Dict[str, Any]]:
        """
        Retrieve non-expired messages for the user, ordered by message_id.

        Args:
            user: The User instance.

        Returns:
            A list of message dicts with keys 'role' and 'content'.
        """

        def _sync_query() -> List[Dict[str, Any]]:
            now_ts = int(time.time())
            try:
                resp = self.messages_table.query(
                    KeyConditionExpression="user_id = :uid",
                    ExpressionAttributeValues={":uid": str(user.id)},
                )
            except ClientError:
                return []
            items = resp.get("Items", [])
            # Filter expired
            valid = []
            for it in items:
                exp = it.get("expire_at")
                if exp is None or exp >= now_ts:
                    valid.append({"role": it["role"], "content": it["content"]})
            # Sort by message_id
            valid.sort(key=lambda x: int(it.get("message_id", "0")))
            return valid

        return await asyncio.to_thread(_sync_query)

    async def drop_messages(self, user: User) -> None:
        """
        Delete all messages for a user.

        Args:
            user: The User instance.
        """

        def _sync_delete_all() -> None:
            try:
                resp = self.messages_table.query(
                    KeyConditionExpression="user_id = :uid",
                    ExpressionAttributeValues={":uid": str(user.id)},
                )
            except ClientError:
                return
            items = resp.get("Items", [])
            with self.messages_table.batch_writer() as batch:
                for it in items:
                    batch.delete_item(Key={"user_id": it["user_id"], "message_id": it["message_id"]})

        await asyncio.to_thread(_sync_delete_all)
