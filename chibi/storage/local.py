import os
import pickle
import time
from typing import Optional

from loguru import logger

from chibi.models import Message, User
from chibi.storage.abstract import Database


class LocalStorage(Database):
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        logger.info("Local storage initialized.")

    def _get_storage_filename(self, user_id: int) -> str:
        return os.path.join(self.storage_path, f"{user_id}.pkl")

    async def save_user(self, user: User) -> None:
        filename = self._get_storage_filename(user.id)
        with open(filename, "wb") as f:
            pickle.dump(user.dict(), f)

    async def create_user(self, user_id: int) -> User:
        user = User(id=user_id)
        await self.save_user(user=user)
        return user

    async def get_user(self, user_id: int) -> User | None:
        filename = self._get_storage_filename(user_id)
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return User(**data)
        if isinstance(data, User):
            return User(**data.dict())
        return None

    async def get_or_create_user(self, user_id: int) -> User:
        if user := await self.get_user(user_id=user_id):
            current_time = time.time()
            if hasattr(user, "images"):
                user.images = [img for img in user.images if img.expire_at > current_time]
            return user
        return await self.create_user(user_id=user_id)

    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None:
        user_refreshed = await self.get_or_create_user(user_id=user.id)
        expire_at = time.time() + ttl if ttl else None

        message_with_ttl = Message(role=message.role, content=message.content, expire_at=expire_at)
        user_refreshed.messages.append(message_with_ttl)
        await self.save_user(user_refreshed)

    async def get_messages(self, user: User) -> list[dict[str, str]]:
        user_refreshed = await self.get_or_create_user(user_id=user.id)
        current_time = time.time()

        msgs = [
            msg.dict(exclude={"expire_at", "id"})
            for msg in user_refreshed.messages
            if msg.expire_at is None or msg.expire_at > current_time
        ]
        return msgs

    async def drop_messages(self, user: User) -> None:
        user.messages = []
        await self.save_user(user=user)
