import os
import pickle
import time

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
            pickle.dump(user.model_dump(), f)

    async def create_user(self, user_id: int) -> User:
        user = User(id=user_id)
        await self.save_user(user=user)
        return user

    async def get_user(self, user_id: int) -> User | None:
        filename = self._get_storage_filename(user_id)
        user = None
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            user = User(**data)
        if isinstance(data, User):
            user = User(**data.model_dump())
        if user:
            current_time = time.time()
            if hasattr(user, "images"):
                user.images = [img for img in user.images if img.expire_at > current_time]
            return user

        return None

    async def add_message(self, user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
        user_refreshed = await self.get_or_create_user(user_id=user.id)
        expire_at = time.time() + ttl if ttl else None

        message.expire_at = expire_at
        if thread_id:
            user_refreshed.thread_messages_map[thread_id].append(message)
        else:
            user_refreshed.messages.append(message)
        await self.save_user(user_refreshed)

    async def get_messages(self, user: User, thread_id: int = 0) -> list[dict[str, str]]:
        user_refreshed = await self.get_or_create_user(user_id=user.id)
        current_time = time.time()

        if thread_id:
            messages = user_refreshed.thread_messages_map.get(thread_id, [])
        else:
            messages = user_refreshed.messages

        msgs = [
            msg.model_dump(exclude={"expire_at", "id"})
            for msg in messages
            if msg.expire_at is None or msg.expire_at > current_time
        ]
        return msgs

    async def drop_messages(self, user: User, thread_id: int = 0) -> None:
        if thread_id:
            user.thread_messages_map[thread_id] = []
        else:
            user.messages = []
        await self.save_user(user=user)
