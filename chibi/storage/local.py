import os
import pickle
import time
from typing import Optional

from chibi.config import gpt_settings
from chibi.models import Message, User
from chibi.storage.abc import Database


class LocalStorage(Database):
    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def _get_storage_filename(self, user_id: int) -> str:
        return os.path.join(self.storage_path, f"{user_id}.pkl")

    async def save_user(self, user: User) -> None:
        filename = self._get_storage_filename(user.id)
        with open(filename, "wb") as f:
            pickle.dump(user, f)

    async def get_or_create_user(self, user_id: int) -> User:
        filename = self._get_storage_filename(user_id)
        if not os.path.exists(filename):
            user = User(id=user_id)
            initial_message = Message(role="system", content=gpt_settings.assistant_prompt)
            user.messages = [
                initial_message,
            ]
            await self.save_user(user=user)
            return user

        with open(filename, "rb") as f:
            return pickle.load(f)

    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None:
        if ttl:
            expire_at = time.time() + ttl
        else:
            expire_at = None

        message_with_ttl = Message(role=message.role, content=message.content, expire_at=expire_at)
        user.messages.append(message_with_ttl)
        await self.save_user(user)

    async def get_messages(self, user: User) -> list[dict[str, str]]:
        current_time = time.time()

        return [
            msg.dict(exclude={"expire_at", "id"})
            for msg in user.messages
            if msg.expire_at is None or msg.expire_at > current_time
        ]

    async def drop_messages(self, user: User) -> None:
        initial_message = Message(role="system", content=gpt_settings.assistant_prompt)
        user.messages = [
            initial_message,
        ]
        await self.save_user(user=user)
