from typing import Any, Optional

from chibi.models import Message, User
from chibi.services.task_manager import task_manager
from chibi.storage.abstract import Database


class ChromaDecoratedStorage(Database):
    """Decorator that adds automatic archival to ChromaDB when messages are added.

    Uses __getattr__ to delegate all methods to inner storage except add_message.
    """

    def __init__(self, inner: Database, memory: Optional[Any] = None):
        self.inner = inner
        self.memory = memory

    async def add_message(self, user: User, message: Message, ttl: int | None = None) -> None:
        # First, add to primary storage
        await self.inner.add_message(user, message, ttl)

        # Then, archive to ChromaDB (fire-and-forget via background task manager)
        if self.memory is not None:
            task_manager.run_task(self.memory.archive(user.id, [message]), user_id=user.id)

    # Required abstract methods - delegate via __getattr__
    async def get_messages(self, user: User) -> list[dict[str, str]]:
        return await self.inner.get_messages(user)

    async def drop_messages(self, user: User) -> None:
        return await self.inner.drop_messages(user)

    async def get_user(self, user_id: int) -> User | None:
        return await self.inner.get_user(user_id)

    async def create_user(self, user_id: int) -> User:
        return await self.inner.create_user(user_id)

    async def save_user(self, user: User) -> None:
        return await self.inner.save_user(user)

    async def count_image(self, user_id: int) -> None:
        return await self.inner.count_image(user_id)

    # Additional methods via __getattr__
    def __getattr__(self, name: str) -> Any:
        """Delegate all other methods to inner storage."""
        return getattr(self.inner, name)
