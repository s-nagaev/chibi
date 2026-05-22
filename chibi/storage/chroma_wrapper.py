from chibi.memory.abstract import LongConversationMemory
from chibi.models import Message, User
from chibi.services.task_manager import task_manager
from chibi.storage.abstract import Database


class ChromaWrappedStorage(Database):
    """Wrapper that adds automatic archival to ChromaDB when messages are added.

    Uses __getattr__ to delegate all methods to inner storage except add_message.

    Attributes:
        inner: The wrapped database instance.
        memory: Optional ChromaDB memory instance for semantic search.
    """

    def __init__(self, inner: Database, memory: LongConversationMemory | None = None):
        """Initialize the wrapper.

        Args:
            inner: The database to wrap.
            memory: Optional memory for ChromaDB archival.
        """
        self.inner = inner
        self.memory = memory

    async def add_message(self, user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
        """Add message to storage and archive to ChromaDB.

        Args:
            user: The user instance.
            message: The message to add.
            ttl: Time to live in minutes (optional).
            thread_id: Thread ID (default: 0).
        """
        # First, add to primary storage
        await self.inner.add_message(user, message, ttl)

        # Then, archive to ChromaDB (fire-and-forget via background task manager)
        if self.memory:
            task_manager.run_task(self.memory.archive(user.id, [message]), user_id=user.id)

    async def get_messages(self, user: User, thread_id: int = 0) -> list[dict[str, str]]:
        """Get messages for user.

        Args:
            user: The user instance.
            thread_id: Thread ID (default: 0).

        Returns:
            List of message dictionaries.
        """
        return await self.inner.get_messages(user)

    async def drop_messages(self, user: User, thread_id: int = 0) -> None:
        """Drop messages for user.

        Args:
            user: The user instance.
            thread_id: Thread ID (default: 0).
        """
        return await self.inner.drop_messages(user)

    async def get_user(self, user_id: int) -> User | None:
        """Get user by ID.

        Args:
            user_id: The user ID.

        Returns:
            User instance or None.
        """
        return await self.inner.get_user(user_id)

    async def create_user(self, user_id: int) -> User:
        """Create new user.

        Args:
            user_id: The user ID.

        Returns:
            Created user instance.
        """
        return await self.inner.create_user(user_id)

    async def save_user(self, user: User) -> None:
        """Save user to storage.

        Args:
            user: The user to save.
        """
        return await self.inner.save_user(user)

    async def count_image(self, user_id: int) -> None:
        """Count images for user.

        Args:
            user_id: The user ID.
        """
        return await self.inner.count_image(user_id)

    def __getattr__(self, name: str) -> object:
        """Delegate all other methods to inner storage.

        Args:
            name: Attribute name.

        Returns:
            The delegated attribute from inner storage.
        """
        return getattr(self.inner, name)
