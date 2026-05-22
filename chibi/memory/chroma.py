import asyncio
from datetime import datetime, timedelta

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from loguru import logger

from chibi.config import application_settings
from chibi.memory.abstract import LongConversationMemory, MemorySearchResult
from chibi.models import Message

# Global cache for embedding function to avoid repeated downloads
_cached_embedding_fn: EmbeddingFunction | None = None


def _get_embedding_function() -> EmbeddingFunction:
    """Get cached embedding function singleton.

    Returns:
        Cached DefaultEmbeddingFunction instance.
    """
    global _cached_embedding_fn
    if _cached_embedding_fn is None:
        _cached_embedding_fn = DefaultEmbeddingFunction()
    return _cached_embedding_fn


class ChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using embedded PersistentClient (synchronous).

    This class uses ChromaDB's persistent client to store conversation history
    locally on disk. Suitable for single-instance deployments.
    """

    def __init__(self) -> None:
        """Initialize ChromaDB embedded client."""
        self._cached_embedding_fn: EmbeddingFunction = _get_embedding_function()
        self._client = chromadb.PersistentClient(
            path=application_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user.

        Args:
            user_id: The user ID.

        Returns:
            Collection name string.
        """
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> "chromadb.Collection":
        """Get or create collection for user.

        Args:
            user_id: The user ID.

        Returns:
            ChromaDB collection.
        """
        collection_name = self._get_collection_name(user_id)
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=collection_name,
            embedding_function=self._cached_embedding_fn,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.

        Raises:
            Exception: If archiving fails.
        """
        if not messages:
            return None

        try:
            collection = await self._get_or_create_collection(user_id)

            await asyncio.to_thread(
                collection.add,
                metadatas=[
                    {"role": msg.role, "timestamp": datetime.now().isoformat(), "message_id": str(msg.id)}
                    for msg in messages
                ],
                ids=[str(msg.id) for msg in messages],
                documents=[msg.content for msg in messages],
            )
            logger.debug(f"Archived {len(messages)} messages for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to archive messages for user {user_id}: {e}")
            raise

    async def search(self, user_id: int, query: str, n_results: int) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return.

        Returns:
            List of search results.
        """
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=n_results,
            )

            formatted_results: list[MemorySearchResult] = []
            documents = result.get("documents")
            metadatas = result.get("metadatas")
            if documents and metadatas and documents[0]:
                for i, doc in enumerate(documents[0]):
                    metadata = metadatas[0][i]
                    formatted_results.append(
                        MemorySearchResult(
                            content=doc,
                            role=str(metadata.get("role", "")),
                            timestamp=str(metadata.get("timestamp", "")),
                            message_id=str(metadata.get("message_id", "")),
                        )
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        This method performs cleanup of old conversation records to manage storage
        and comply with data retention policies. It iterates through all user collections
        and removes messages that exceed the retention period.

        Args:
            retention_days: Number of days to retain messages.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)

        try:
            collections = await asyncio.to_thread(self._client.list_collections)

            for collection in [c for c in collections if c.name.startswith("user_")]:
                result = await asyncio.to_thread(collection.get)
                if not result:
                    continue

                ids = result.get("ids", [])
                metadatas = result.get("metadatas")
                to_delete = []
                if not metadatas:
                    return None
                for i, meta in enumerate(metadatas):
                    ts = str(meta.get("timestamp", ""))
                    try:
                        if ts and datetime.fromisoformat(ts) < cutoff:
                            to_delete.append(ids[i])
                    except (ValueError, TypeError):
                        pass

                if to_delete:
                    await asyncio.to_thread(collection.delete, ids=to_delete)
                    logger.info(f"Deleted {len(to_delete)} messages from {collection.name}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")


class AsyncChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using AsyncHttpClient (asynchronous, for external server).

    This class uses ChromaDB's async HTTP client to connect to an external
    ChromaDB server. Suitable for distributed deployments.
    """

    def __init__(self) -> None:
        """Initialize ChromaDB async client for external server."""
        self._client: chromadb.AsyncClientAPI | None = None
        self._cached_embedding_fn: EmbeddingFunction = _get_embedding_function()
        logger.info(
            f"ChromaDB: using async external mode ("
            f"{application_settings.chroma_host}:{application_settings.chroma_port})"
        )

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        """Get or create async client (lazy initialization).

        Returns:
            ChromaDB async client.
        """
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=application_settings.chroma_host,
                port=application_settings.chroma_port,
            )
        return self._client

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user.

        Args:
            user_id: The user ID.

        Returns:
            Collection name string.
        """
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> AsyncCollection:
        """Get or create collection for user.

        Args:
            user_id: The user ID.

        Returns:
            ChromaDB async collection.
        """
        collection_name = self._get_collection_name(user_id)
        client = await self._get_client()
        return await client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._cached_embedding_fn,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.

        Raises:
            Exception: If archiving fails.
        """
        if not messages:
            return None

        try:
            collection = await self._get_or_create_collection(user_id)

            await collection.add(
                metadatas=[
                    {"role": msg.role, "timestamp": datetime.now().isoformat(), "message_id": str(msg.id)}
                    for msg in messages
                ],
                ids=[str(msg.id) for msg in messages],
                documents=[msg.content for msg in messages],
            )
            logger.debug(f"Archived {len(messages)} messages for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to archive messages for user {user_id}: {e}")
            raise

    async def search(self, user_id: int, query: str, n_results: int) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return.

        Returns:
            List of search results.
        """
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            formatted_results: list[MemorySearchResult] = []
            documents = result.get("documents")
            metadatas = result.get("metadatas")
            if documents and metadatas and documents[0]:
                for i, doc in enumerate(documents[0]):
                    metadata = metadatas[0][i]
                    formatted_results.append(
                        MemorySearchResult(
                            content=doc,
                            role=str(metadata.get("role", "")),
                            timestamp=str(metadata.get("timestamp", "")),
                            message_id=str(metadata.get("message_id", "")),
                        )
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        This method performs cleanup of old conversation records to manage storage
        and comply with data retention policies. It iterates through all user collections
        and removes messages that exceed the retention period.

        Args:
            retention_days: Number of days to retain messages.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)

        try:
            client = await self._get_client()
            collections = await client.list_collections()

            for collection in [c for c in collections if c.name.startswith("user_")]:
                result = await collection.get()
                if not result:
                    continue

                ids = result.get("ids", [])
                metadatas = result.get("metadatas", [])
                to_delete = []
                if not metadatas:
                    return
                for i, meta in enumerate(metadatas):
                    ts = str(meta.get("timestamp", ""))
                    try:
                        if ts and datetime.fromisoformat(ts) < cutoff:
                            to_delete.append(ids[i])
                    except (ValueError, TypeError):
                        pass

                if to_delete:
                    await collection.delete(ids=to_delete)
                    logger.info(f"Deleted {len(to_delete)} messages from {collection.name}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")


def create_memory() -> LongConversationMemory | None:
    """Create memory instance based on ChromaDB configuration.

    Returns:
        LongConversationMemory object.
    """

    if not application_settings.is_chroma_configured:
        logger.info("ChromaDB not configured, semantic memory disabled")
        return None

    try:
        if application_settings.chroma_host:
            # External mode - use async client
            memory: LongConversationMemory = AsyncChromaLongConversationMemory()
        else:
            # Embedded mode - use sync client
            memory = ChromaLongConversationMemory()

        logger.info("Semantic memory initialized successfully")
        return memory
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return None


memory: LongConversationMemory | None = create_memory()
