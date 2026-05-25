import asyncio
from datetime import datetime, timedelta

import chromadb
from chromadb import EmbeddingFunction, Metadatas
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    GoogleGeminiEmbeddingFunction,
    MistralEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from loguru import logger

from chibi.config import application_settings
from chibi.memory.abstract import LongConversationMemory, MemorySearchResult
from chibi.models import Message
from chibi.services.lock_manager import LockManager


class InternalChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using embedded PersistentClient (synchronous).

    This class uses ChromaDB's persistent client to store conversation history
    locally on disk. Suitable for single-instance deployments.
    """

    def __init__(self, embedding_function: EmbeddingFunction = DefaultEmbeddingFunction()) -> None:
        """Initialize ChromaDB embedded client."""
        self.embedding_function = embedding_function
        self._client = chromadb.PersistentClient(
            path=application_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    async def _get_or_create_collection(self, user_id: int) -> "chromadb.Collection":
        """Get or create a collection for user.

        Args:
            user_id: The user ID.

        Returns:
            ChromaDB collection.
        """
        collection_name = self._get_collection_name(user_id)
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
        """
        if not messages:
            return None

        meta: Metadatas = [
            {"role": msg.role, "timestamp": datetime.now().isoformat(), "message_id": str(msg.id)} for msg in messages
        ]
        ids = [str(msg.id) for msg in messages]
        documents = [msg.content for msg in messages]

        lock = await LockManager().get_lock(key=str(user_id))

        async with lock:
            try:
                collection = await self._get_or_create_collection(user_id)

                await asyncio.to_thread(
                    collection.add,
                    metadatas=meta,
                    ids=ids,
                    documents=documents,
                )
                logger.debug(f"Archived {len(messages)} messages for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to archive messages for user {user_id}: {e}")

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


class ExternalChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using AsyncHttpClient (asynchronous, for external server).

    This class uses ChromaDB's async HTTP client to connect to an external
    ChromaDB server. Suitable for distributed deployments.
    """

    def __init__(self, embedding_function: EmbeddingFunction = DefaultEmbeddingFunction()) -> None:
        """Initialize ChromaDB async client for external server."""
        self._client: chromadb.AsyncClientAPI | None = None
        self.embedding_function = embedding_function
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
            embedding_function=self.embedding_function,
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

        meta: Metadatas = [
            {"role": msg.role, "timestamp": datetime.now().isoformat(), "message_id": str(msg.id)} for msg in messages
        ]
        ids = [str(msg.id) for msg in messages]
        documents = [msg.content for msg in messages]

        lock = await LockManager().get_lock(key=str(user_id))
        async with lock:
            try:
                collection = await self._get_or_create_collection(user_id)

                await collection.add(
                    metadatas=meta,
                    ids=ids,
                    documents=documents,
                )
                logger.debug(f"Archived {len(messages)} messages for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to archive messages for user {user_id}: {e}")

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
    conversation_memory: LongConversationMemory
    embedding_function: EmbeddingFunction
    match application_settings.embedding_function:
        case "GEMINI":
            embedding_function = GoogleGeminiEmbeddingFunction()
        case "OPENAI":
            embedding_function = OpenAIEmbeddingFunction(api_key_env_var="OPENAI_API_KEY")
        case "MISTRALAI":
            embedding_function = MistralEmbeddingFunction(api_key_env_var="MISTRALAI_API_KEY", model="mistral-embed")
        case _:
            embedding_function = DefaultEmbeddingFunction()

    try:
        if application_settings.chroma_host:
            # External mode - use async client
            conversation_memory = ExternalChromaLongConversationMemory(embedding_function=embedding_function)
        else:
            # Embedded mode - use sync client
            conversation_memory = InternalChromaLongConversationMemory(embedding_function=embedding_function)

        logger.info("Semantic memory initialized successfully")
        return conversation_memory
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

    return None


memory: LongConversationMemory | None = create_memory()
