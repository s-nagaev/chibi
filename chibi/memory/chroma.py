import asyncio
from datetime import datetime, timedelta

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from loguru import logger

from chibi.config import application_settings
from chibi.memory.abstract import LongConversationMemory
from chibi.models import Message


class ChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using embedded PersistentClient (synchronous)."""

    def __init__(self) -> None:
        """Initialize ChromaDB embedded client."""
        self._cached_embedding_fn = DefaultEmbeddingFunction()
        self._client = chromadb.PersistentClient(
            path=application_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user."""
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> "chromadb.Collection":
        """Get or create collection for user."""
        collection_name = self._get_collection_name(user_id)
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=collection_name,
            embedding_function=self._cached_embedding_fn,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB."""
        if not messages:
            return

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

    async def search(self, user_id: int, query: str, n_results: int) -> list[dict]:
        """Search archived messages by semantic similarity."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=n_results,
            )

            formatted_results = []
            if result.get("documents") and result["documents"][0]:
                for i, doc in enumerate(result["documents"][0]):
                    formatted_results.append({"content": doc, **result["metadatas"][0][i]})

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days."""
        cutoff = datetime.now() - timedelta(days=retention_days)

        try:
            collections = await asyncio.to_thread(self._client.list_collections)

            for coll in [c for c in collections if c.name.startswith("user_")]:
                try:
                    result = await asyncio.to_thread(coll.get)

                    if not result or not result.get("ids"):
                        continue

                    ids_to_delete = []
                    metadatas = result.get("metadatas", [])
                    for i, metadata in enumerate(metadatas):
                        timestamp_str = metadata.get("timestamp")
                        try:
                            if (timestamp_str and datetime.fromisoformat(timestamp_str) < cutoff) or not timestamp_str:
                                ids_to_delete.append(result["ids"][i])
                        except (ValueError, TypeError):
                            ids_to_delete.append(result["ids"][i])

                    if ids_to_delete:
                        await asyncio.to_thread(coll.delete, ids=ids_to_delete)
                        logger.info(f"Deleted {len(ids_to_delete)} old messages from {coll.name}")

                except Exception as e:
                    logger.warning(f"Failed to process collection {coll.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")
            raise


class AsyncChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using AsyncHttpClient (asynchronous, for external server)."""

    def __init__(self) -> None:
        """Initialize ChromaDB async client for external server."""
        self._client: chromadb.AsyncClient | None = None
        self._cached_embedding_fn = DefaultEmbeddingFunction()
        logger.info(
            f"ChromaDB: using async external mode ("
            f"{application_settings.chroma_host}:{application_settings.chroma_port})"
        )

    async def _get_client(self) -> "chromadb.AsyncClient":
        """Get or create async client (lazy initialization)."""
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=application_settings.chroma_host,
                port=application_settings.chroma_port,
            )
        return self._client

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user."""
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> "chromadb.AsyncCollection":
        """Get or create collection for user."""
        collection_name = self._get_collection_name(user_id)
        client = await self._get_client()
        return await client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._cached_embedding_fn,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB."""
        if not messages:
            return

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

    async def search(self, user_id: int, query: str, n_results: int) -> list[dict]:
        """Search archived messages by semantic similarity."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            formatted_results = []
            if result.get("documents") and result["documents"][0]:
                for i, doc in enumerate(result["documents"][0]):
                    formatted_results.append({"content": doc, **result["metadatas"][0][i]})

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days."""
        cutoff = datetime.now() - timedelta(days=retention_days)

        try:
            client = await self._get_client()
            collections = await client.list_collections()

            for coll in [c for c in collections if c.name.startswith("user_")]:
                try:
                    result = await coll.get()

                    if not result or not result.get("ids"):
                        continue

                    ids_to_delete = []
                    metadatas = result.get("metadatas", [])
                    for i, metadata in enumerate(metadatas):
                        timestamp_str = metadata.get("timestamp")
                        try:
                            if (timestamp_str and datetime.fromisoformat(timestamp_str) < cutoff) or not timestamp_str:
                                ids_to_delete.append(result["ids"][i])
                        except (ValueError, TypeError):
                            ids_to_delete.append(result["ids"][i])

                    if ids_to_delete:
                        await coll.delete(ids=ids_to_delete)
                        logger.info(f"Deleted {len(ids_to_delete)} old messages from {coll.name}")

                except Exception as e:
                    logger.warning(f"Failed to process collection {coll.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")
            raise


def create_memory() -> LongConversationMemory | None:
    """Create memory instance based on ChromaDB configuration."""

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


def register_memory_tool() -> None:
    """Register the search tool if memory is available."""
    if memory is not None:
        from chibi.services.providers.tools import RegisteredChibiTools
        from chibi.services.providers.tools.memory import SearchInConversationHistoryTool

        RegisteredChibiTools.register(SearchInConversationHistoryTool)
        logger.info("SearchInConversationHistoryTool registered")


memory: LongConversationMemory | None = create_memory()
