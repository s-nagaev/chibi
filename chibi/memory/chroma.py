"""ChromaDB memory implementation with batch metadata and context retrieval."""
import asyncio
import uuid
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
from chibi.memory.abstract import (
    EDGE_THRESHOLD,
    LongConversationMemory,
    MemorySearchResult,
)
from chibi.memory.batch import Batch, BatchManager, create_user_batch_manager
from chibi.models import Message
from chibi.services.lock_manager import LockManager


class InternalChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using embedded PersistentClient (synchronous).

    This class uses ChromaDB's persistent client to store conversation history
    locally on disk. Supports batch metadata and context retrieval.

    Features:
        - Batch accumulation with metadata (batch_id, msg_pos, prev_batch_id)
        - Context retrieval (neighboring batches around semantic search hits)
        - No full scan required
        - Restart-safe (ULID-based batch IDs)
    """

    def __init__(self, embedding_function: EmbeddingFunction = DefaultEmbeddingFunction()) -> None:
        """Initialize ChromaDB embedded client."""
        self.embedding_function = embedding_function
        self._client = chromadb.PersistentClient(
            path=application_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._batch_managers: dict[int, BatchManager] = {}
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    def _get_batch_manager(self, user_id: int) -> BatchManager:
        """Get or create batch manager for user."""
        if user_id not in self._batch_managers:
            self._batch_managers[user_id] = create_user_batch_manager(
                user_id,
                application_settings.batch_size ,
            )
        return self._batch_managers[user_id]

    def _get_batch_size(self) -> int:
        """Get configured batch size."""
        return application_settings.batch_size 

    async def _get_or_create_collection(self, user_id: int) -> "chromadb.Collection":
        """Get or create a collection for user."""
        collection_name = self._get_collection_name(user_id)
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB with batch metadata.

        Messages are accumulated in batches. When batch is full, it's archived
        with metadata for context retrieval.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
        """
        if not messages:
            return

        batch_manager = self._get_batch_manager(user_id)

        # Process each message through batch manager
        for msg in messages:
            batch, is_full = batch_manager.add_message(user_id, msg)

            # Archive full batch immediately
            if is_full and batch:
                await self._archive_batch(batch, user_id)
        
        # Flush remaining partial batch
        remaining = batch_manager.flush()
        if remaining:
            await self._archive_batch(remaining, user_id)

        logger.debug(f"Archived messages for user {user_id}")

    async def _archive_batch(self, batch: Batch, user_id: int) -> None:
        """Archive a single batch to ChromaDB with metadata.

        Args:
            batch: Batch object with messages.
            user_id: The user ID.
        """
        if not batch.messages:
            return

        # Build metadata for each message in batch
        metadatas: Metadatas = []
        for i, msg in enumerate(batch.messages):
            metadatas.append(
                {
                    "message_id": str(msg.id),
                    "batch_id": batch.batch_id,
                    "msg_pos": i,
                    "prev_batch_id": batch.prev_batch_id,
                    "role": msg.role,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        ids = [str(msg.id) for msg in batch.messages]
        documents = [msg.content for msg in batch.messages]

        lock = await LockManager().get_lock(key=str(user_id))
        async with lock:
            try:
                collection = await self._get_or_create_collection(user_id)
                await asyncio.to_thread(
                    collection.add,
                    metadatas=metadatas,
                    ids=ids,
                    documents=documents,
                )
                logger.debug(f"Archived batch {batch.batch_id} ({len(batch.messages)} messages)")
            except Exception as e:
                logger.error(f"Failed to archive batch {batch.batch_id}: {e}")

    async def search(self, user_id: int, query: str, n_results: int) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Performs semantic search first, then retrieves surrounding context
        from neighboring batches if the hit is near batch edges.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Maximum number of results to return per batch.

        Returns:
            List of search results with context.
        """
        try:
            # Step 1: Semantic search
            hit = await self._semantic_search(user_id, query, n_results)
            if not hit:
                return []

            # Step 2: Get batch of found message
            hit_batch_id = hit.get("batch_id")
            hit_msg_pos = hit.get("msg_pos")
            hit_prev_batch_id = hit.get("prev_batch_id")

            if not hit_batch_id:
                return [{"content": hit["content"], "role": hit["role"], "timestamp": hit["timestamp"],
                         "message_id": hit["message_id"], "batch_id": None, "msg_pos": None, "prev_batch_id": None}]

            # Step 3: Get current batch
            context_messages = await self._get_batch_by_id(user_id, hit_batch_id)

            # Step 4: Determine if we need neighboring batches
            batch_size = self._get_batch_size()

            # Near beginning - add previous batch
            if hit_msg_pos is not None and hit_msg_pos <= EDGE_THRESHOLD and hit_prev_batch_id:
                prev_batch = await self._get_batch_by_prev_id(user_id, hit_prev_batch_id)
                if prev_batch:
                    context_messages = prev_batch + context_messages

            # Near end - add next batch
            if hit_msg_pos is not None and hit_msg_pos >= batch_size - EDGE_THRESHOLD - 1:
                next_batch = await self._get_next_batch(user_id, hit_batch_id)
                if next_batch:
                    context_messages.extend(next_batch)

            # Sort by (batch_id, msg_pos)
            context_messages.sort(key=lambda x: (x.get("batch_id", ""), x.get("msg_pos", 0)))

            return context_messages

        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int
    ) -> MemorySearchResult | None:
        """Perform semantic search."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=1,  # Get top hit only
            )

            documents = result.get("documents")
            metadatas = result.get("metadatas")

            if documents and metadatas and documents[0]:
                metadata = metadatas[0][0]
                return MemorySearchResult(
                    content=documents[0][0],
                    role=str(metadata.get("role", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    message_id=str(metadata.get("message_id", "")),
                    batch_id=str(metadata.get("batch_id", "")),
                    msg_pos=int(metadata.get("msg_pos", -1)),
                    prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                )
            return None

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return None

    async def _get_batch_by_id(
        self, user_id: int, batch_id: str
    ) -> list[MemorySearchResult]:
        """Get all messages in a batch."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await asyncio.to_thread(
                collection.get,
                where={"batch_id": batch_id},
            )

            return self._format_batch_results(result)

        except Exception as e:
            logger.error(f"Failed to get batch {batch_id}: {e}")
            return []

    async def _get_batch_by_prev_id(
        self, user_id: int, prev_batch_id: str
    ) -> list[MemorySearchResult]:
        """Get batch by prev_batch_id reference."""
        return await self._get_batch_by_id(user_id, prev_batch_id)

    async def _get_next_batch(
        self, user_id: int, current_batch_id: str
    ) -> list[MemorySearchResult]:
        """Get next batch (where prev_batch_id == current_batch_id)."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await asyncio.to_thread(
                collection.get,
                where={"prev_batch_id": current_batch_id},
            )

            return self._format_batch_results(result)

        except Exception as e:
            logger.error(f"Failed to get next batch: {e}")
            return []

    def _format_batch_results(self, result) -> list[MemorySearchResult]:
        """Format batch query results."""
        formatted: list[MemorySearchResult] = []
        documents = result.get("documents")
        metadatas = result.get("metadatas")

        if documents and metadatas and documents[0]:
            for i, doc in enumerate(documents[0]):
                metadata = metadatas[0][i]
                formatted.append(
                    MemorySearchResult(
                        content=doc,
                        role=str(metadata.get("role", "")),
                        timestamp=str(metadata.get("timestamp", "")),
                        message_id=str(metadata.get("message_id", "")),
                        batch_id=str(metadata.get("batch_id", "")),
                        msg_pos=int(metadata.get("msg_pos", -1)),
                        prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                    )
                )

        return formatted

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

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
                    return
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
    """ChromaDB implementation using AsyncHttpClient (asynchronous).

    Same features as InternalChromaLongConversationMemory but uses
    async HTTP client for external ChromaDB server.
    """

    def __init__(self, embedding_function: EmbeddingFunction = DefaultEmbeddingFunction()) -> None:
        """Initialize ChromaDB async client for external server."""
        self._client: chromadb.AsyncClientAPI | None = None
        self.embedding_function = embedding_function
        self._batch_managers: dict[int, BatchManager] = {}
        logger.info(
            f"ChromaDB: using async external mode ("
            f"{application_settings.chroma_host}:{application_settings.chroma_port})"
        )

    def _get_batch_manager(self, user_id: int) -> BatchManager:
        """Get or create batch manager for user."""
        if user_id not in self._batch_managers:
            self._batch_managers[user_id] = create_user_batch_manager(
                user_id,
                application_settings.batch_size ,
            )
        return self._batch_managers[user_id]

    def _get_batch_size(self) -> int:
        """Get configured batch size."""
        return application_settings.batch_size 

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        """Get or create async client (lazy initialization)."""
        if self._client is None:
            self._client = await chromadb.AsyncHttpClient(
                host=application_settings.chroma_host,
                port=application_settings.chroma_port,
            )
        return self._client

    async def _get_or_create_collection(self, user_id: int) -> AsyncCollection:
        """Get or create collection for user."""
        collection_name = self._get_collection_name(user_id)
        client = await self._get_client()
        return await client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB with batch metadata.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
        """
        if not messages:
            return

        batch_manager = self._get_batch_manager(user_id)

        for msg in messages:
            batch, is_full = batch_manager.add_message(user_id, msg)

            if is_full and batch:
                await self._archive_batch(batch, user_id)
        
        remaining = batch_manager.flush()
        if remaining:
            await self._archive_batch(remaining, user_id)

        logger.debug(f"Archived messages for user {user_id}")

    async def _archive_batch(self, batch: Batch, user_id: int) -> None:
        """Archive a single batch to ChromaDB with metadata."""
        if not batch.messages:
            return

        metadatas: Metadatas = []
        for i, msg in enumerate(batch.messages):
            metadatas.append(
                {
                    "message_id": str(msg.id),
                    "batch_id": batch.batch_id,
                    "msg_pos": i,
                    "prev_batch_id": batch.prev_batch_id,
                    "role": msg.role,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        ids = [str(msg.id) for msg in batch.messages]
        documents = [msg.content for msg in batch.messages]

        lock = await LockManager().get_lock(key=str(user_id))
        async with lock:
            try:
                collection = await self._get_or_create_collection(user_id)
                await collection.add(
                    metadatas=metadatas,
                    ids=ids,
                    documents=documents,
                )
                logger.debug(f"Archived batch {batch.batch_id} ({len(batch.messages)} messages)")
            except Exception as e:
                logger.error(f"Failed to archive batch {batch.batch_id}: {e}")

    async def search(self, user_id: int, query: str, n_results: int) -> list[MemorySearchResult]:
        """Search archived messages with context retrieval."""
        try:
            hit = await self._semantic_search(user_id, query, n_results)
            if not hit:
                return []

            hit_batch_id = hit.get("batch_id")
            hit_msg_pos = hit.get("msg_pos")
            hit_prev_batch_id = hit.get("prev_batch_id")

            if not hit_batch_id:
                return [{"content": hit["content"], "role": hit["role"], "timestamp": hit["timestamp"],
                         "message_id": hit["message_id"], "batch_id": None, "msg_pos": None, "prev_batch_id": None}]

            context_messages = await self._get_batch_by_id(user_id, hit_batch_id)
            batch_size = self._get_batch_size()

            if hit_msg_pos is not None and hit_msg_pos <= EDGE_THRESHOLD and hit_prev_batch_id:
                prev_batch = await self._get_batch_by_prev_id(user_id, hit_prev_batch_id)
                if prev_batch:
                    context_messages = prev_batch + context_messages

            if hit_msg_pos is not None and hit_msg_pos >= batch_size - EDGE_THRESHOLD - 1:
                next_batch = await self._get_next_batch(user_id, hit_batch_id)
                if next_batch:
                    context_messages.extend(next_batch)

            context_messages.sort(key=lambda x: (x.get("batch_id", ""), x.get("msg_pos", 0)))

            return context_messages

        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int
    ) -> MemorySearchResult | None:
        """Perform semantic search."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await collection.query(
                query_texts=[query],
                n_results=1,
            )

            documents = result.get("documents")
            metadatas = result.get("metadatas")

            if documents and metadatas and documents[0]:
                metadata = metadatas[0][0]
                return MemorySearchResult(
                    content=documents[0][0],
                    role=str(metadata.get("role", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    message_id=str(metadata.get("message_id", "")),
                    batch_id=str(metadata.get("batch_id", "")),
                    msg_pos=int(metadata.get("msg_pos", -1)),
                    prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                )
            return None

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return None

    async def _get_batch_by_id(
        self, user_id: int, batch_id: str
    ) -> list[MemorySearchResult]:
        """Get all messages in a batch."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await collection.get(where={"batch_id": batch_id})

            return self._format_batch_results(result)

        except Exception as e:
            logger.error(f"Failed to get batch {batch_id}: {e}")
            return []

    async def _get_batch_by_prev_id(
        self, user_id: int, prev_batch_id: str
    ) -> list[MemorySearchResult]:
        """Get batch by prev_batch_id reference."""
        return await self._get_batch_by_id(user_id, prev_batch_id)

    async def _get_next_batch(
        self, user_id: int, current_batch_id: str
    ) -> list[MemorySearchResult]:
        """Get next batch."""
        try:
            collection = await self._get_or_create_collection(user_id)
            result = await collection.get(where={"prev_batch_id": current_batch_id})

            return self._format_batch_results(result)

        except Exception as e:
            logger.error(f"Failed to get next batch: {e}")
            return []

    def _format_batch_results(self, result) -> list[MemorySearchResult]:
        """Format batch query results."""
        formatted: list[MemorySearchResult] = []
        documents = result.get("documents")
        metadatas = result.get("metadatas")

        if documents and metadatas and documents[0]:
            for i, doc in enumerate(documents[0]):
                metadata = metadatas[0][i]
                formatted.append(
                    MemorySearchResult(
                        content=doc,
                        role=str(metadata.get("role", "")),
                        timestamp=str(metadata.get("timestamp", "")),
                        message_id=str(metadata.get("message_id", "")),
                        batch_id=str(metadata.get("batch_id", "")),
                        msg_pos=int(metadata.get("msg_pos", -1)),
                        prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                    )
                )

        return formatted

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days."""
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

    embedding_function: EmbeddingFunction
    match application_settings.embedding_function:
        case "GEMINI":
            embedding_function = GoogleGeminiEmbeddingFunction()
        case "OPENAI":
            embedding_function = OpenAIEmbeddingFunction(api_key_env_var="OPENAI_API_KEY")
        case "MISTRALAI":
            embedding_function = MistralEmbeddingFunction(
                api_key_env_var="MISTRALAI_API_KEY",
                model="mistral-embed",
            )
        case _:
            embedding_function = DefaultEmbeddingFunction()

    try:
        if application_settings.chroma_host:
            conversation_memory = ExternalChromaLongConversationMemory(
                embedding_function=embedding_function
            )
        else:
            conversation_memory = InternalChromaLongConversationMemory(
                embedding_function=embedding_function
            )

        logger.info("Semantic memory initialized successfully")
        return conversation_memory

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

    return None


memory: LongConversationMemory | None = create_memory()