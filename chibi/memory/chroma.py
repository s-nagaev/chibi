"""ChromaDB memory implementation with batch metadata and context retrieval."""

import asyncio
from datetime import datetime, timedelta
from typing import cast

import chromadb
import ulid
from chromadb import Collection, EmbeddingFunction, GetResult, Metadata, Where
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    GoogleGeminiEmbeddingFunction,
    MistralEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from loguru import logger
from pydantic import BaseModel

from chibi.config import application_settings
from chibi.exceptions import (
    ChromaArchiveError,
    ChromaCollectionError,
    ChromaConnectionError,
    ChromaSearchError,
)
from chibi.memory.abstract import (
    EDGE_THRESHOLD,
    LongConversationMemory,
    MemorySearchResult,
)
from chibi.models import Message
from chibi.services.lock_manager import LockManager


class ArchiveState(BaseModel):
    batch_id: str
    prev_batch_id: str | None = None
    next_msg_pos: int
    token_count: int


class InternalChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using embedded PersistentClient.

    This class uses ChromaDB's persistent client to store conversation history
    locally on disk. Supports context retrieval.

    Features:
        - Per-message persistence with batch metadata (batch_id, msg_pos, prev_batch_id)
        - Context retrieval (neighboring batches around semantic search hits)
        - No full scan required
        - Restart-safe: prev_batch_id is loaded from DB on first archive call
        - Per-thread batch: each thread_id has separate batch tracking
    """

    def __init__(self, embedding_function: EmbeddingFunction = DefaultEmbeddingFunction()) -> None:
        """Initialize ChromaDB embedded client."""
        self.embedding_function = embedding_function
        self._client = chromadb.PersistentClient(
            path=application_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Per-(user, thread) archive state: tracks current batch metadata and token counts
        self._archive_state: dict[tuple[int, int], ArchiveState] = {}
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    @staticmethod
    def _get_batch_token_limit() -> int:
        """Get configured batch token limit."""
        return application_settings.batch_token_limit

    @staticmethod
    def _generate_batch_id() -> str:
        """Generate chronologically ordered unique batch ID via ULID."""
        return str(ulid.ulid())

    async def _get_last_batch_id(self, user_id: int, thread_id: int = 0) -> str | None:
        """Get batch_id of the most recent message for this user+thread in ChromaDB.

        Filters to the last 7 days using the numeric, include=["metadatas"] to avoid pulling documents.

        Args:
            user_id: The user ID.
            thread_id: The thread ID (default: 0).

        Returns:
            The most recent batch_id, or None if no recent messages or on error.
        """
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            one_week_ago = (datetime.now() - timedelta(days=7)).timestamp()

            where_filter = cast(Where, {"timestamp_unix": {"$gte": one_week_ago}})
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                include=["metadatas"],
            )
            metadatas = result.get("metadatas")
            if not metadatas:
                return None
            latest = max(metadatas, key=lambda m: float(str(m.get("timestamp_unix", 0))))
            bid = str(latest.get("batch_id"))
            return bid if bid else None
        except Exception as e:
            logger.error(f"Failed to get last batch_id for user {user_id}, thread {thread_id}: {e}")
            return None

    async def _get_or_create_collection(self, user_id: int, thread_id: int = 0) -> Collection:
        """Get or create a collection for user+thread.

        Args:
            user_id: The user ID.
            thread_id: The thread ID (default: 0).

        Returns:
            ChromaDB collection instance.

        Raises:
            ChromaCollectionError: If collection access fails.
        """
        collection_name = self._get_collection_name(user_id, thread_id)
        try:
            return await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            raise ChromaCollectionError(f"Failed to get or create collection '{collection_name}': {e}") from e

    async def get_or_create_archive_state(self, user_id: int, thread_id: int = 0) -> ArchiveState:
        """Get or create per-(user, thread) archive state.

        On first call for a user+thread, queries ChromaDB for the last batch_id
        to enable proper batch chaining across restarts.

        Args:
            user_id: The user ID.
            thread_id: The thread ID (default: 0).

        Returns:
            ArchiveState.
        """
        key = (user_id, thread_id)
        if key not in self._archive_state:
            last_batch_id = await self._get_last_batch_id(user_id, thread_id)
            self._archive_state[key] = ArchiveState(
                batch_id=self._generate_batch_id(),
                prev_batch_id=last_batch_id,
                next_msg_pos=0,
                token_count=0,
            )
        return self._archive_state[key]

    async def update_archive_state(self, user_id: int, thread_id: int, tokens_to_add: int) -> None:
        """Increment token count and message position; rotate batch on overflow.

        When the accumulated token count exceeds the configured batch limit,
        the current batch is sealed (prev_batch_id updated) and a new batch
        is started with zero counters.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.
            tokens_to_add: Number of tokens to add to the current batch count.
        """
        state = await self.get_or_create_archive_state(user_id, thread_id)
        state.token_count += tokens_to_add
        state.next_msg_pos += 1

        if state.token_count > self._get_batch_token_limit():
            state.prev_batch_id = state.batch_id
            state.batch_id = self._generate_batch_id()
            state.next_msg_pos = 0
            state.token_count = 0
        return None

    async def archive(self, user_id: int, messages: list[Message], thread_id: int = 0) -> None:
        """Archive messages to ChromaDB with batch metadata.

        Each message is saved immediately. Token counting and batch_id rotation
        happen inline: msg_pos increments while total tokens stay under limit;
        on overflow the next message starts a new batch with prev_batch_id
        pointing to the previous one.

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
            thread_id: Thread ID (default: 0).

        Raises:
            ChromaArchiveError: If any message fails to archive.
        """
        if not messages:
            return None

        lock = await LockManager().get_lock(key=f"{user_id}:{thread_id}")
        async with lock:
            state = await self.get_or_create_archive_state(user_id, thread_id)

            for msg in messages:
                msg_tokens = msg.estimate_tokens

                # Save message first with current batch metadata
                pos = state.next_msg_pos
                await self._archive_message(
                    msg=msg,
                    batch_id=state.batch_id,
                    msg_pos=pos,
                    prev_batch_id=state.prev_batch_id,
                    user_id=user_id,
                    thread_id=thread_id,
                    token_count=state.token_count,
                )
                await self.update_archive_state(user_id, thread_id, msg_tokens)
        return None

    async def _archive_message(
        self,
        msg: Message,
        batch_id: str,
        msg_pos: int,
        prev_batch_id: str | None,
        user_id: int,
        thread_id: int = 0,
        token_count: int = 0,
    ) -> None:
        """Archive a single message to ChromaDB with batch metadata.

        Args:
            msg: Message to archive.
            batch_id: Current batch ID.
            msg_pos: Position of message within the batch.
            prev_batch_id: Previous batch ID (for context retrieval).
            user_id: The user ID.
            thread_id: Thread ID (default: 0).
            token_count: Current accumulated token count for logging.

        Raises:
            ChromaArchiveError: If the ChromaDB add operation fails.
        """
        now = datetime.now()
        metadata: Metadata = {
            "message_id": str(msg.id),
            "batch_id": batch_id,
            "msg_pos": msg_pos,
            "prev_batch_id": prev_batch_id or "",
            "thread_id": str(thread_id),
            "role": msg.role,
            "timestamp": now.isoformat(),
            "timestamp_unix": now.timestamp(),
        }

        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            await asyncio.to_thread(
                collection.add,
                metadatas=[metadata],
                ids=[str(msg.id)],
                documents=[msg.content],
            )
            logger.debug(
                f"Archived message {msg.id} in batch {batch_id} at pos {msg_pos} with avr_tokens {token_count}"
            )
        except Exception as e:
            logger.error(f"Failed to archive message {msg.id}: {e}")
            raise ChromaArchiveError(f"Failed to archive message {msg.id}: {e}") from e

    async def search(self, user_id: int, query: str, n_results: int, thread_id: int = 0) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Performs semantic search first, then retrieves surrounding context
        from neighboring batches if the hit is near batch edges.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Max results for semantic search (top hit used for context).
            thread_id: Thread ID (default: 0).

        Returns:
            List of search results with context; empty list on error or no matches.
        """
        try:
            # Step 1: Semantic search
            hit = await self._semantic_search(user_id, query, n_results, thread_id)
            if not hit:
                return []

            # Step 2: Get batch of found message
            hit_batch_id = hit.get("batch_id")
            hit_msg_pos = hit.get("msg_pos")
            hit_prev_batch_id = hit.get("prev_batch_id")

            if not hit_batch_id:
                return [
                    {
                        "content": hit["content"],
                        "role": hit["role"],
                        "timestamp": hit["timestamp"],
                        "message_id": hit["message_id"],
                        "batch_id": None,
                        "msg_pos": None,
                        "prev_batch_id": None,
                        "thread_id": thread_id,
                    }
                ]

            # Step 3: Get current batch
            context_messages = await self._get_batch_by_field(user_id, hit_batch_id, thread_id=thread_id)

            # Get actual batch size in database (handles partial batches correctly)
            current_batch_count = len(context_messages)

            # Near beginning: add previous batch (only if valid prev_batch_id exists)
            if (
                current_batch_count > 0
                and hit_msg_pos is not None
                and hit_msg_pos <= EDGE_THRESHOLD
                and hit_prev_batch_id
            ):
                prev_batch = await self._get_batch_by_field(user_id, hit_prev_batch_id, thread_id=thread_id)
                if prev_batch:
                    context_messages = prev_batch + context_messages
                    current_batch_count = len(context_messages)

            # Near end: use ACTUAL batch count
            if (
                hit_msg_pos is not None
                and current_batch_count > 0
                and hit_msg_pos >= current_batch_count - EDGE_THRESHOLD - 1
            ):
                next_batch = await self._get_batch_by_field(
                    user_id=user_id,
                    batch_id=hit_batch_id,
                    field="prev_batch_id",
                    thread_id=thread_id,
                )
                if next_batch:
                    context_messages.extend(next_batch)

            # Sort by (batch_id, msg_pos) - ULID ensures chronological order
            context_messages.sort(key=lambda x: (x.get("batch_id", ""), x.get("msg_pos", 0)))

            return context_messages

        except ChromaSearchError:
            return []
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int, thread_id: int = 0
    ) -> MemorySearchResult | None:
        """Perform semantic search.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Max results (only top hit used).
            thread_id: Thread ID (default: 0).

        Returns:
            Best matching MemorySearchResult or None if no hits.

        Raises:
            ChromaSearchError: If the search query fails.
        """
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            result = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=1,
            )

            documents = result.get("documents")
            metadatas = result.get("metadatas")

            if documents and metadatas and len(documents) > 0 and len(documents[0]) > 0:
                metadata = metadatas[0][0]
                return MemorySearchResult(
                    content=documents[0][0],
                    role=str(metadata.get("role", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    message_id=str(metadata.get("message_id", "")),
                    batch_id=str(metadata.get("batch_id", "")),
                    msg_pos=int(str(metadata.get("msg_pos", -1))),
                    prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                    thread_id=int(str(metadata.get("thread_id", 0))),
                )
            return None

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise ChromaSearchError(f"Semantic search failed: {e}") from e

    async def _get_batch_by_field(
        self,
        user_id: int,
        batch_id: str,
        field: str = "batch_id",
        thread_id: int = 0,
    ) -> list[MemorySearchResult]:
        """Get all messages matching a batch-related metadata field.

        Used both for direct batch lookup (field="batch_id") and reverse
        lookup (field="prev_batch_id") to find the next batch.

        Args:
            user_id: The user ID.
            batch_id: Value to match against the metadata field.
            field: Metadata field name to filter by ("batch_id" or "prev_batch_id").
            thread_id: Thread ID (default: 0).

        Returns:
            List of formatted search results; empty list on error or no matches.
        """
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            result = await asyncio.to_thread(
                collection.get,
                where={field: batch_id},
            )

            return self._format_batch_results(result)

        except Exception as e:
            logger.error(f"Failed to get batch {batch_id}: {e}")
            return []

    def _format_batch_results(self, result: GetResult) -> list[MemorySearchResult]:
        """Format raw ChromaDB collection.get() result into search results.

        Args:
            result: Raw result from ChromaDB (GetResult / QueryResult TypedDict).

        Returns:
            List of formatted MemorySearchResult dicts; empty list if no data.
        """
        formatted: list[MemorySearchResult] = []
        documents = result.get("documents")
        metadatas = result.get("metadatas")

        if documents and metadatas:
            for doc, metadata in zip(documents, metadatas):
                formatted.append(
                    MemorySearchResult(
                        content=doc,
                        role=str(metadata.get("role", "")),
                        timestamp=str(metadata.get("timestamp", "")),
                        message_id=str(metadata.get("message_id", "")),
                        batch_id=str(metadata.get("batch_id", "")),
                        msg_pos=int(str(metadata.get("msg_pos", -1))),
                        prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                        thread_id=int(str(metadata.get("thread_id", 0))),
                    )
                )

        return formatted

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        Args:
            retention_days: Number of days to retain messages.

        Raises:
            ChromaDeleteError: If cleanup fails.
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
                    continue
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
        # Per-(user, thread) archive state
        self._archive_state: dict[tuple[int, int], ArchiveState] = {}
        logger.info(
            f"ChromaDB: using async external mode ("
            f"{application_settings.chroma_host}:{application_settings.chroma_port})"
        )

    def _get_batch_token_limit(self) -> int:
        """Get configured batch token limit."""
        return application_settings.batch_token_limit

    def _generate_batch_id(self) -> str:
        """Generate chronologically ordered unique batch ID via ULID."""
        return str(ulid.ulid())

    async def _get_last_batch_id(self, user_id: int, thread_id: int = 0) -> str | None:
        """Get batch_id of the most recent message for this user+thread in ChromaDB."""
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            one_week_ago = (datetime.now() - timedelta(days=7)).timestamp()

            where_filter = cast(Where, {"timestamp_unix": {"$gte": one_week_ago}})
            result = await collection.get(
                where=where_filter,
                include=["metadatas"],
            )
            metadatas = result.get("metadatas")
            if not metadatas:
                return None
            latest = max(
                metadatas,
                key=lambda m: float(str(m.get("timestamp_unix", 0))),
            )
            bid = str(latest.get("batch_id", ""))
            return bid if bid else None
        except Exception as e:
            logger.error(f"Failed to get last batch_id for user {user_id}, thread {thread_id}: {e}")
            return None

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        """Get or create async client (lazy initialization)."""
        if self._client is None:
            try:
                self._client = await chromadb.AsyncHttpClient(
                    host=application_settings.chroma_host,
                    port=application_settings.chroma_port,
                )
            except Exception as e:
                raise ChromaConnectionError(
                    f"Failed to connect to ChromaDB at "
                    f"{application_settings.chroma_host}:{application_settings.chroma_port}: {e}"
                ) from e
        return self._client

    async def _get_or_create_collection(self, user_id: int, thread_id: int = 0) -> AsyncCollection:
        """Get or create collection for user+thread."""
        collection_name = self._get_collection_name(user_id, thread_id)
        client = await self._get_client()
        try:
            return await client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            raise ChromaCollectionError(f"Failed to get or create collection '{collection_name}': {e}") from e

    async def archive(self, user_id: int, messages: list[Message], thread_id: int = 0) -> None:
        """Archive messages to ChromaDB with batch metadata."""
        if not messages:
            return None

        lock = await LockManager().get_lock(key=f"{user_id}:{thread_id}")
        async with lock:
            key = (user_id, thread_id)
            if key not in self._archive_state:
                last_batch_id = await self._get_last_batch_id(user_id, thread_id)
                self._archive_state[key] = ArchiveState(
                    batch_id=self._generate_batch_id(),
                    prev_batch_id=last_batch_id,
                    next_msg_pos=0,
                    token_count=0,
                )

            state = self._archive_state[key]
            batch_limit = self._get_batch_token_limit()

            for msg in messages:
                msg_tokens = msg.estimate_tokens
                pos = state.next_msg_pos
                await self._archive_message(
                    msg=msg,
                    batch_id=state.batch_id,
                    msg_pos=pos,
                    prev_batch_id=state.prev_batch_id,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                state.token_count += msg_tokens
                state.next_msg_pos += 1

                if state.token_count > batch_limit:
                    state.prev_batch_id = state.batch_id
                    state.batch_id = self._generate_batch_id()
                    state.next_msg_pos = 0
                    state.token_count = 0

    async def _archive_message(
        self,
        msg: Message,
        batch_id: str,
        msg_pos: int,
        prev_batch_id: str | None,
        user_id: int,
        thread_id: int = 0,
    ) -> None:
        """Archive a single message to ChromaDB with batch metadata."""
        now = datetime.now()
        metadata: Metadata = {
            "message_id": str(msg.id),
            "batch_id": batch_id,
            "msg_pos": msg_pos,
            "prev_batch_id": prev_batch_id or "",
            "thread_id": str(thread_id),
            "role": msg.role,
            "timestamp": now.isoformat(),
            "timestamp_unix": now.timestamp(),
        }

        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            await collection.add(
                metadatas=[metadata],
                ids=[str(msg.id)],
                documents=[msg.content],
            )
            logger.debug(f"Archived message {msg.id} in batch {batch_id} at pos {msg_pos}")
        except Exception as e:
            logger.error(f"Failed to archive message {msg.id}: {e}")
            raise ChromaArchiveError(f"Failed to archive message {msg.id}: {e}") from e

    async def search(self, user_id: int, query: str, n_results: int, thread_id: int = 0) -> list[MemorySearchResult]:
        """Search archived messages with context retrieval."""
        try:
            hit = await self._semantic_search(user_id, query, n_results, thread_id)
            if not hit:
                return []

            hit_batch_id = hit.get("batch_id")
            hit_msg_pos = hit.get("msg_pos")
            hit_prev_batch_id = hit.get("prev_batch_id")

            if not hit_batch_id:
                return [
                    {
                        "content": hit["content"],
                        "role": hit["role"],
                        "timestamp": hit["timestamp"],
                        "message_id": hit["message_id"],
                        "batch_id": None,
                        "msg_pos": None,
                        "prev_batch_id": None,
                        "thread_id": thread_id,
                    }
                ]

            context_messages = await self._get_batch_by_id(user_id, hit_batch_id, thread_id)
            current_batch_count = len(context_messages)

            if (
                current_batch_count > 0
                and hit_msg_pos is not None
                and hit_msg_pos <= EDGE_THRESHOLD
                and hit_prev_batch_id
            ):
                prev_batch = await self._get_batch_by_prev_id(user_id, hit_prev_batch_id, thread_id)
                if prev_batch:
                    context_messages = prev_batch + context_messages
                    current_batch_count = len(context_messages)

            if (
                hit_msg_pos is not None
                and current_batch_count > 0
                and hit_msg_pos >= current_batch_count - EDGE_THRESHOLD - 1
            ):
                next_batch = await self._get_next_batch(user_id, hit_batch_id, thread_id)
                if next_batch:
                    context_messages.extend(next_batch)

            context_messages.sort(key=lambda x: (x.get("batch_id", ""), x.get("msg_pos", 0)))
            return context_messages

        except ChromaSearchError:
            return []
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int, thread_id: int = 0
    ) -> MemorySearchResult | None:
        """Perform semantic search."""
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            result = await collection.query(
                query_texts=[query],
                n_results=1,
            )

            documents = result.get("documents")
            metadatas = result.get("metadatas")

            if documents and metadatas and len(documents) > 0 and len(documents[0]) > 0:
                metadata = metadatas[0][0]
                return MemorySearchResult(
                    content=documents[0][0],
                    role=str(metadata.get("role", "")),
                    timestamp=str(metadata.get("timestamp", "")),
                    message_id=str(metadata.get("message_id", "")),
                    batch_id=str(metadata.get("batch_id", "")),
                    msg_pos=int(str(metadata.get("msg_pos", -1))),
                    prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                    thread_id=int(str(metadata.get("thread_id", 0))),
                )
            return None

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise ChromaSearchError(f"Semantic search failed: {e}") from e

    async def _get_batch_by_id(self, user_id: int, batch_id: str, thread_id: int = 0) -> list[MemorySearchResult]:
        """Get all messages in a batch by batch_id."""
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            result = await collection.get(where={"batch_id": batch_id})
            return self._format_batch_results(result)
        except Exception as e:
            logger.error(f"Failed to get batch {batch_id}: {e}")
            return []

    async def _get_batch_by_prev_id(
        self, user_id: int, prev_batch_id: str, thread_id: int = 0
    ) -> list[MemorySearchResult]:
        """Get a batch by its ID."""
        return await self._get_batch_by_id(user_id, prev_batch_id, thread_id)

    async def _get_next_batch(
        self, user_id: int, current_batch_id: str, thread_id: int = 0
    ) -> list[MemorySearchResult]:
        """Get the batch that follows the current one."""
        try:
            collection = await self._get_or_create_collection(user_id, thread_id)
            result = await collection.get(where={"prev_batch_id": current_batch_id})
            return self._format_batch_results(result)
        except Exception as e:
            logger.error(f"Failed to get next batch: {e}")
            return []

    def _format_batch_results(self, result: GetResult) -> list[MemorySearchResult]:
        """Format raw ChromaDB collection.get() result into search results."""
        formatted: list[MemorySearchResult] = []
        documents = result.get("documents")
        metadatas = result.get("metadatas")

        if documents and metadatas:
            for doc, metadata in zip(documents, metadatas):
                formatted.append(
                    MemorySearchResult(
                        content=doc,
                        role=str(metadata.get("role", "")),
                        timestamp=str(metadata.get("timestamp", "")),
                        message_id=str(metadata.get("message_id", "")),
                        batch_id=str(metadata.get("batch_id", "")),
                        msg_pos=int(str(metadata.get("msg_pos", -1))),
                        prev_batch_id=str(metadata.get("prev_batch_id", "")) or None,
                        thread_id=int(str(metadata.get("thread_id", 0))),
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
                    continue
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
    """Create memory instance based on ChromaDB configuration."""
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
        conversation_memory: LongConversationMemory
        if application_settings.chroma_host:
            conversation_memory = ExternalChromaLongConversationMemory(embedding_function=embedding_function)
        else:
            conversation_memory = InternalChromaLongConversationMemory(embedding_function=embedding_function)

        logger.info("Semantic memory initialized successfully")
        return conversation_memory

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

    return None


memory: LongConversationMemory | None = create_memory()
