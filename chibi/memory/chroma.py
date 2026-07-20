"""ChromaDB memory implementation with batch metadata and context retrieval."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Callable, cast

import chromadb
from chromadb import Collection, EmbeddingFunction, Metadata, Where
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import Documents, Embeddings
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    GoogleGeminiEmbeddingFunction,
    JinaEmbeddingFunction,
    MistralEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from loguru import logger

from chibi.config import application_settings
from chibi.exceptions import (
    ChromaArchiveError,
    ChromaCollectionError,
    ChromaConnectionError,
    ChromaSearchError,
)
from chibi.memory.abstract import (
    EDGE_THRESHOLD,
    ArchiveState,
    LongConversationMemory,
    MemorySearchResult,
)
from chibi.models import Message, User
from chibi.services.lock_manager import LockManager
from chibi.services.task_manager import task_manager
from chibi.storage.abstract import Database


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
            path=application_settings.local_data_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Per-(user, thread) archive state: tracks current batch metadata and token counts
        self._archive_state: dict[tuple[int, int], ArchiveState] = {}
        logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.local_data_path})")

    async def _get_last_batch_id(self, user_id: int, thread_id: int = 0) -> str | None:
        """Get batch_id of the most recent message for this user+thread in ChromaDB.

        Filters to the last 7 days using the numeric, include=["metadatas"] to avoid pulling documents.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.

        Returns:
            The most recent batch_id, or None if no recent messages or on error.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        one_week_ago = (datetime.now() - timedelta(days=7)).timestamp()
        where_filter = cast(Where, {"timestamp_unix": {"$gte": one_week_ago}})

        try:
            result = await asyncio.to_thread(
                collection.get,
                where=where_filter,
                include=["metadatas"],
            )
        except ChromaError:
            logger.exception(f"Failed to get last batch_id for user {user_id}, thread {thread_id}")
            return None

        metadatas = result.get("metadatas")
        if not metadatas:
            return None
        latest = max(metadatas, key=lambda m: float(str(m.get("timestamp_unix", 0))))
        bid = latest.get("batch_id")
        return str(bid) if bid else None

    async def _get_or_create_collection(self, user_id: int, thread_id: int = 0) -> Collection:
        """Get or create a collection for user+thread.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.

        Returns:
            ChromaDB collection instance.

        Raises:
            ChromaCollectionError: If collection access fails.
        """
        collection_name = self._get_collection_name(user_id=user_id, thread_id=thread_id)
        try:
            return await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except ChromaError as e:
            raise ChromaCollectionError(f"Failed to get or create collection '{collection_name}': {e}") from e

    async def _get_or_create_archive_state(self, user_id: int, thread_id: int = 0) -> ArchiveState:
        """Get or create per-(user, thread) archive state.

        On first call for a user+thread, queries ChromaDB for the last batch_id
        to enable proper batch chaining across restarts.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.

        Returns:
            ArchiveState.

        Raises:
            ChromaCollectionError: If collection access fails.
        """
        key = (user_id, thread_id)
        if key not in self._archive_state:
            last_batch_id = await self._get_last_batch_id(user_id=user_id, thread_id=thread_id)
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
        state = await self._get_or_create_archive_state(user_id=user_id, thread_id=thread_id)
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

        Args:
            user_id: The user ID.
            messages: List of messages to archive.
            thread_id: Thread ID.

        Raises:
            ChromaArchiveError: If any message fails to archive.
        """
        if not messages:
            return None

        lock = await LockManager().get_lock(key=f"{user_id}:{thread_id}")
        async with lock:
            state = await self._get_or_create_archive_state(user_id=user_id, thread_id=thread_id)

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
                await self.update_archive_state(user_id=user_id, thread_id=thread_id, tokens_to_add=msg_tokens)
        return None

    async def _archive_message(
        self,
        msg: Message,
        batch_id: str,
        msg_pos: int,
        prev_batch_id: str | None,
        user_id: int,
        thread_id: int = 0,
    ) -> None:
        """Archive a single message to ChromaDB with batch metadata.

        Args:
            msg: Message to archive.
            batch_id: Current batch ID.
            msg_pos: Position of message within the batch.
            prev_batch_id: Previous batch ID.
            user_id: The user ID.
            thread_id: Thread ID.

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

        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            await asyncio.to_thread(
                collection.add,
                metadatas=[metadata],
                ids=[str(msg.id)],
                documents=[msg.content],
            )
            return None
        except ChromaError as e:
            logger.exception(f"Failed to archive message {msg.id}")
            raise ChromaArchiveError(f"Failed to archive message {msg.id}: {e}") from e

    async def search(self, user_id: int, query: str, n_results: int, thread_id: int = 0) -> list[MemorySearchResult]:
        """Search archived messages by semantic similarity.

        Performs semantic search first, then retrieves surrounding context
        from neighboring batches if the hit is near batch edges.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Max results for semantic search.
            thread_id: Thread ID.

        Returns:
            List of search results with context; empty list on error or no matches.
        """
        # Step 1: Semantic search (ChromaSearchError is treated as "no results", not a crash)
        hit = await self._semantic_search(user_id=user_id, query=query, n_results=n_results, thread_id=thread_id)
        if not hit:
            return []

        hit_batch_id = hit.batch_id
        hit_msg_pos = hit.msg_pos
        hit_prev_batch_id = hit.prev_batch_id

        if not hit_batch_id:
            return [
                MemorySearchResult(
                    content=hit.content,
                    role=hit.role,
                    timestamp=hit.timestamp,
                    message_id=hit.message_id,
                    batch_id=None,
                    msg_pos=None,
                    prev_batch_id=None,
                    thread_id=thread_id,
                )
            ]

        # Step 2: Get current batch
        context_messages = await self._get_batch_by_field(user_id=user_id, batch_id=hit_batch_id, thread_id=thread_id)
        current_batch_count = len(context_messages)

        # Near beginning: add previous batch (only if valid prev_batch_id exists)
        if current_batch_count > 0 and hit_msg_pos is not None and hit_msg_pos <= EDGE_THRESHOLD and hit_prev_batch_id:
            prev_batch = await self._get_batch_by_field(
                user_id=user_id, batch_id=hit_prev_batch_id, thread_id=thread_id
            )
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

        # Sort by (batch_id, msg_pos) - UUID7 ensures chronological order
        context_messages.sort(key=lambda x: (x.batch_id or "", x.msg_pos or 0))
        return context_messages

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int = 1, thread_id: int = 0
    ) -> MemorySearchResult | None:
        """Perform semantic search.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Max results.
            thread_id: Thread ID.

        Returns:
            Best matching MemorySearchResult or None if no hits.

        Raises:
            ChromaSearchError: If the search query fails.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            result = await asyncio.to_thread(
                collection.query,
                query_texts=[query],
                n_results=n_results,
            )
        except ChromaError as e:
            logger.exception(f"Semantic search failed for user {user_id}, thread {thread_id}")
            raise ChromaSearchError(f"Semantic search failed: {e}") from e

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

    async def _get_batch_by_field(
        self,
        user_id: int,
        batch_id: str,
        field: str = "batch_id",
        thread_id: int = 0,
    ) -> list[MemorySearchResult]:
        """Get all messages matching a batch-related metadata field.

        Args:
            user_id: The user ID.
            batch_id: Value to match against the metadata field.
            field: Metadata field name to filter by.
            thread_id: Thread ID.

        Returns:
            List of formatted search results; empty list on error or no matches.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            result = await asyncio.to_thread(
                collection.get,
                where={field: batch_id},
            )
        except ChromaError:
            logger.exception(f"Failed to get batch {batch_id} (field={field}) for user {user_id}")
            return []

        return self._format_batch_results(result)

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        Args:
            retention_days: Number of days to retain messages.

        Raises:
            ChromaDeleteError: If cleanup fails.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)

        # Outer guard: a failure listing collections must not crash the whole job.
        try:
            collections = await asyncio.to_thread(self._client.list_collections)
        except ChromaError:
            logger.exception("Failed to list ChromaDB collections during retention cleanup")
            return

        for collection in [c for c in collections if c.name.startswith("user_")]:
            # Per-collection failure is isolated: one bad collection must not skip others.
            try:
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
            except ChromaError:
                logger.exception(f"Failed to clean up collection {collection.name}")
                continue


class ExternalChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation using AsyncHttpClient (asynchronous).

    Same features as InternalChromaLongConversationMemory but uses
    async HTTP client for external ChromaDB server.

    Features:
        - Per-message persistence with batch metadata (batch_id, msg_pos, prev_batch_id)
        - Context retrieval (neighboring batches around semantic search hits)
        - No full scan required
        - Restart-safe: prev_batch_id is loaded from DB on first archive call
        - Per-thread batch: each thread_id has separate batch tracking
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

    async def _get_last_batch_id(self, user_id: int, thread_id: int = 0) -> str | None:
        """Get batch_id of the most recent message for this user+thread in ChromaDB.

        Filters to the last 7 days using the numeric timestamp_unix field.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.

        Returns:
            The most recent batch_id, or None if no recent messages or on error.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        one_week_ago = (datetime.now() - timedelta(days=7)).timestamp()
        where_filter = cast(Where, {"timestamp_unix": {"$gte": one_week_ago}})

        try:
            result = await collection.get(
                where=where_filter,
                include=["metadatas"],
            )
        except ChromaError:
            logger.exception(f"Failed to get last batch_id for user {user_id}, thread {thread_id}")
            return None

        metadatas = result.get("metadatas")
        if not metadatas:
            return None
        latest = max(
            metadatas,
            key=lambda m: float(str(m.get("timestamp_unix", 0))),
        )
        bid = latest.get("batch_id")
        return str(bid) if bid else None

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        """Get or create async client (lazy initialization)."""
        if self._client is None:
            try:
                self._client = await chromadb.AsyncHttpClient(
                    host=application_settings.chroma_host,
                    port=application_settings.chroma_port,
                )
            except ChromaError as e:
                raise ChromaConnectionError(
                    f"Failed to connect to ChromaDB at "
                    f"{application_settings.chroma_host}:{application_settings.chroma_port}: {e}"
                ) from e
        return self._client

    async def _get_or_create_collection(self, user_id: int, thread_id: int = 0) -> AsyncCollection:
        """Get or create collection for user+thread.

        Args:
            user_id: The user ID.
            thread_id: The thread ID.

        Returns:
            ChromaDB async collection instance.

        Raises:
            ChromaCollectionError: If collection access fails.
        """
        collection_name = self._get_collection_name(user_id=user_id, thread_id=thread_id)
        client = await self._get_client()
        try:
            return await client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except ChromaError as e:
            raise ChromaCollectionError(f"Failed to get or create collection '{collection_name}': {e}") from e

    async def archive(self, user_id: int, messages: list[Message], thread_id: int = 0) -> None:
        """Archive messages to ChromaDB with batch metadata."""
        if not messages:
            return None

        lock = await LockManager().get_lock(key=f"{user_id}:{thread_id}")
        async with lock:
            key = (user_id, thread_id)
            if key not in self._archive_state:
                last_batch_id = await self._get_last_batch_id(user_id=user_id, thread_id=thread_id)
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
        return None

    async def _archive_message(
        self,
        msg: Message,
        batch_id: str,
        msg_pos: int,
        prev_batch_id: str | None,
        user_id: int,
        thread_id: int = 0,
    ) -> None:
        """Archive a single message to ChromaDB with batch metadata.

        Args:
            msg: Message to archive.
            batch_id: Current batch ID.
            msg_pos: Position of message within the batch.
            prev_batch_id: Previous batch ID (for context retrieval).
            user_id: The user ID.
            thread_id: Thread ID.

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

        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            await collection.add(
                metadatas=[metadata],
                ids=[str(msg.id)],
                documents=[msg.content],
            )
            logger.debug(f"Archived message {msg.id} in batch {batch_id} at pos {msg_pos}")
        except ChromaError as e:
            logger.exception(f"Failed to archive message {msg.id}")
            raise ChromaArchiveError(f"Failed to archive message {msg.id}: {e}") from e

    async def search(self, user_id: int, query: str, n_results: int, thread_id: int = 0) -> list[MemorySearchResult]:
        """Search archived messages with context retrieval."""
        try:
            hit = await self._semantic_search(user_id=user_id, query=query, n_results=n_results, thread_id=thread_id)
        except ChromaSearchError:
            return []

        if not hit:
            return []

        hit_batch_id = hit.batch_id
        hit_msg_pos = hit.msg_pos
        hit_prev_batch_id = hit.prev_batch_id

        if not hit_batch_id:
            return [
                MemorySearchResult(
                    content=hit.content,
                    role=hit.role,
                    timestamp=hit.timestamp,
                    message_id=hit.message_id,
                    batch_id=None,
                    msg_pos=None,
                    prev_batch_id=None,
                    thread_id=thread_id,
                )
            ]

        context_messages = await self._get_batch_by_id(user_id=user_id, batch_id=hit_batch_id, thread_id=thread_id)
        current_batch_count = len(context_messages)

        if current_batch_count > 0 and hit_msg_pos is not None and hit_msg_pos <= EDGE_THRESHOLD and hit_prev_batch_id:
            prev_batch = await self._get_batch_by_prev_id(
                user_id=user_id, prev_batch_id=hit_prev_batch_id, thread_id=thread_id
            )
            if prev_batch:
                context_messages = prev_batch + context_messages
                current_batch_count = len(context_messages)

        if (
            hit_msg_pos is not None
            and current_batch_count > 0
            and hit_msg_pos >= current_batch_count - EDGE_THRESHOLD - 1
        ):
            next_batch = await self._get_next_batch(user_id=user_id, current_batch_id=hit_batch_id, thread_id=thread_id)
            if next_batch:
                context_messages.extend(next_batch)

        context_messages.sort(key=lambda x: (x.batch_id or "", x.msg_pos or 0))
        return context_messages

    async def _semantic_search(
        self, user_id: int, query: str, n_results: int = 1, thread_id: int = 0
    ) -> MemorySearchResult | None:
        """Perform semantic search against archived messages.

        Args:
            user_id: The user ID.
            query: Search query string.
            n_results: Max results (only top hit used).
            thread_id: Thread ID.

        Returns:
            Best matching MemorySearchResult or None if no hits.

        Raises:
            ChromaSearchError: If the search query fails.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            result = await collection.query(
                query_texts=[query],
                n_results=1,
            )
        except ChromaError as e:
            logger.exception(f"Semantic search failed for user {user_id}, thread {thread_id}")
            raise ChromaSearchError(f"Semantic search failed: {e}") from e

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

    async def _get_batch_by_id(self, user_id: int, batch_id: str, thread_id: int = 0) -> list[MemorySearchResult]:
        """Get all messages in a batch by batch_id.

        Args:
            user_id: The user ID.
            batch_id: Batch ID to retrieve.
            thread_id: Thread ID.

        Returns:
            List of MemorySearchResult objects; empty list on error or no matches.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            result = await collection.get(where={"batch_id": batch_id})
        except ChromaError:
            logger.exception(f"Failed to get batch {batch_id} for user {user_id}")
            return []
        return self._format_batch_results(result)

    async def _get_batch_by_prev_id(
        self, user_id: int, prev_batch_id: str, thread_id: int = 0
    ) -> list[MemorySearchResult]:
        """Get a batch by its prev_batch_id.

        Args:
            user_id: The user ID.
            prev_batch_id: Previous batch ID to search for.
            thread_id: Thread ID.

        Returns:
            List of MemorySearchResult objects; empty list if not found.
        """
        return await self._get_batch_by_id(user_id=user_id, batch_id=prev_batch_id, thread_id=thread_id)

    async def _get_next_batch(
        self, user_id: int, current_batch_id: str, thread_id: int = 0
    ) -> list[MemorySearchResult]:
        """Get the batch that follows the current one.

        Args:
            user_id: The user ID.
            current_batch_id: Current batch ID to find successor for.
            thread_id: Thread ID.

        Returns:
            List of MemorySearchResult objects; empty list if not found.
        """
        collection = await self._get_or_create_collection(user_id=user_id, thread_id=thread_id)
        try:
            result = await collection.get(where={"prev_batch_id": current_batch_id})
        except ChromaError:
            logger.exception(f"Failed to get next batch after {current_batch_id} for user {user_id}")
            return []
        return self._format_batch_results(result)

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days.

        Args:
            retention_days: Number of days to retain messages.

        Raises:
            ChromaDeleteError: If cleanup fails.
        """
        cutoff = datetime.now() - timedelta(days=retention_days)

        # Outer guard: a failure listing collections must not crash the whole job.
        try:
            client = await self._get_client()
            collections = await client.list_collections()
        except ChromaError:
            logger.exception("Failed to list ChromaDB collections during retention cleanup")
            return

        for collection in [c for c in collections if c.name.startswith("user_")]:
            # Per-collection failure is isolated: one bad collection must not skip others.
            try:
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
            except ChromaError:
                logger.exception(f"Failed to clean up collection {collection.name}")
                continue


class FastEmbedEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible EmbeddingFunction backed by qdrant/fastembed.

    fastembed ships pure-Python wheels (``py3-none-any``) and bundles its own
    onnxruntime, so it works across platforms without extra system deps.
    We use it as a drop-in replacement for chromadb's ``DefaultEmbeddingFunction``
    on platforms where the latter is not available.

    ``fastembed`` is an optional dependency. We import it lazily so the package
    can be installed on platforms that don't need it (e.g. macOS x86_64 with its
    own onnxruntime pin). When the import fails, callers should treat ChromaDB
    as unsupported on that machine.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # fastembed is an optional dep

        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # fastembed.embed() returns a generator of numpy arrays; ChromaDB expects lists
        return [vec.tolist() for vec in self._model.embed(input)]


def create_memory() -> LongConversationMemory | None:
    """Create memory instance based on ChromaDB configuration."""
    if not application_settings.is_chroma_configured:
        logger.info("ChromaDB not configured, semantic memory disabled")
        return None

    embedding_function: EmbeddingFunction
    model = application_settings.embedding_model
    match application_settings.embedding_function:
        case "GEMINI":
            embedding_function = (
                GoogleGeminiEmbeddingFunction(model_name=model) if model else GoogleGeminiEmbeddingFunction()
            )
        case "OPENAI":
            embedding_function = OpenAIEmbeddingFunction(
                api_key_env_var="OPENAI_API_KEY", model_name=model or "text-embedding-3-small"
            )
        case "MISTRALAI":
            embedding_function = MistralEmbeddingFunction(
                api_key_env_var="MISTRALAI_API_KEY",
                model=model or "mistral-embed",
            )
        case "JINA":
            embedding_function = JinaEmbeddingFunction(
                api_key_env_var="JINA_API_KEY", model_name=model or "jina-embeddings-v5-omni-small"
            )
        case "LOCAL":
            try:
                embedding_function = FastEmbedEmbeddingFunction(
                    model_name=model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            except Exception:
                logger.warning("Fastebmed is unavailable. Loading default ChromaDB embedding function...")
                embedding_function = DefaultEmbeddingFunction()
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

    except ChromaError as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

    return None


def with_chroma_archival(long_conv_memory: LongConversationMemory | None) -> Callable:
    """Decorator: wrap a Database instance so that add_message also archives to ChromaDB.

    Archival is dispatched fire-and-forget via task_manager, so a ChromaDB
    failure never breaks the primary write path. The original add_message is
    bound once at decoration time (not looked up on self.inner per call) to
    avoid recursion and keep the closure cheap.

    Args:
        long_conv_memory: The ChromaDB memory backend, or None to return the storage unchanged.

    Returns:
        A function that takes a Database and returns the (possibly wrapped) Database.
    """
    if long_conv_memory is None:
        return lambda storage: storage

    def wrap(storage: Database) -> Database:
        original_add_message = storage.add_message

        async def add_message(user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
            await original_add_message(user=user, message=message, ttl=ttl, thread_id=thread_id)
            if message.role not in ("user", "assistant"):
                return None
            if any((message.tool_name, message.tool_calls, message.tool_call_id)):
                return None

            if message.role == "user":
                try:
                    snippet = json.loads(message.content).get("prompt", message.content)[:50]
                except (json.JSONDecodeError, TypeError):
                    snippet = str(message.content)[:50]
            else:
                snippet = message.content[:50]
            logger.debug(f"Scheduling archival of message {message.id} for user {user.id}. Message: {snippet}")
            task_manager.run_task(
                long_conv_memory.archive(user_id=user.id, messages=[message], thread_id=thread_id), user_id=user.id
            )
            return None

        setattr(storage, "add_message", add_message)
        return storage

    return wrap


memory: LongConversationMemory | None = create_memory()
