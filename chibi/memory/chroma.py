import asyncio
from datetime import datetime, timedelta

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from chibi.config import application_settings
from chibi.memory.abstract import LongConversationMemory
from chibi.models import Message


def create_memory() -> LongConversationMemory | None:
    """Create memory instance if ChromaDB is configured."""

    if not application_settings.is_chroma_configured:
        logger.info("ChromaDB not configured, semantic memory disabled")
        return None

    try:
        # Import here to avoid circular import at module level
        from chibi.memory.chroma import ChromaLongConversationMemory

        mem = ChromaLongConversationMemory()
        logger.info("Semantic memory initialized successfully")
        return mem
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


class ChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation of long-term conversation memory."""

    def __init__(self) -> None:
        """Initialize ChromaDB client based on configuration."""
        self._client: chromadb.AsyncHttpClient | None = None
        self._is_embedded: bool = False
        self._init_client()

    @property
    def client(self) -> chromadb.AsyncHttpClient:
        if self._client is None:
            self._init_client()
            return self._client
        return self._client

    def _init_client(self) -> None:
        """Initialize ChromaDB client (embedded or external)."""
        if application_settings.chroma_host:
            # External mode
            self._client = chromadb.AsyncHttpClient(
                host=application_settings.chroma_host, port=application_settings.chroma_port
            )
            self._is_embedded = False
            logger.info(
                f"ChromaDB: using external mode ({application_settings.chroma_host}:{application_settings.chroma_port})"
            )
        else:
            # Embedded mode
            self._client = chromadb.PersistentClient(
                path=application_settings.chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._is_embedded = True
            logger.info(f"ChromaDB: using embedded mode (persist: {application_settings.chroma_persist_dir})")

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user."""
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> chromadb.AsyncClientAPI:
        """Get or create collection for user."""
        collection_name = self._get_collection_name(user_id)
        if self._is_embedded:
            return await asyncio.to_thread(self.client.get_or_create_collection, name=collection_name)
        return await self.client.get_or_create_collection(name=collection_name)

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB."""
        if not messages:
            return

        try:
            collection = await self._get_or_create_collection(user_id)

            args_to_save = {
                "metadatas": [
                    {"role": msg.role, "timestamp": datetime.now().isoformat(), "message_id": str(msg.id)}
                    for msg in messages
                ],
                "ids": [str(msg.id) for msg in messages],
                "documents": [msg.content for msg in messages],
            }

            if self._is_embedded:
                await asyncio.to_thread(collection.add, **args_to_save)
            else:
                await collection.add(**args_to_save)
            logger.debug(f"Archived {len(messages)} messages for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to archive messages for user {user_id}: {e}")
            raise

    async def search(self, user_id: int, query: str, n_results: int) -> list[dict]:
        """Search archived messages by semantic similarity."""
        try:
            collection = await self._get_or_create_collection(user_id)
            query_args = {
                "query_texts": [query],
                "n_results": n_results,
            }
            if self._is_embedded:
                result = await asyncio.to_thread(collection.query, **query_args)
            else:
                result = await collection.query(**query_args)

            # Format results
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
            # Get all collections
            if self._is_embedded:
                collections = await asyncio.to_thread(self.client.list_collections)
            else:
                collections = await self.client.list_collections()

            for coll in [c for c in collections if c.name.startswith("user_")]:
                try:
                    # Get all IDs and metadata
                    if self._is_embedded:
                        result = await asyncio.to_thread(coll.get)
                    else:
                        result = await coll.get()

                    if not result or not result.get("ids"):
                        continue

                    # Find IDs to delete
                    ids_to_delete = []
                    metadatas = result.get("metadatas", [])
                    for i, metadata in enumerate(metadatas):
                        timestamp_str = metadata.get("timestamp")
                        try:
                            if (timestamp_str and datetime.fromisoformat(timestamp_str) < cutoff) or not timestamp_str:
                                ids_to_delete.append(result["ids"][i])
                        except (ValueError, TypeError):
                            # remove invalid data as well
                            ids_to_delete.append(result["ids"][i])

                    # Delete old documents
                    if ids_to_delete:
                        if self._is_embedded:
                            await asyncio.to_thread(coll.delete, ids=ids_to_delete)
                        else:
                            await coll.delete(ids=ids_to_delete)
                        logger.info(f"Deleted {len(ids_to_delete)} old messages from {coll.name}")

                except Exception as e:
                    logger.warning(f"Failed to process collection {coll.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")
            raise


memory: LongConversationMemory | None = create_memory()
