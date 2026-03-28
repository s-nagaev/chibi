import asyncio
from datetime import datetime, timedelta
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from chibi.config import application_settings
from chibi.memory.abstract import LongConversationMemory
from chibi.models import Message


class ChromaLongConversationMemory(LongConversationMemory):
    """ChromaDB implementation of long-term conversation memory."""

    def __init__(self) -> None:
        """Initialize ChromaDB client based on configuration."""
        self._client: Any = None
        self._is_embedded: bool = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client (embedded or external)."""
        if application_settings.chroma_host:
            # External mode
            logger.info(f"ChromaDB: using external mode ({application_settings.chroma_host}:{application_settings.chroma_port})")
            self._client = chromadb.AsyncHttpClient(
                host=application_settings.chroma_host,
                port=application_settings.chroma_port
            )
            self._is_embedded = False
        else:
            # Embedded mode
            persist_path = application_settings.chroma_persist_dir
            logger.info(f"ChromaDB: using embedded mode (persist: {persist_path})")
            self._client = chromadb.PersistentClient(
                path=persist_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._is_embedded = True

    def _get_collection_name(self, user_id: int) -> str:
        """Get collection name for user."""
        return f"user_{user_id}"

    async def _get_or_create_collection(self, user_id: int) -> Any:
        """Get or create collection for user."""
        collection_name = self._get_collection_name(user_id)
        if self._is_embedded:
            return await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=collection_name
            )
        return await self._client.get_or_create_collection(name=collection_name)

    async def archive(self, user_id: int, messages: list[Message]) -> None:
        """Archive messages to ChromaDB."""
        if not messages:
            return

        try:
            collection = await self._get_or_create_collection(user_id)

            documents = [msg.content for msg in messages]
            metadatas = [
                {
                    "role": msg.role,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_id": str(msg.id)
                }
                for msg in messages
            ]
            ids = [str(msg.id) for msg in messages]

            if self._is_embedded:
                await asyncio.to_thread(
                    collection.add,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                await collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            logger.debug(f"Archived {len(messages)} messages for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to archive messages for user {user_id}: {e}")
            raise

    async def search(self, user_id: int, query: str, n_results: int) -> list[dict]:
        """Search archived messages by semantic similarity."""
        try:
            collection = await self._get_or_create_collection(user_id)

            if self._is_embedded:
                result = await asyncio.to_thread(
                    collection.query,
                    query_texts=[query],
                    n_results=n_results
                )
            else:
                result = await collection.query(
                    query_texts=[query],
                    n_results=n_results
                )

            # Format results
            formatted_results = []
            if result.get("documents") and result["documents"][0]:
                for i, doc in enumerate(result["documents"][0]):
                    formatted_results.append({
                        "content": doc,
                        "role": result["metadatas"][0][i].get("role"),
                        "timestamp": result["metadatas"][0][i].get("timestamp"),
                        "message_id": result["metadatas"][0][i].get("message_id"),
                        "distance": result["distances"][0][i] if result.get("distances") else None
                    })

            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search messages for user {user_id}: {e}")
            return []

    async def delete_old(self, retention_days: int) -> None:
        """Delete archived messages older than retention_days."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        try:
            # Get all collections
            if self._is_embedded:
                collections = await asyncio.to_thread(self._client.list_collections)
            else:
                collections = await self._client.list_collections()

            # Filter user collections
            user_collections = [c for c in collections if c.name.startswith("user_")]

            for coll in user_collections:
                try:
                    # Get all IDs and metadata
                    if self._is_embedded:
                        result = await asyncio.to_thread(
                            coll.get
                        )
                    else:
                        result = await coll.get()

                    if not result or not result.get("ids"):
                        continue

                    # Find IDs to delete
                    ids_to_delete = []
                    metadatas = result.get("metadatas", [])
                    for i, metadata in enumerate(metadatas):
                        timestamp_str = metadata.get("timestamp")
                        if timestamp_str:
                            try:
                                msg_time = datetime.fromisoformat(timestamp_str)
                                if msg_time < cutoff:
                                    ids_to_delete.append(result["ids"][i])
                            except (ValueError, TypeError):
                                pass

                    # Delete old documents
                    if ids_to_delete:
                        if self._is_embedded:
                            await asyncio.to_thread(
                                coll.delete,
                                ids=ids_to_delete
                            )
                        else:
                            await coll.delete(ids=ids_to_delete)
                        logger.info(f"Deleted {len(ids_to_delete)} old messages from {coll.name}")

                except Exception as e:
                    logger.warning(f"Failed to process collection {coll.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to delete old messages: {e}")
            raise