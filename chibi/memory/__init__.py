from loguru import logger

from chibi.memory.abstract import LongConversationMemory
from chibi.config import application_settings


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
        from chibi.services.providers.tools.memory import SearchInConversationHistoryTool
        from chibi.services.providers.tools import RegisteredChibiTools
        RegisteredChibiTools.register(SearchInConversationHistoryTool)
        logger.info("SearchInConversationHistoryTool registered")


memory: LongConversationMemory | None = create_memory()

# __all__ = ["memory", "create_memory", "register_memory_tool", "LongConversationMemory"]
