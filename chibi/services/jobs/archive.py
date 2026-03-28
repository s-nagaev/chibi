from loguru import logger

from chibi.config import application_settings
from chibi.memory import memory


async def perform_retention_cleanup() -> None:
    """Delete old archived messages from ChromaDB based on retention settings."""
    if memory is None:
        logger.info("Memory not configured, skipping retention cleanup")
        return

    logger.info(f"Starting retention cleanup for {application_settings.chroma_history_retention_days} days")

    try:
        await memory.delete_old(application_settings.chroma_history_retention_days)
        logger.info("Retention cleanup completed")
    except Exception as e:
        logger.error(f"Retention cleanup failed: {e}")
