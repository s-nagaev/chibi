import asyncio
from typing import Any, Coroutine

from loguru import logger

from chibi.utils.app import SingletonMeta


class BackgroundTaskManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        """Initialize the task manager."""
        if not hasattr(self, "_tasks"):
            self._tasks: set[asyncio.Task] = set()
            self._shutting_down: bool = False

    async def _wrap_with_timeout(self, coro: Coroutine[Any, Any, Any], timeout: float) -> Any:
        """Wrap a coroutine with a timeout.

        Args:
            coro: The coroutine to wrap.

        Returns:
            The wrapped coroutine result.
        """
        try:
            return await asyncio.wait_for(fut=coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Background task timed out after {timeout}s")
            raise

    def run_task(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> asyncio.Task | None:
        """Schedule a coroutine to run in the background.

        Args:
            coro: the coroutine to run
            timeout: optional timeout in seconds. If the task doesn't complete
                     within this time, it will be cancelled with TimeoutError
        """
        if self._shutting_down:
            logger.warning("Task manager is shutting down, refusing new task")
            coro.close()
            return None

        # Wrap with timeout if specified
        if timeout is not None:
            coro = self._wrap_with_timeout(coro, timeout)

        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._discard_task)
        return task

    def _discard_task(self, task: asyncio.Task) -> None:
        """Callback to remove a task from the set when it finishes."""
        try:
            exc = task.exception()
            if exc:
                logger.error(
                    f"Background task '{task.get_name()}' failed: {exc.__class__.__name__} ({str(exc) or 'no details'})"
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error checking background task result: {e}")
        finally:
            self._tasks.discard(task)

    async def shutdown(self, *args: Any) -> None:
        """
        Wait for all background tasks to complete with a timeout.
        If tasks do not finish within 15 seconds, they are cancelled.
        """
        logger.info("Shutting down background tasks...")
        self._shutting_down = True
        tasks_to_wait = list(self._tasks)
        if not tasks_to_wait:
            logger.info("No background tasks to wait, we're good.")
            return None

        logger.info(f"Waiting for {len(tasks_to_wait)} background tasks to complete...")
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_wait, return_exceptions=True), timeout=5.0)
            logger.info("All background tasks completed.")

        except asyncio.TimeoutError:
            logger.warning("Timeout reached. Cancelling remaining background tasks...")
            remaining = [t for t in tasks_to_wait if not t.done()]
            for task in remaining:
                task.cancel()
            logger.info(f"Cancelled {len(remaining)} remaining tasks.")


task_manager = BackgroundTaskManager()
