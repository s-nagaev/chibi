import asyncio
from typing import Any, Coroutine

from loguru import logger

from chibi.utils.app import SingletonMeta


class BackgroundTaskManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        """Initialize the task manager."""
        if not hasattr(self, "_tasks"):
            self._tasks: dict[str, set[asyncio.Task]] = {}
            self._task_to_user_id: dict[asyncio.Task, str] = {}
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

    def _get_task_id(self, user_id: int, thread_id: int) -> str:
        """Get the task id for a user and thread.

        Args:
            user_id: The user id.
            thread_id: The thread id.

        Returns:
            Task id.
        """
        return f"{user_id}-{thread_id}"

    def run_task(
        self, coro: Coroutine[Any, Any, Any], user_id: int, thread_id: int = 0, timeout: float | None = None
    ) -> asyncio.Task | None:
        """Schedule a coroutine to run in the background.

        Args:
            coro: the coroutine to run
            user_id: the user id to link the task with
            thread_id: the thread id to link the task with
            timeout: optional timeout in seconds. If the task doesn't complete
                     within this time, it will be cancelled with TimeoutError
        """
        if self._shutting_down:
            logger.warning("Task manager is shutting down, refusing new task")
            coro.close()
            return None

        task_id = self._get_task_id(user_id=user_id, thread_id=thread_id)

        # Wrap with timeout if specified
        if timeout is not None:
            coro = self._wrap_with_timeout(coro, timeout)
        logger.info(f"Starting background task {coro.__name__} for user {user_id}...")
        task = asyncio.create_task(coro)
        if self._tasks.get(task_id):
            self._tasks[task_id].add(task)
        else:
            self._tasks[task_id] = {task}

        self._task_to_user_id[task] = task_id
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
            if user_id := self._task_to_user_id.pop(task, None):
                self._tasks.get(user_id, set()).discard(task)

    async def shutdown(self, *args: Any) -> None:
        """
        Wait for all background tasks to complete with a timeout.
        If tasks do not finish within 15 seconds, they are cancelled.
        """
        logger.info("Shutting down background tasks...")
        self._shutting_down = True
        tasks_to_wait = list(self._task_to_user_id.keys())
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

    def kill_all_user_tasks(self, user_id: int, thread_id: int) -> None:
        logger.info(f"Killing all tasks for user {user_id} in thread {thread_id}...")
        task_id = self._get_task_id(user_id=user_id, thread_id=thread_id)
        user_tasks = self._tasks.pop(task_id, None)
        if not user_tasks:
            logger.info(f"Nothing to kill there: the user {user_id} has no running tasks.")
            return None

        for task in user_tasks:
            self._task_to_user_id.pop(task, None)
            task.cancel()

        logger.info(f"Cancelled {len(user_tasks)} tasks connected to user {user_id}.")
        return None


task_manager = BackgroundTaskManager()
