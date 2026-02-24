import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Callable


@asynccontextmanager
async def indicator(
    coro_func: Callable[[], Coroutine[Any, Any, Any]],
    interval: float = 5.3,
) -> AsyncIterator[None]:
    """Repeatedly calls a coroutine function while the wrapped block is running.

    Useful for keeping a "typing..." or similar status indicator alive
    during long-running operations.

    Args:
        coro_func: A zero-argument callable that returns a coroutine.
            Called every ``interval`` seconds until the context exits.
        interval: Seconds to wait between consecutive calls.

    Yields:
        None
    """

    async def _keep_indicating() -> None:
        try:
            while True:
                await coro_func()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    task: asyncio.Task[None] = asyncio.create_task(_keep_indicating())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
