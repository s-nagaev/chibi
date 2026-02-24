import asyncio
from contextlib import asynccontextmanager
from typing import Any, Coroutine


@asynccontextmanager
async def indicator(coro: Coroutine[Any, Any, Any], interval: float = 4.0):
    async def _keep_indicating():
        try:
            while True:
                await coro
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_keep_indicating())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
