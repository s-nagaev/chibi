import asyncio
from unittest.mock import patch

import pytest

from chibi.services.task_manager import BackgroundTaskManager


@pytest.fixture
async def manager():
    # Create a fresh instance for each test
    # We need to bypass the singleton pattern for testing purposes or reset it
    # But since it's a singleton, we should probably reset the global one or patch it
    # Here we will reset the singleton instance
    mgr = BackgroundTaskManager()
    mgr._shutting_down = False
    mgr._tasks = {}
    mgr._task_to_user_id = {}
    yield mgr
    await mgr.shutdown()


@pytest.mark.asyncio
async def test_run_task_adds_task(manager):
    async def dummy_task():
        await asyncio.sleep(0.01)
        return "done"

    manager.run_task(dummy_task(), user_id=1)

    assert len(manager._tasks) == 1
    assert 1 in manager._tasks
    assert len(manager._tasks[1]) == 1

    # Wait for task to complete
    await asyncio.sleep(0.02)

    # Task should be removed from set
    assert len(manager._tasks[1]) == 0


@pytest.mark.asyncio
async def test_run_task_creates_user_set(manager):
    async def dummy_task():
        await asyncio.sleep(0.01)

    manager.run_task(dummy_task(), user_id=42)

    assert 42 in manager._tasks
    assert isinstance(manager._tasks[42], set)


@pytest.mark.asyncio
async def test_run_task_maps_task_to_user_id(manager):
    async def dummy_task():
        await asyncio.sleep(0.01)

    task = manager.run_task(dummy_task(), user_id=123)

    assert manager._task_to_user_id[task] == 123

    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_multiple_users(manager):
    async def dummy_task():
        await asyncio.sleep(0.01)

    manager.run_task(dummy_task(), user_id=1)
    manager.run_task(dummy_task(), user_id=1)
    manager.run_task(dummy_task(), user_id=2)

    assert len(manager._tasks) == 2
    assert len(manager._tasks[1]) == 2
    assert len(manager._tasks[2]) == 1

    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_task_exception_handling(manager):
    async def failing_task():
        raise ValueError("Oops")

    # Should not raise exception to the caller
    manager.run_task(failing_task(), user_id=1)

    # Wait for task to fail and callback to run
    await asyncio.sleep(0.01)

    # Task should be removed even if failed
    assert len(manager._tasks[1]) == 0


@pytest.mark.asyncio
async def test_shutdown_waits_for_tasks(manager):
    cleanup_done = False

    async def long_task():
        nonlocal cleanup_done
        await asyncio.sleep(0.1)
        cleanup_done = True

    manager.run_task(long_task(), user_id=1)

    await manager.shutdown()

    assert cleanup_done is True
    assert len(manager._tasks.get(1, set())) == 0


@pytest.mark.asyncio
async def test_shutdown_timeout_cancels_tasks(manager):
    # Note: shutdown has 5s timeout, we test cancellation by using a very short timeout in shutdown
    task_cancelled = False

    async def forever_task():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    manager.run_task(forever_task(), user_id=1)

    # Allow the task to start
    await asyncio.sleep(0.01)

    # Patch wait_for inside asyncio to simulate timeout immediately
    # original_wait_for = asyncio.wait_for

    async def mock_wait_for(coro, timeout):
        raise asyncio.TimeoutError()

    with patch.object(asyncio, "wait_for", mock_wait_for):
        await manager.shutdown()

    # Give the loop a chance to process the cancellation
    await asyncio.sleep(0.01)

    assert task_cancelled is True


@pytest.mark.asyncio
async def test_singleton_behavior():
    m1 = BackgroundTaskManager()
    m1._shutting_down = False
    m2 = BackgroundTaskManager()
    assert m1 is m2

    # Check that init doesn't reset tasks if called again
    async def t():
        await asyncio.sleep(0.1)

    m1.run_task(t(), user_id=1)
    assert len(m1._tasks) == 1

    await m1.shutdown()
