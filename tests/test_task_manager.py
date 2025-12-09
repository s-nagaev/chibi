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
    BackgroundTaskManager._instance = None
    mgr = BackgroundTaskManager()
    yield mgr
    await mgr.shutdown()
    BackgroundTaskManager._instance = None


@pytest.mark.asyncio
async def test_run_task_adds_task(manager):
    async def dummy_task():
        await asyncio.sleep(0.01)
        return "done"

    manager.run_task(dummy_task())

    assert len(manager._tasks) == 1

    # Wait for task to complete
    await asyncio.sleep(0.02)

    # Task should be removed from set
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_task_exception_handling(manager):
    async def failing_task():
        raise ValueError("Oops")

    # Should not raise exception to the caller
    manager.run_task(failing_task())

    # Wait for task to fail and callback to run
    await asyncio.sleep(0.01)

    # Task should be removed even if failed
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_shutdown_waits_for_tasks(manager):
    cleanup_done = False

    async def long_task():
        nonlocal cleanup_done
        await asyncio.sleep(0.1)
        cleanup_done = True

    manager.run_task(long_task())

    await manager.shutdown()

    assert cleanup_done is True
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_shutdown_timeout_cancels_tasks(manager):
    # We need to patch asyncio.wait_for to simulate timeout or use a very short timeout in test
    # But since wait_for is used inside shutdown with hardcoded 15s, we can't easily change it without patching.
    # However, for this test, let's try to mock the timeout behavior by creating a task
    # that sleeps longer than we want to wait,
    # but we don't want to wait 15s in test. So we must patch `asyncio.wait_for`.

    task_cancelled = False

    async def forever_task():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    manager.run_task(forever_task())

    # Allow the task to start
    await asyncio.sleep(0.01)

    # Patch wait_for to raise TimeoutError immediately
    # We must patch it where it is imported/used, which is asyncio itself in this case
    # Since the code calls asyncio.wait_for directly, patching asyncio.wait_for works.

    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        await manager.shutdown()

    # Give the loop a chance to process the cancellation and the task to wake up
    await asyncio.sleep(0.01)

    assert task_cancelled is True
    assert len(manager._tasks) == 0


@pytest.mark.asyncio
async def test_singleton_behavior():
    BackgroundTaskManager._instance = None
    m1 = BackgroundTaskManager()
    m2 = BackgroundTaskManager()
    assert m1 is m2

    # Check that init doesn't reset tasks if called again
    async def t():
        await asyncio.sleep(0.1)

    m1.run_task(t())
    assert len(m2._tasks) == 1

    await m1.shutdown()
