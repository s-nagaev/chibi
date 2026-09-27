"""End-to-end tests for the scheduler v1 pipeline (design §3.3, §5.2, §6.5).

Covers the full path in a single scenario: a job created through the agent
tool → persistence in the Redis job store (fakeredis) → firing by schedule →
agent job dispatcher → `self` action → LLM chain → delivery or ACK silence.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fakeredis import FakeStrictRedis

from chibi.schemas.app import ChatResponseSchema
from chibi.services.jobs.agent_task import run_agent_job
from chibi.services.providers.tools.scheduler import ScheduleTaskTool
from chibi.services.scheduler import ChibiScheduler
from chibi.services.scheduler_interface import SchedulerInterface
from chibi.utils.app import SingletonMeta

pytestmark = pytest.mark.asyncio

TOOL_MODULE = "chibi.services.providers.tools.scheduler"
SCHEDULER_MODULE = "chibi.services.scheduler"
TICK_TIMEOUT_SECONDS = 30.0


async def _wait_until(predicate, timeout: float = TICK_TIMEOUT_SECONDS, poll_interval: float = 0.05) -> None:
    """Poll a predicate until it becomes True or the timeout expires.

    Args:
        predicate: Zero-argument callable checked on every poll iteration.
        timeout: Maximum time to wait in seconds.
        poll_interval: Sleep between polls in seconds.

    Raises:
        AssertionError: If the predicate is still False when the timeout expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll_interval)
    raise AssertionError("Condition was not met within the timeout")


@pytest.fixture()
def redis_scheduler_settings(tmp_path):
    """Scheduler settings pointed at a (faked) Redis URL and a temp data path."""
    return SimpleNamespace(
        redis="redis://localhost:6379/0",
        redis_password=None,
        local_data_path=str(tmp_path),
        scheduler_misfire_grace_time=3600,
    )


@pytest.fixture()
def tool_settings():
    """Tool-gate settings with every scheduler gate enabled."""
    return SimpleNamespace(
        scheduler_tool_enabled=True,
        scheduler_notify_enabled=True,
        scheduler_agent_commands_enabled=True,
    )


@pytest.fixture()
def interface():
    """A user interface stub carrying a full Telegram session context."""
    mock = MagicMock()
    mock.user_id = 123
    mock.storage_id = 111
    mock.chat_id = 555
    mock.thread_id = 42
    return mock


@pytest.fixture()
def fresh_scheduler_singleton():
    """Ensure a clean ChibiScheduler singleton around the test."""
    SingletonMeta._instances.pop(ChibiScheduler, None)
    yield
    SingletonMeta._instances.pop(ChibiScheduler, None)


@pytest.mark.usefixtures("fresh_scheduler_singleton")
async def test_interval_self_job_end_to_end(redis_scheduler_settings, tool_settings, interface) -> None:
    """A tool-created interval job survives the job store, fires twice and delivers on the second tick."""
    ack_response = ChatResponseSchema(answer="<chibi>ACK</chibi>", provider="OpenAI", model="m", usage=None)
    report_response = ChatResponseSchema(answer="All systems nominal", provider="OpenAI", model="m", usage=None)
    responses = [ack_response, report_response]
    llm_calls: list[int] = []

    async def fake_llm_answer(**kwargs) -> ChatResponseSchema:
        """Return the ACK answer on the first tick and the report on every later tick.

        Args:
            kwargs: Ignored LLM chain arguments.

        Returns:
            The scripted chat response for the current tick.
        """
        index = min(len(llm_calls), len(responses) - 1)
        llm_calls.append(index)
        return responses[index]

    with (
        patch(f"{SCHEDULER_MODULE}.application_settings", redis_scheduler_settings),
        patch("apscheduler.jobstores.redis.Redis", FakeStrictRedis),
        patch(f"{TOOL_MODULE}.application_settings", tool_settings),
        patch.object(SchedulerInterface, "send_message", new=AsyncMock()) as send_mock,
        patch("chibi.services.bot.get_llm_chat_completion_answer", new=AsyncMock(side_effect=fake_llm_answer)),
        patch("chibi.services.bot.check_history_and_summarize", new=AsyncMock(return_value=False)),
    ):
        result = await ScheduleTaskTool.function(
            title="Daily check",
            schedule={"kind": "interval", "every_seconds": 1},
            action={"type": "self", "trigger_text": "Run the daily check"},
            user_id=123,
            caller_model="test-model",
            interface=interface,
        )

        assert result["status"] == "ok"
        job_id = result["job_id"]

        scheduler = ChibiScheduler()
        jobs = scheduler.get_jobs()
        assert [job.id for job in jobs] == [job_id]
        job = jobs[0]
        assert job.func is run_agent_job
        assert job.kwargs["user_id"] == 123
        assert job.kwargs["storage_id"] == 111
        assert job.kwargs["thread_id"] == 42
        assert job.kwargs["chat_id"] == 555
        assert job.kwargs["action"]["type"] == "self"
        assert job.kwargs["action"]["trigger_text"] == "Run the daily check"

        scheduler.start()
        try:
            await _wait_until(lambda: len(llm_calls) >= 1)
            await asyncio.sleep(0.2)
            send_mock.assert_not_awaited()

            await _wait_until(lambda: send_mock.await_count >= 1)
            scheduler.remove_scheduled_job(job_id)

            first_send = send_mock.await_args_list[0]
            assert first_send.kwargs["message"] == "All systems nominal"
        finally:
            scheduler.shutdown(wait=False)
