"""Tests for the Telegram runner scheduler lifecycle (post_init / post_shutdown)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.schedulers.base import STATE_STOPPED

from chibi.runners.telegram import (
    ChibiBot,
    _register_retention_cleanup_job,
    _shutdown_scheduler_and_tasks,
)
from chibi.services.scheduler import RETENTION_CLEANUP_JOB_ID, ChibiScheduler
from chibi.utils.app import SingletonMeta


@pytest.fixture()
def sqlite_scheduler_settings(tmp_path):
    """Return scheduler settings patched onto the scheduler module."""
    return SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)


@pytest.fixture()
def fresh_scheduler_singleton():
    """Ensure a clean ChibiScheduler singleton around the test."""
    SingletonMeta._instances.pop(ChibiScheduler, None)
    yield
    SingletonMeta._instances.pop(ChibiScheduler, None)


def make_bot() -> tuple[ChibiBot, MagicMock]:
    """Build a ChibiBot plus a mocked Telegram bot client for post_init."""
    bot = ChibiBot("test-token")
    telegram_bot_mock = MagicMock()
    telegram_bot_mock.get_me = AsyncMock(return_value=SimpleNamespace(has_topics_enabled=False))
    return bot, telegram_bot_mock


class TestPostInitSchedulerLifecycle:
    """Tests for scheduler startup and retention job registration in post_init."""

    @pytest.mark.asyncio
    async def test_post_init_starts_scheduler_and_registers_retention_job(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """post_init must always start the scheduler and register the retention job."""
        bot, telegram_bot_mock = make_bot()
        application = MagicMock()
        application.bot.set_my_commands = AsyncMock()

        with (
            patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
            patch("chibi.runners.telegram.Bot", return_value=telegram_bot_mock),
        ):
            await bot.post_init(application)

            scheduler = ChibiScheduler()
            assert scheduler.get_jobs(), "Scheduler must contain the retention job after post_init"
            retention = [job for job in scheduler.get_jobs() if job.id == RETENTION_CLEANUP_JOB_ID]
            assert len(retention) == 1
            assert retention[0].misfire_grace_time == 3600
            assert retention[0].coalesce is True
            assert retention[0].max_instances == 1

            scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_repeated_post_init_does_not_duplicate_retention_job(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """Running post_init twice must leave exactly one retention job (fixed id + replace_existing)."""
        bot, telegram_bot_mock = make_bot()
        application = MagicMock()
        application.bot.set_my_commands = AsyncMock()

        with (
            patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
            patch("chibi.runners.telegram.Bot", return_value=telegram_bot_mock),
        ):
            await bot.post_init(application)
            await bot.post_init(application)

            scheduler = ChibiScheduler()
            jobs = scheduler.list_jobs(prefix=RETENTION_CLEANUP_JOB_ID)
            assert len(jobs) == 1
            assert jobs[0].job_id == RETENTION_CLEANUP_JOB_ID

            scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_retention_job_registered_even_without_memory(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """Registration must not depend on the chroma memory being configured."""
        bot, telegram_bot_mock = make_bot()
        application = MagicMock()
        application.bot.set_my_commands = AsyncMock()

        with (
            patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
            patch("chibi.runners.telegram.Bot", return_value=telegram_bot_mock),
            patch("chibi.runners.telegram.perform_retention_cleanup", new_callable=MagicMock),
        ):
            await bot.post_init(application)

            scheduler = ChibiScheduler()
            assert [job.id for job in scheduler.get_jobs()] == [RETENTION_CLEANUP_JOB_ID]

            scheduler.shutdown(wait=False)


class TestRegisterRetentionCleanupJob:
    """Tests for the retention job registration helper."""

    @pytest.mark.asyncio
    async def test_registering_twice_yields_single_job(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """The helper relies on a fixed id + replace_existing and stays idempotent."""
        with patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings):
            scheduler = ChibiScheduler()
            scheduler.start()
            try:
                _register_retention_cleanup_job(scheduler)
                _register_retention_cleanup_job(scheduler)
                assert [job.id for job in scheduler.get_jobs()] == [RETENTION_CLEANUP_JOB_ID]
            finally:
                scheduler.shutdown(wait=False)


class TestShutdown:
    """Tests for the combined post_shutdown wrapper."""

    @pytest.mark.asyncio
    async def test_shuts_down_scheduler_and_task_manager(self) -> None:
        """The wrapper must call both scheduler.shutdown and task_manager.shutdown."""
        with (
            patch("chibi.runners.telegram.ChibiScheduler") as mock_scheduler_cls,
            patch("chibi.runners.telegram.task_manager") as mock_task_manager,
        ):
            mock_task_manager.shutdown = AsyncMock()

            await _shutdown_scheduler_and_tasks(MagicMock())

            mock_scheduler_cls.return_value.shutdown.assert_called_once_with(wait=False)
            mock_task_manager.shutdown.assert_awaited_once()

    def test_run_registers_combined_shutdown_wrapper(self) -> None:
        """run() must register the combined wrapper instead of bare task_manager.shutdown."""
        bot = ChibiBot("test-token")
        with patch("chibi.runners.telegram.ApplicationBuilder") as builder_cls:
            builder = builder_cls.return_value
            for method in (
                "base_url",
                "base_file_url",
                "token",
                "post_init",
                "post_shutdown",
                "proxy",
                "get_updates_proxy",
            ):
                getattr(builder, method).return_value = builder
            builder.build.return_value = MagicMock()

            bot.run()

        registered = builder.post_shutdown.call_args.args[0]
        assert registered is _shutdown_scheduler_and_tasks


class TestShutdownGuard:
    """Tests for the never-started scheduler guard in the shutdown wrapper."""

    @pytest.mark.asyncio
    async def test_shutdown_wrapper_skips_never_started_scheduler(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """A scheduler that never started (failed post_init) must not raise on shutdown."""
        with (
            patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
            patch("chibi.runners.telegram.task_manager") as mock_task_manager,
        ):
            mock_task_manager.shutdown = AsyncMock()
            ChibiScheduler()

            await _shutdown_scheduler_and_tasks(MagicMock())

            mock_task_manager.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_wrapper_shuts_down_running_scheduler(
        self, fresh_scheduler_singleton, sqlite_scheduler_settings
    ) -> None:
        """A running scheduler is shut down by the wrapper and ends up stopped."""
        with (
            patch("chibi.services.scheduler.application_settings", sqlite_scheduler_settings),
            patch("chibi.runners.telegram.task_manager") as mock_task_manager,
        ):
            mock_task_manager.shutdown = AsyncMock()
            scheduler = ChibiScheduler()
            scheduler.start()

            await _shutdown_scheduler_and_tasks(MagicMock())

            mock_task_manager.shutdown.assert_awaited_once()
            await asyncio.sleep(0)
            assert scheduler.state == STATE_STOPPED
