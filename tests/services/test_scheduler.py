"""Tests for the ChibiScheduler core."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from apscheduler.jobstores.base import ConflictingIdError, JobLookupError

from chibi.exceptions import SchedulerJobError
from chibi.services.jobs.archive import perform_retention_cleanup
from chibi.services.scheduler import (
    RETENTION_CLEANUP_JOB_ID,
    ChibiScheduler,
    _validate_job_id,
)
from chibi.utils.app import SingletonMeta


async def dummy_job(**kwargs) -> None:
    """No-op job function used by the tests."""


@pytest.fixture()
def scheduler(tmp_path):
    """Provide a fresh ChibiScheduler backed by a temporary SQLite job store."""
    settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)
    SingletonMeta._instances.pop(ChibiScheduler, None)
    with patch("chibi.services.scheduler.application_settings", settings):
        yield ChibiScheduler()
    SingletonMeta._instances.pop(ChibiScheduler, None)


class TestSingleton:
    """Tests for the ChibiScheduler singleton behavior."""

    def test_returns_same_instance(self, scheduler):
        assert ChibiScheduler() is scheduler

    def test_reinitialization_keeps_state(self, scheduler):
        scheduler.schedule_interval_job(job_id="system:singleton_check", func=dummy_job, interval_seconds=60)
        again = ChibiScheduler()
        assert again is scheduler
        assert [job.job_id for job in again.list_jobs(prefix="system:singleton_check")] == ["system:singleton_check"]


class TestJobIdValidation:
    """Tests for the job_id namespace contract."""

    @pytest.mark.parametrize(
        "job_id",
        [
            "system:retention_cleanup",
            "system:a",
            "system:a-b_c1",
            "agent:134604548:daily-report",
            "agent:1:x",
        ],
    )
    def test_valid_job_ids(self, job_id):
        _validate_job_id(job_id)

    @pytest.mark.parametrize(
        "job_id",
        [
            "retention_cleanup-20260927",
            "some_job",
            "system:",
            "system:UPPER",
            "system:-starts-with-dash",
            "system:" + "a" * 65,
            "agent:123",
            "agent:123:",
            "agent:abc:x",
            "agent:1:2:x",
            "agent:-1:x",
            "",
        ],
    )
    def test_invalid_job_ids(self, job_id):
        with pytest.raises(SchedulerJobError):
            _validate_job_id(job_id)

    def test_retention_constant_is_valid(self):
        assert RETENTION_CLEANUP_JOB_ID == "system:retention_cleanup"
        _validate_job_id(RETENTION_CLEANUP_JOB_ID)


class TestScheduleIntervalJob:
    """Tests for schedule_interval_job."""

    def test_schedules_with_misfire_defaults(self, scheduler):
        job_id = scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
        assert job_id == "system:tick"
        job = scheduler.get_jobs()[0]
        assert job.id == "system:tick"
        assert job.misfire_grace_time == 3600
        assert job.coalesce is True
        assert job.max_instances == 1

    def test_default_grace_time_comes_from_config(self, tmp_path):
        """The interval misfire grace default must follow the configured setting."""
        settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=60)
        SingletonMeta._instances.pop(ChibiScheduler, None)
        with patch("chibi.services.scheduler.application_settings", settings):
            scheduler = ChibiScheduler()
            scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
            assert scheduler.get_jobs()[0].misfire_grace_time == 60
        SingletonMeta._instances.pop(ChibiScheduler, None)

    def test_misfire_grace_time_override(self, scheduler):
        scheduler.schedule_interval_job(
            job_id="system:tick", func=dummy_job, interval_seconds=60, misfire_grace_time=120
        )
        assert scheduler.get_jobs()[0].misfire_grace_time == 120

    @pytest.mark.asyncio
    async def test_replace_semantics(self, scheduler):
        scheduler.start()
        try:
            scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
            with pytest.raises(ConflictingIdError):
                scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=30)
            scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=30, replace=True)
            jobs = scheduler.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].trigger.interval.total_seconds() == 30
        finally:
            scheduler.shutdown(wait=False)

    def test_invalid_job_id_rejected(self, scheduler):
        with pytest.raises(SchedulerJobError):
            scheduler.schedule_interval_job(job_id="no-namespace", func=dummy_job, interval_seconds=60)


class TestScheduleCronJob:
    """Tests for schedule_cron_job."""

    def test_schedules_valid_five_field_cron(self, scheduler):
        job_id = scheduler.schedule_cron_job(job_id="agent:123:morning", func=dummy_job, cron="0 9 * * *")
        assert job_id == "agent:123:morning"
        job = scheduler.get_jobs()[0]
        assert job.misfire_grace_time == 3600
        assert job.coalesce is True
        assert job.max_instances == 1

    @pytest.mark.parametrize("cron", ["0 9 * * * *", "*/10 * * * * *", "99 9 * * *", "* * * *"])
    def test_rejects_non_five_field_or_invalid_cron(self, scheduler, cron):
        with pytest.raises(SchedulerJobError):
            scheduler.schedule_cron_job(job_id="system:cron_job", func=dummy_job, cron=cron)


class TestScheduleOnceJob:
    """Tests for schedule_once_job."""

    def test_schedules_one_time_job_with_unlimited_grace(self, scheduler):
        run_at = datetime.now() + timedelta(minutes=5)
        job_id = scheduler.schedule_once_job(
            job_id="agent:123:remind-1", func=dummy_job, run_at=run_at, kwargs={"text": "wake up"}
        )
        assert job_id == "agent:123:remind-1"
        job = scheduler.get_jobs()[0]
        assert job.misfire_grace_time is None
        assert job.coalesce is True
        assert job.max_instances == 1

    def test_misfire_grace_time_override(self, scheduler):
        run_at = datetime.now() + timedelta(minutes=5)
        scheduler.schedule_once_job(job_id="agent:123:remind-1", func=dummy_job, run_at=run_at, misfire_grace_time=60)
        assert scheduler.get_jobs()[0].misfire_grace_time == 60

    @pytest.mark.asyncio
    async def test_replace_is_always_enabled(self, scheduler):
        scheduler.start()
        try:
            run_at = datetime.now() + timedelta(minutes=5)
            scheduler.schedule_once_job(job_id="agent:123:remind-1", func=dummy_job, run_at=run_at)
            scheduler.schedule_once_job(job_id="agent:123:remind-1", func=dummy_job, run_at=run_at)
            jobs = scheduler.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].id == "agent:123:remind-1"
        finally:
            scheduler.shutdown(wait=False)


class TestListJobs:
    """Tests for list_jobs and the JobInfo schema."""

    def test_list_and_prefix_filtering(self, scheduler):
        scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
        scheduler.schedule_cron_job(job_id="agent:123:report", func=dummy_job, cron="0 9 * * *")
        all_jobs = scheduler.list_jobs()
        assert {info.job_id for info in all_jobs} == {"system:tick", "agent:123:report"}
        agent_jobs = scheduler.list_jobs(prefix="agent:123:")
        assert [info.job_id for info in agent_jobs] == ["agent:123:report"]
        assert all(isinstance(info.trigger, str) for info in all_jobs)

    def test_kwargs_summary(self, scheduler):
        scheduler.schedule_interval_job(
            job_id="system:tick", func=dummy_job, interval_seconds=60, kwargs={"text": "hello"}
        )
        info = scheduler.list_jobs(prefix="system:")[0]
        assert info.kwargs_summary == "text='hello'"

    def test_kwargs_summary_truncated(self, scheduler):
        scheduler.schedule_interval_job(
            job_id="system:tick", func=dummy_job, interval_seconds=60, kwargs={"text": "x" * 500}
        )
        info = scheduler.list_jobs(prefix="system:")[0]
        assert info.kwargs_summary is not None
        assert len(info.kwargs_summary) <= 200
        assert info.kwargs_summary.endswith("…")

    def test_kwargs_summary_none_when_empty(self, scheduler):
        scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
        info = scheduler.list_jobs(prefix="system:")[0]
        assert info.kwargs_summary is None


class TestRemoveScheduledJob:
    """Tests for remove_scheduled_job."""

    def test_removes_existing_job(self, scheduler):
        scheduler.schedule_interval_job(job_id="system:tick", func=dummy_job, interval_seconds=60)
        scheduler.remove_scheduled_job("system:tick")
        assert scheduler.list_jobs() == []

    def test_raises_for_unknown_job(self, scheduler):
        with pytest.raises(JobLookupError):
            scheduler.remove_scheduled_job("system:missing")


class TestLegacyAddJob:
    """Tests for backward compatibility of the add_job passthrough."""

    def test_legacy_retention_registration_style(self, scheduler):
        legacy_id = f"retention_cleanup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        scheduler.add_job(
            dummy_job,
            trigger="interval",
            days=180,
            id=legacy_id,
            replace_existing=False,
            next_run_time=datetime.now(),
        )
        assert [job.id for job in scheduler.get_jobs()] == [legacy_id]


class TestRetentionJobRegistration:
    """Tests for the retention cleanup job registration contract (design §1.3)."""

    @pytest.mark.asyncio
    async def test_registration_with_fixed_id_is_idempotent(self, scheduler):
        """Re-registering the retention job must never accumulate duplicates."""
        scheduler.start()
        try:
            for _ in range(2):
                scheduler.add_job(
                    perform_retention_cleanup,
                    trigger="interval",
                    days=180,
                    id=RETENTION_CLEANUP_JOB_ID,
                    replace_existing=True,
                    next_run_time=datetime.now(),
                    misfire_grace_time=3600,
                    coalesce=True,
                    max_instances=1,
                )
            assert [job.id for job in scheduler.get_jobs()] == [RETENTION_CLEANUP_JOB_ID]
        finally:
            scheduler.shutdown(wait=False)


class TestRedisPasswordMerge:
    """Tests for the redis_password env fallback (design doc §3.2)."""

    def test_env_password_used_when_url_has_none(self):
        settings = SimpleNamespace(redis="redis://redis-host:6380/2", redis_password="envpass")
        SingletonMeta._instances.pop(ChibiScheduler, None)
        with (
            patch("chibi.services.scheduler.application_settings", settings),
            patch("chibi.services.scheduler.RedisJobStore") as mock_store,
            patch("chibi.services.scheduler.AsyncIOScheduler"),
        ):
            ChibiScheduler()
            kwargs = mock_store.call_args.kwargs
        SingletonMeta._instances.pop(ChibiScheduler, None)
        assert kwargs == {"host": "redis-host", "port": 6380, "db": 2, "password": "envpass"}

    def test_url_password_takes_priority(self):
        settings = SimpleNamespace(redis="redis://:urlpass@redis-host:6380/2", redis_password="envpass")
        SingletonMeta._instances.pop(ChibiScheduler, None)
        with (
            patch("chibi.services.scheduler.application_settings", settings),
            patch("chibi.services.scheduler.RedisJobStore") as mock_store,
            patch("chibi.services.scheduler.AsyncIOScheduler"),
        ):
            ChibiScheduler()
            kwargs = mock_store.call_args.kwargs
        SingletonMeta._instances.pop(ChibiScheduler, None)
        assert kwargs["password"] == "urlpass"


class TestDocumentedLimitations:
    """Tests for documented known limitations."""

    def test_sqlite_blocking_io_documented(self):
        assert ChibiScheduler.__doc__ is not None
        assert "blocking" in ChibiScheduler.__doc__
        assert "SQLAlchemy" in ChibiScheduler.__doc__
