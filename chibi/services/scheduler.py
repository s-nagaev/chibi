"""Persistent job scheduler for Chibi built on APScheduler."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.base import STATE_STOPPED
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from chibi.config import application_settings
from chibi.exceptions import SchedulerJobError
from chibi.schemas.scheduler import JobInfo
from chibi.utils.app import SingletonMeta

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from apscheduler.job import Job
    from apscheduler.triggers.base import BaseTrigger

SYSTEM_JOB_PREFIX = "system:"
AGENT_JOB_PREFIX = "agent:"
RETENTION_CLEANUP_JOB_ID = "system:retention_cleanup"
_JOB_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _validate_job_id(job_id: str) -> None:
    """Validate a fully qualified job identifier against the namespace contract.

    Allowed formats are ``system:<suffix>`` for built-in Chibi jobs and
    ``agent:<user_id>:<suffix>`` for agent-created jobs, where ``<user_id>`` is
    a numeric Telegram user id and ``<suffix>`` matches
    ``^[a-z0-9][a-z0-9_-]{0,63}$``.

    Args:
        job_id: Job identifier to validate.

    Raises:
        SchedulerJobError: If the job_id does not follow the namespace contract.
    """
    if job_id.startswith(SYSTEM_JOB_PREFIX):
        suffix = job_id.removeprefix(SYSTEM_JOB_PREFIX)
    elif job_id.startswith(AGENT_JOB_PREFIX):
        parts = job_id.split(":", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            raise SchedulerJobError(f"Invalid agent job_id {job_id!r}: expected format 'agent:<user_id>:<suffix>'")
        suffix = parts[2]
    else:
        raise SchedulerJobError(
            f"Invalid job_id {job_id!r}: must start with '{SYSTEM_JOB_PREFIX}' or '{AGENT_JOB_PREFIX}'"
        )
    if not _JOB_ID_SUFFIX_PATTERN.fullmatch(suffix):
        raise SchedulerJobError(f"Invalid job_id {job_id!r}: suffix {suffix!r} must match ^[a-z0-9][a-z0-9_-]{{0,63}}$")


def _summarize_kwargs(kwargs: dict[str, Any], max_length: int = 200) -> str | None:
    """Build a truncated, single-line summary of job keyword arguments.

    Args:
        kwargs: Keyword arguments stored on the job.
        max_length: Maximum length of the resulting summary.

    Returns:
        Human-readable kwargs summary, or None when there are no kwargs.
    """
    if not kwargs:
        return None
    summary = ", ".join(f"{key}={value!r}" for key, value in sorted(kwargs.items()))
    if len(summary) > max_length:
        summary = summary[: max_length - 1] + "…"
    return summary


class ChibiScheduler(metaclass=SingletonMeta):
    """Singleton wrapper around AsyncIOScheduler with a persistent job store.

    A Redis job store is used when ``application_settings.redis`` is configured;
    otherwise the scheduler falls back to a SQLite-backed SQLAlchemy job store.

    Known limitation (v1): the SQLAlchemy/SQLite fallback performs blocking
    SQLite I/O on the event loop whenever the job store is accessed. This is an
    accepted trade-off for local deployments without Redis and is not fixed in v1.

    All new jobs must use the namespace contract enforced by the
    ``schedule_*_job`` methods: ``system:<suffix>`` for built-in jobs and
    ``agent:<user_id>:<suffix>`` for agent-created jobs.
    """

    def __init__(self) -> None:
        """Initialize the scheduler and its persistent job store."""
        if hasattr(self, "_scheduler"):
            return
        if application_settings.redis:
            parsed = urlparse(application_settings.redis)
            password = parsed.password or application_settings.redis_password
            job_store = RedisJobStore(
                host=parsed.hostname or "localhost",
                port=parsed.port or 6379,
                db=int(parsed.path.lstrip("/") or 0),
                password=password,
            )
            logger.info("Scheduler: using Redis job store ({}:{})", parsed.hostname, parsed.port)
        else:
            db_path = Path(application_settings.local_data_path) / "scheduler.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            job_store = SQLAlchemyJobStore(url=f"sqlite:///{db_path}")
            logger.info("Scheduler: using SQLite job store ({})", db_path)

        self._scheduler = AsyncIOScheduler(jobstores={"default": job_store})

    def start(self) -> None:
        """Start the scheduler.

        No-op when the scheduler is already running, so repeated calls (e.g.
        from a repeated ``post_init``) are safe.
        """
        if self._scheduler.state != STATE_STOPPED:
            return
        self._scheduler.start()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the scheduler.

        Args:
            wait: Whether to wait until all currently executing jobs have finished.
        """
        self._scheduler.shutdown(wait=wait)

    @property
    def state(self) -> int:
        """Return the current state of the underlying APScheduler instance.

        Use :data:`apscheduler.schedulers.base.STATE_STOPPED` (and the other
        ``STATE_*`` constants) to interpret the value.
        """
        return self._scheduler.state

    def add_job(
        self,
        func: Callable[..., Coroutine[Any, Any, None]],
        trigger: str | BaseTrigger | None = None,
        *,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        id: str | None = None,
        name: str | None = None,
        **options: Any,
    ) -> Job:
        """Add a job to the scheduler.

        Thin typed passthrough to APScheduler's ``add_job`` kept for backward
        compatibility with legacy call sites. New code should prefer the
        ``schedule_*_job`` methods, which enforce the job_id namespace contract
        and misfire policy defaults.

        Args:
            func: Async callable to run when the job is triggered.
            trigger: Trigger or trigger alias (e.g. ``"interval"``, ``"cron"``, ``"date"``).
            args: Positional arguments for the job function.
            kwargs: Keyword arguments for the job function.
            id: Explicit job identifier.
            name: Human-readable job name.
            options: Additional APScheduler job options (e.g. ``misfire_grace_time``,
                ``next_run_time``, ``replace_existing``) and trigger-specific
                arguments (e.g. ``days``, ``seconds``).

        Returns:
            The scheduled APScheduler job.
        """
        return self._scheduler.add_job(func, trigger=trigger, args=args, kwargs=kwargs, id=id, name=name, **options)

    def schedule_interval_job(
        self,
        *,
        job_id: str,
        func: Callable[..., Coroutine[Any, Any, None]],
        interval_seconds: int,
        kwargs: dict[str, Any] | None = None,
        replace: bool = False,
        misfire_grace_time: int | None = None,
        next_run_time: datetime | None = None,
    ) -> str:
        """Schedule a recurring job on a fixed interval.

        Applies the default misfire policy: bounded catch-up grace time,
        ``coalesce=True``, ``max_instances=1``.

        Args:
            job_id: Fully qualified job identifier (``system:<suffix>`` or ``agent:<user_id>:<suffix>``).
            func: Async callable to run on every tick.
            interval_seconds: Interval between runs in seconds.
            kwargs: Keyword arguments passed to the job function.
            replace: Whether to replace an existing job with the same id.
            misfire_grace_time: Grace period in seconds for catching up missed
                runs. None (default) uses ``application_settings.scheduler_misfire_grace_time``.
            next_run_time: Time of the first run. None (default) lets the
                trigger compute it (i.e. one full interval from now). Naive
                datetimes are interpreted in the scheduler's local timezone.

        Returns:
            The scheduled job identifier.

        Raises:
            SchedulerJobError: If the job_id violates the namespace contract.
        """
        _validate_job_id(job_id)
        grace = application_settings.scheduler_misfire_grace_time if misfire_grace_time is None else misfire_grace_time
        options: dict[str, Any] = {}
        if next_run_time is not None:
            # Passing next_run_time=None to APScheduler's add_job would pause the job;
            # the default (`undefined`) must be used instead so the trigger computes it.
            options["next_run_time"] = next_run_time
        self._scheduler.add_job(
            func,
            trigger="interval",
            seconds=interval_seconds,
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=replace,
            misfire_grace_time=grace,
            coalesce=True,
            max_instances=1,
            **options,
        )
        logger.info("Scheduled interval job {} every {}s (misfire_grace_time={})", job_id, interval_seconds, grace)
        return job_id

    def schedule_cron_job(
        self,
        *,
        job_id: str,
        func: Callable[..., Coroutine[Any, Any, None]],
        cron: str,
        kwargs: dict[str, Any] | None = None,
        replace: bool = False,
        misfire_grace_time: int | None = None,
    ) -> str:
        """Schedule a recurring job from a crontab expression.

        Applies the default misfire policy: bounded catch-up grace time,
        ``coalesce=True``, ``max_instances=1``.

        Args:
            job_id: Fully qualified job identifier (``system:<suffix>`` or ``agent:<user_id>:<suffix>``).
            func: Async callable to run on every trigger.
            cron: Crontab expression with exactly 5 fields
                (minute, hour, day of month, month, day of week).
            kwargs: Keyword arguments passed to the job function.
            replace: Whether to replace an existing job with the same id.
            misfire_grace_time: Grace period in seconds for catching up missed
                runs. None (default) uses ``application_settings.scheduler_misfire_grace_time``.

        Returns:
            The scheduled job identifier.

        Raises:
            SchedulerJobError: If the job_id violates the namespace contract or
                the cron expression does not have exactly 5 fields or is invalid.
        """
        _validate_job_id(job_id)
        fields = cron.split()
        if len(fields) != 5:
            raise SchedulerJobError(
                f"Invalid cron expression {cron!r}: expected exactly 5 fields "
                f"(minute hour day-of-month month day-of-week), got {len(fields)}"
            )
        try:
            trigger = CronTrigger.from_crontab(cron)
        except ValueError as error:
            raise SchedulerJobError(f"Invalid cron expression {cron!r}: {error}") from error
        grace = application_settings.scheduler_misfire_grace_time if misfire_grace_time is None else misfire_grace_time
        self._scheduler.add_job(
            func,
            trigger=trigger,
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=replace,
            misfire_grace_time=grace,
            coalesce=True,
            max_instances=1,
        )
        logger.info("Scheduled cron job {} ({!r}, misfire_grace_time={})", job_id, cron, grace)
        return job_id

    def schedule_once_job(
        self,
        *,
        job_id: str,
        func: Callable[..., Coroutine[Any, Any, None]],
        run_at: datetime,
        kwargs: dict[str, Any] | None = None,
        misfire_grace_time: int | None = None,
    ) -> str:
        """Schedule a one-time job.

        Args:
            job_id: Fully qualified job identifier (``system:<suffix>`` or ``agent:<user_id>:<suffix>``).
            func: Async callable to run once.
            run_at: Datetime of the single run. Naive datetimes are interpreted
                in the scheduler's local timezone.
            kwargs: Keyword arguments passed to the job function.
            misfire_grace_time: Grace period in seconds for the delayed run.
                None (default) means unlimited: the reminder always fires no
                matter how late the process wakes up, so one-time reminders are
                never silently dropped.

        Returns:
            The scheduled job identifier.

        Raises:
            SchedulerJobError: If the job_id violates the namespace contract.
        """
        _validate_job_id(job_id)
        self._scheduler.add_job(
            func,
            trigger="date",
            run_date=run_at,
            kwargs=kwargs or {},
            id=job_id,
            replace_existing=True,
            misfire_grace_time=misfire_grace_time,
            coalesce=True,
            max_instances=1,
        )
        logger.info("Scheduled one-time job {} at {}", job_id, run_at)
        return job_id

    def list_jobs(self, prefix: str | None = None) -> list[JobInfo]:
        """List scheduled jobs, optionally filtered by job_id prefix.

        Args:
            prefix: If provided, only jobs whose id starts with this prefix
                (e.g. ``"system:"`` or ``"agent:134604548:"``) are returned.

        Returns:
            Summary information for the matching scheduled jobs.
        """
        jobs: list[JobInfo] = []
        for job in self._scheduler.get_jobs():
            current_job_id = str(job.id)
            if prefix is not None and not current_job_id.startswith(prefix):
                continue
            jobs.append(
                JobInfo(
                    job_id=current_job_id,
                    trigger=str(job.trigger),
                    next_run_time=getattr(job, "next_run_time", None),
                    kwargs_summary=_summarize_kwargs(job.kwargs or {}),
                )
            )
        return jobs

    def remove_scheduled_job(self, job_id: str) -> None:
        """Remove a scheduled job by its identifier.

        Args:
            job_id: Exact identifier of the job to remove.

        Raises:
            apscheduler.jobstores.base.JobLookupError: If no job with the given id exists.
        """
        self._scheduler.remove_job(job_id)

    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job by its identifier (legacy alias).

        Args:
            job_id: Exact identifier of the job to remove.

        Raises:
            apscheduler.jobstores.base.JobLookupError: If no job with the given id exists.
        """
        self._scheduler.remove_job(job_id)

    def get_jobs(self) -> list[Job]:
        """Return all scheduled jobs from the job store.

        Returns:
            The list of scheduled APScheduler jobs.
        """
        return self._scheduler.get_jobs()
