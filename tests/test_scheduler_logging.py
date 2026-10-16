"""Tests for the APScheduler stdlib→loguru bridge and scheduler event logging."""

import logging
import re
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from apscheduler.events import EVENT_JOB_MISSED, JobExecutionEvent
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from chibi.config.app import application_settings
from chibi.config.logging import (
    SCHEDULER_LOGGER_NAME,
    StdlibToLoguruHandler,
    install_scheduler_log_bridge,
)
from chibi.services.scheduler import ChibiScheduler, StdioScheduler, _next_run_time
from chibi.utils.app import SingletonMeta


@pytest.fixture()
def restore_default_bridge() -> Iterator[None]:
    """Remove every handler from the apscheduler logger after the test."""
    yield
    scheduler_logger = logging.getLogger(SCHEDULER_LOGGER_NAME)
    for handler in list(scheduler_logger.handlers):
        if isinstance(handler, StdlibToLoguruHandler):
            scheduler_logger.removeHandler(handler)


@pytest.fixture()
def log_capture() -> Iterator[list]:
    """Capture loguru records at DEBUG level into a plain list."""
    records: list = []
    handler_id = logger.add(records.append, level="DEBUG")
    yield records
    logger.remove(handler_id)


@pytest.fixture()
def scheduler(tmp_path):
    """Provide a fresh ChibiScheduler backed by a temporary SQLite job store."""
    settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)
    SingletonMeta._instances.pop(ChibiScheduler, None)
    with patch("chibi.services.scheduler.application_settings", settings):
        yield ChibiScheduler()
    SingletonMeta._instances.pop(ChibiScheduler, None)


@pytest.fixture()
def stdio_scheduler(tmp_path):
    """Provide a fresh StdioScheduler backed by a temporary SQLite job store."""
    settings = SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)
    SingletonMeta._instances.pop(StdioScheduler, None)
    with patch("chibi.services.scheduler.application_settings", settings):
        yield StdioScheduler()
    SingletonMeta._instances.pop(StdioScheduler, None)


async def dummy_job(**kwargs) -> None:
    """No-op job function used by the tests."""


class TestStdlibBridge:
    """The stdlib→loguru bridge for the apscheduler.* logger family."""

    def test_warning_record_reaches_loguru_sink(self, log_capture, restore_default_bridge) -> None:
        install_scheduler_log_bridge()
        stdlib_logger = logging.getLogger("apscheduler.executors.default")
        stdlib_logger.warning('Run time of job "sleep" was missed by 2:00:00.004039')
        assert len(log_capture) == 1
        record = log_capture[0].record
        assert record["level"].name == "WARNING"
        assert 'Run time of job "sleep" was missed by 2:00:00.004039' in record["message"]
        # The stdlib logger name is embedded so `grep -i scheduler` can match.
        assert "[apscheduler.executors.default]" in record["message"]

    def test_debug_record_below_default_floor_is_dropped(self, log_capture, restore_default_bridge) -> None:
        install_scheduler_log_bridge()  # default floor: INFO
        logging.getLogger("apscheduler.scheduler").debug("Next wakeup is due at 2026-10-16 07:00:00+02:00")
        assert log_capture == []

    def test_debug_floor_passes_debug_records(self, log_capture, restore_default_bridge) -> None:
        install_scheduler_log_bridge(level="DEBUG")
        stdlib_logger = logging.getLogger("apscheduler.scheduler")
        stdlib_logger.debug("Next wakeup is due at 2026-10-16 07:00:00+02:00 (in 67968 seconds)")
        assert len(log_capture) == 1
        assert log_capture[0].record["level"].name == "DEBUG"

    def test_debug_flag_lowers_floor_without_explicit_level(self, log_capture, restore_default_bridge, monkeypatch):
        monkeypatch.setattr(application_settings, "scheduler_debug_logs", True)
        install_scheduler_log_bridge()
        logging.getLogger("apscheduler.scheduler").debug('Added job "agent:1:daily" to job store "default"')
        assert len(log_capture) == 1
        assert log_capture[0].record["level"].name == "DEBUG"

    def test_install_is_idempotent(self, log_capture, restore_default_bridge) -> None:
        install_scheduler_log_bridge()
        install_scheduler_log_bridge()
        handlers = [
            h for h in logging.getLogger(SCHEDULER_LOGGER_NAME).handlers if isinstance(h, StdlibToLoguruHandler)
        ]
        assert len(handlers) == 1
        logging.getLogger("apscheduler.scheduler").warning("only once")
        assert len(log_capture) == 1

    def test_no_last_resort_double_output(self, log_capture, capsys, restore_default_bridge) -> None:
        install_scheduler_log_bridge()
        scheduler_logger = logging.getLogger(SCHEDULER_LOGGER_NAME)
        assert scheduler_logger.propagate is False
        scheduler_logger.warning("no bare stderr line")
        assert len(log_capture) == 1
        # With propagation off the record must not leak through logging.lastResort.
        assert "no bare stderr line" not in capsys.readouterr().err

    def test_non_apscheduler_logger_is_not_bridged(self, log_capture, restore_default_bridge) -> None:
        install_scheduler_log_bridge()
        logging.getLogger("some.other.library").warning("not a scheduler record")
        assert log_capture == []


class TestMissedRunWarning:
    """Chibi-level WARNING on missed runs, independent of the stdlib bridge."""

    def test_missed_event_logs_warning_with_job_id_and_duration(self, scheduler, log_capture) -> None:
        run_time = datetime.now(timezone.utc) - timedelta(hours=2)
        scheduler._scheduler._dispatch_event(
            JobExecutionEvent(EVENT_JOB_MISSED, "agent:134604548:daily-report", "default", run_time)
        )
        warnings = [entry.record for entry in log_capture if entry.record["level"].name == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0]["message"]
        assert "agent:134604548:daily-report" in message
        assert "missed by" in message
        assert "2:00:00" in message

    def test_non_missed_events_are_ignored(self, scheduler, log_capture) -> None:
        scheduler._scheduler._dispatch_event(
            JobExecutionEvent(4096, "agent:1:probe", "default", datetime.now(timezone.utc))  # EVENT_JOB_ERROR
        )
        warnings = [entry.record for entry in log_capture if entry.record["level"].name == "WARNING"]
        assert warnings == []


class TestSchedulingEventLogging:
    """INFO on successful scheduling, WARNING on scheduling failure."""

    def test_success_logs_owner_and_next_run_time(self, scheduler, log_capture) -> None:
        run_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        scheduler.schedule_once_job(job_id="agent:134604548:probe", func=dummy_job, run_at=run_at)
        infos = [entry.record for entry in log_capture if entry.record["level"].name == "INFO"]
        message = " ".join(record["message"] for record in infos)
        assert "agent:134604548:probe" in message
        assert "owner: 134604548" in message
        # The value after "next run at" must be a real timestamp, not None
        # (DateTrigger has no .timezone; the run_date must be used instead).
        after = message.split("next run at ", 1)[1]
        assert not after.startswith("None")
        assert re.match(r"\d{4}-\d{2}-\d{2}", after)

    def test_success_logs_system_owner(self, scheduler, log_capture) -> None:
        scheduler.schedule_interval_job(job_id="system:probe", func=dummy_job, interval_seconds=60)
        infos = [entry.record for entry in log_capture if entry.record["level"].name == "INFO"]
        message = " ".join(record["message"] for record in infos)
        assert "owner: system" in message
        assert "next run at" in message

    def test_failure_logs_warning_and_reraises(self, scheduler, log_capture) -> None:
        with patch.object(scheduler._scheduler, "add_job", side_effect=ValueError("boom")):
            with pytest.raises(ValueError, match="boom"):
                scheduler.schedule_once_job(
                    job_id="agent:134604548:probe",
                    func=dummy_job,
                    run_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                )
        warnings = [entry.record for entry in log_capture if entry.record["level"].name == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0]["message"]
        assert "agent:134604548:probe" in message
        assert "boom" in message
        assert "Failed to schedule" in message


class TestStdioSchedulerObservability:
    """The stdio/IDE/TUI flavour installs the same observability as the Telegram one."""

    def test_bridge_and_missed_listener_are_installed(self, stdio_scheduler, restore_default_bridge) -> None:
        scheduler_logger = logging.getLogger(SCHEDULER_LOGGER_NAME)
        handlers = [h for h in scheduler_logger.handlers if isinstance(h, StdlibToLoguruHandler)]
        assert len(handlers) == 1
        assert scheduler_logger.propagate is False
        # The EVENT_JOB_MISSED listener must be registered on the stdio
        # scheduler's own AsyncIOScheduler instance.
        assert (stdio_scheduler._on_scheduler_event, EVENT_JOB_MISSED) in [
            (callback, event) for callback, event in stdio_scheduler._scheduler._listeners
        ]

    def test_missed_event_logs_warning(self, stdio_scheduler, log_capture) -> None:
        run_time = datetime.now(timezone.utc) - timedelta(hours=2)
        stdio_scheduler._scheduler._dispatch_event(
            JobExecutionEvent(EVENT_JOB_MISSED, "agent:134604548:probe", "default", run_time)
        )
        warnings = [entry.record for entry in log_capture if entry.record["level"].name == "WARNING"]
        assert len(warnings) == 1
        message = warnings[0]["message"]
        assert "agent:134604548:probe" in message
        assert "missed by" in message
        assert "2:00:00" in message

    def test_debug_flag_lowers_bridge_floor(self, stdio_scheduler, log_capture, monkeypatch) -> None:
        monkeypatch.setattr(application_settings, "scheduler_debug_logs", True)
        # The bridge is installed at construction time; re-install is idempotent
        # and re-applies the floor from the flag.
        install_scheduler_log_bridge()
        logging.getLogger("apscheduler.scheduler").debug("Next wakeup is due at 2026-10-17 07:00:00+02:00")
        assert len(log_capture) == 1
        assert log_capture[0].record["level"].name == "DEBUG"


class TestNextRunTimeForDateTrigger:
    """``_next_run_time`` must return the run date for one-time (date) jobs."""

    def test_date_trigger_returns_run_date(self) -> None:
        run_date = datetime.now(timezone.utc) + timedelta(minutes=5)
        job = SimpleNamespace(trigger=DateTrigger(run_date=run_date))
        assert _next_run_time(job) == run_date

    def test_missing_timezone_trigger_returns_none(self) -> None:
        trigger = SimpleNamespace()  # neither run_date nor timezone
        assert _next_run_time(SimpleNamespace(trigger=trigger)) is None

    def test_missing_trigger_returns_none(self) -> None:
        assert _next_run_time(SimpleNamespace()) is None
