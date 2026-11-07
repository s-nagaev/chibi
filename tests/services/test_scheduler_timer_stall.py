"""Tests for the silent-stall guards of ``GuardedAsyncIOScheduler``.

Production incident (APScheduler 3.11.3): agent jobs stopped firing FOREVER
while the scheduler looked healthy — during the silent window there were ZERO
``apscheduler`` DEBUG lines, no ``Running job``, no missed-run warnings. The
signature matches ``_process_jobs()`` returning ``None`` while jobs exist:
``AsyncIOScheduler._start_timer(None)`` arms NO timer at all, silently, and
nothing but the (dead) timer would ever call ``wakeup()`` again.

``GuardedAsyncIOScheduler`` guards against this in two layers: the wakeup
re-arm (nearest job ``next_run_time``, bounded) and a periodic self-check
watchdog that detects a disarmed timer or an overdue job and forces a
``wakeup()``. The watchdog always re-arms itself before running its check.

TODO(apscheduler-3.11.4): remove — delete this module together with
``GuardedAsyncIOScheduler`` once APScheduler 3.11.4 is adopted.
"""

import asyncio
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.base import STATE_PAUSED, STATE_RUNNING, STATE_STOPPED

from chibi.services.scheduler import ChibiScheduler, GuardedAsyncIOScheduler, StdioScheduler
from chibi.utils.app import SingletonMeta
from tests.services.test_scheduler_wakeup_guard import _make_settings, _wait_until

APSCHEDULER_LOGGER = "apscheduler.scheduler"


@pytest.fixture
def apscheduler_warnings() -> Iterator[list[str]]:
    """Capture WARNING-and-above records emitted on the ``apscheduler.scheduler`` logger.

    Same reason as ``apscheduler_records`` in ``test_scheduler_wakeup_guard``:
    the scheduler log bridge disables propagation, so a handler is attached
    directly to the emitting logger. WARNING level also catches the ERROR
    records produced by ``logger.exception`` calls in the guard.
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    logger = logging.getLogger(APSCHEDULER_LOGGER)
    handler = _Collector(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _make_running_guarded_scheduler() -> GuardedAsyncIOScheduler:
    """Build a guarded scheduler with a mocked event loop in the running state."""
    scheduler = GuardedAsyncIOScheduler()
    scheduler._eventloop = MagicMock()
    scheduler.state = STATE_RUNNING
    return scheduler


class TestWakeupRearmGuard:
    """The wakeup cycle must never end with no armed timer while jobs exist."""

    async def test_rearms_from_nearest_next_run_time_when_process_jobs_returns_none(
        self, apscheduler_warnings: list[str]
    ) -> None:
        """_process_jobs() returning None with a live job re-arms the timer from the job's next_run_time."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30))
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        scheduler._eventloop.call_later.assert_called_once()
        wait_seconds, callback = scheduler._eventloop.call_later.call_args.args
        assert callback == scheduler.wakeup
        assert 0 < wait_seconds <= 30
        captured = "\n".join(apscheduler_warnings)
        assert "re-arming the wakeup timer" in captured
        assert "1 jobs exist" in captured

    async def test_rearm_is_bounded_by_max_delay(self, apscheduler_warnings: list[str]) -> None:
        """A far-future next_run_time cannot postpone the re-arm beyond MAX_GUARD_REARM_DELAY_SECONDS."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(next_run_time=datetime.now(timezone.utc) + timedelta(minutes=30))
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        wait_seconds, _ = scheduler._eventloop.call_later.call_args.args
        assert wait_seconds == GuardedAsyncIOScheduler.MAX_GUARD_REARM_DELAY_SECONDS

    async def test_overdue_next_run_time_rearms_with_min_delay(self) -> None:
        """An already-due job re-arms the wakeup timer with the minimum re-arm delay, not zero."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(next_run_time=datetime.now(timezone.utc) - timedelta(seconds=5))
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        wait_seconds, _ = scheduler._eventloop.call_later.call_args.args
        assert wait_seconds == GuardedAsyncIOScheduler.MIN_GUARD_REARM_DELAY_SECONDS

    async def test_desync_state_does_not_busy_loop(self) -> None:
        """The re-arm floor bounds the wakeup cycle rate in the overdue desync state.

        Regression for the review finding: with the old lower clamp of 0.0 an
        overdue job in the desync state (store reports nothing) re-armed the
        timer with a 0 delay, spinning the whole wakeup cycle thousands of
        times per second on the event loop. With the floor the cycle rate is
        bounded to roughly one per second.
        """
        loop = asyncio.get_running_loop()
        scheduler = GuardedAsyncIOScheduler()
        scheduler._eventloop = loop
        scheduler.state = STATE_RUNNING
        job = SimpleNamespace(next_run_time=datetime.now(timezone.utc) - timedelta(seconds=5))
        scheduler.get_jobs = MagicMock(return_value=[job])
        process_jobs_invocations: list[int] = []

        def counting_process_jobs() -> None:
            process_jobs_invocations.append(1)
            return None

        scheduler._process_jobs = counting_process_jobs

        scheduler.wakeup()
        await asyncio.sleep(1.1)

        assert len(process_jobs_invocations) <= 4

    async def test_guard_survives_incomparable_next_run_time(self, apscheduler_warnings: list[str]) -> None:
        """A job whose next_run_time is not a comparable datetime cannot leave the timer disarmed.

        The guard's own arithmetic (``min(next_run_times) - now``) may raise
        on a per-job value anomaly; the guard must still produce an armed
        timer by falling back to ``MAX_GUARD_REARM_DELAY_SECONDS``.
        """
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(next_run_time=object())
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        wait_seconds, _ = scheduler._eventloop.call_later.call_args.args
        assert wait_seconds == GuardedAsyncIOScheduler.MAX_GUARD_REARM_DELAY_SECONDS
        assert "Error computing the guarded re-arm delay" in "\n".join(apscheduler_warnings)

    async def test_warning_is_throttled_to_once_per_rearm_interval(self, apscheduler_warnings: list[str]) -> None:
        """Repeated guard re-arms within one re-arm interval log the WARNING only once."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(next_run_time=datetime.now(timezone.utc) - timedelta(seconds=5))
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._process_jobs = MagicMock(return_value=None)

        for _ in range(3):
            GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)
            scheduler._eventloop.call_later.reset_mock()

        assert len(apscheduler_warnings) == 1

        scheduler._guard_warning_suppress_until = 0.0
        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        assert len(apscheduler_warnings) == 2

    async def test_zero_jobs_arms_self_check_watchdog(self) -> None:
        """With no jobs the wakeup timer stays disarmed but the periodic self-check watchdog is armed."""
        scheduler = _make_running_guarded_scheduler()
        scheduler.get_jobs = MagicMock(return_value=[])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        scheduler._eventloop.call_later.assert_called_once_with(
            GuardedAsyncIOScheduler.WATCHDOG_INTERVAL_SECONDS, scheduler._watchdog_tick
        )
        assert scheduler._watchdog_handle is scheduler._eventloop.call_later.return_value

    async def test_paused_scheduler_is_not_rearmed(self) -> None:
        """A paused scheduler legitimately computes no wait_seconds and must not trigger the guard."""
        scheduler = GuardedAsyncIOScheduler()
        scheduler._eventloop = MagicMock()
        scheduler.state = STATE_PAUSED
        scheduler.get_jobs = MagicMock(return_value=[SimpleNamespace(next_run_time=None)])
        scheduler._process_jobs = MagicMock(return_value=None)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        scheduler._eventloop.call_later.assert_not_called()
        scheduler.get_jobs.assert_not_called()


class TestWatchdog:
    """The periodic self-check detects stalls and always re-arms itself."""

    async def test_recovers_disarmed_timer_with_due_job(self, apscheduler_warnings: list[str]) -> None:
        """A disarmed wakeup timer with an overdue job is detected and a wakeup is forced."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(
            id="agent:134604548:reminder",
            next_run_time=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._timeout = None

        scheduler._watchdog_tick()

        scheduler._eventloop.call_soon_threadsafe.assert_called_once()
        captured = "\n".join(apscheduler_warnings)
        assert "Scheduler watchdog" in captured
        assert "disarmed" in captured

    async def test_recovers_overdue_job_with_armed_timer(self, apscheduler_warnings: list[str]) -> None:
        """An overdue job while the timer claims to be armed also forces a wakeup."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(
            id="agent:134604548:reminder",
            next_run_time=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._timeout = MagicMock()

        scheduler._watchdog_tick()

        scheduler._eventloop.call_soon_threadsafe.assert_called_once()
        captured = "\n".join(apscheduler_warnings)
        assert "Scheduler watchdog" in captured
        assert "overdue" in captured

    async def test_does_nothing_when_healthy(self) -> None:
        """Armed timer and no overdue jobs produce no wakeup and no warning."""
        scheduler = _make_running_guarded_scheduler()
        job = SimpleNamespace(
            id="agent:134604548:reminder",
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
        scheduler.get_jobs = MagicMock(return_value=[job])
        scheduler._timeout = MagicMock()

        scheduler._watchdog_tick()

        scheduler._eventloop.call_soon_threadsafe.assert_not_called()

    async def test_rearms_itself_even_when_job_lookup_fails(self) -> None:
        """The watchdog re-arm happens before the check, so a failing jobstore lookup cannot disarm it."""
        scheduler = _make_running_guarded_scheduler()
        scheduler.get_jobs = MagicMock(side_effect=RuntimeError("Simulated Redis outage"))

        scheduler._watchdog_tick()

        scheduler._eventloop.call_later.assert_called_once_with(
            GuardedAsyncIOScheduler.WATCHDOG_INTERVAL_SECONDS, scheduler._watchdog_tick
        )
        scheduler._eventloop.call_soon_threadsafe.assert_not_called()

    async def test_stops_when_scheduler_is_stopped(self) -> None:
        """A stopped scheduler does not re-arm the watchdog and does not check jobs."""
        scheduler = GuardedAsyncIOScheduler()
        scheduler._eventloop = MagicMock()
        scheduler.state = STATE_STOPPED
        scheduler.get_jobs = MagicMock(return_value=[SimpleNamespace(id="agent:1:x", next_run_time=None)])
        scheduler._timeout = None

        scheduler._watchdog_tick()

        scheduler._eventloop.call_later.assert_not_called()
        scheduler.get_jobs.assert_not_called()

    async def test_start_arms_watchdog_and_shutdown_stops_it(self) -> None:
        """The watchdog is armed on start and cancelled on shutdown."""
        loop = asyncio.get_running_loop()
        scheduler = GuardedAsyncIOScheduler()
        scheduler._eventloop = loop

        scheduler.start()
        assert scheduler._watchdog_handle is not None
        try:
            scheduler.shutdown(wait=False)
            assert scheduler._watchdog_handle is None
        finally:
            scheduler._stop_watchdog()


# Module-level sink: job kwargs are pickled through the persistent job store, so a
# per-test list passed as a kwarg would be deserialized into a *different* object.
_stall_probe_sink: list[datetime] = []


async def _stall_probe_job(**kwargs) -> None:
    """Record that the job actually fired after the recovery (serializable textual reference)."""
    _stall_probe_sink.append(datetime.now())


class TestSilentStallRecovery:
    """End-to-end recovery through the real scheduler wrappers and job stores."""

    @pytest.mark.asyncio
    async def test_recovers_when_jobstore_reports_no_next_run_time(
        self, tmp_path, apscheduler_warnings: list[str]
    ) -> None:
        """A jobstore whose get_next_run_time() returns None with live jobs must not silence the scheduler.

        This is the exact production signature: the wakeup cycle computes no
        next wakeup (so upstream arms NO timer), while jobs exist in the
        store. The guard must re-arm from the job's next_run_time and keep the
        job firing.
        """
        SingletonMeta._instances.pop(ChibiScheduler, None)
        try:
            with patch("chibi.services.scheduler.application_settings", _make_settings(tmp_path)):
                scheduler = ChibiScheduler()
                _stall_probe_sink.clear()

                scheduler.add_job(
                    _stall_probe_job,
                    trigger="interval",
                    seconds=0.3,
                    id="system:stall_probe",
                    replace_existing=True,
                )
                scheduler.start()
                inner = scheduler._scheduler
                jobstore = inner._jobstores["default"]
                try:
                    jobstore.get_next_run_time = lambda: None

                    await _wait_until(lambda: len(_stall_probe_sink) >= 2, timeout=10)

                    captured = "\n".join(apscheduler_warnings)
                    assert "re-arming the wakeup timer" in captured
                    assert inner._timeout is not None
                finally:
                    scheduler.shutdown(wait=False)

                assert len(_stall_probe_sink) >= 2
        finally:
            SingletonMeta._instances.pop(ChibiScheduler, None)

    @pytest.mark.asyncio
    async def test_watchdog_recovers_disarmed_timer(self, tmp_path, apscheduler_warnings: list[str]) -> None:
        """A permanently disarmed wakeup timer is resurrected by the watchdog."""
        SingletonMeta._instances.pop(StdioScheduler, None)
        try:
            with patch("chibi.services.scheduler.application_settings", _make_settings(tmp_path)):
                scheduler = StdioScheduler()
                _stall_probe_sink.clear()

                scheduler.add_job(
                    _stall_probe_job,
                    trigger="interval",
                    seconds=0.3,
                    id="system:stall_probe",
                    replace_existing=True,
                )
                scheduler.start()
                inner = scheduler._scheduler
                # setattr (not attribute assignment): the constant is declared
                # as an int class attribute, and mypy rejects a float override.
                setattr(inner, "WATCHDOG_INTERVAL_SECONDS", 0.2)
                inner._stop_watchdog()
                inner._ensure_watchdog()
                try:
                    await _wait_until(lambda: len(_stall_probe_sink) >= 1, timeout=5)
                    inner._stop_timer()

                    await _wait_until(lambda: len(_stall_probe_sink) >= 2, timeout=10)

                    captured = "\n".join(apscheduler_warnings)
                    assert "Scheduler watchdog" in captured
                    assert "disarmed" in captured
                    assert inner._timeout is not None
                finally:
                    scheduler.shutdown(wait=False)

                assert len(_stall_probe_sink) >= 2
        finally:
            SingletonMeta._instances.pop(StdioScheduler, None)
