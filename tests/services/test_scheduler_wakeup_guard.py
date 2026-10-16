"""Tests for the local APScheduler wakeup guard (upstream bug #1132).

Upstream bug (agronholm/apscheduler#1132, fixed on the 3.x branch but
unreleased as of 2026-10-16): ``AsyncIOScheduler.wakeup()`` disarms the wakeup
timer, then calls ``_process_jobs()`` and only afterwards re-arms the timer —
with no try/except. An exception escaping ``_process_jobs()`` (e.g. a
transient Redis job store error) therefore leaves the scheduler disarmed: it
looks alive but never runs another job until an unrelated external
``wakeup()`` arrives.

These tests mirror the upstream regression test
(``test_wakeup_rearms_timer_on_error``) and additionally verify the recovery
through the real ``ChibiScheduler`` / ``StdioScheduler`` wrappers with a real
event loop and a jobstore that fails mid-wakeup.

TODO(apscheduler-3.11.4): remove guard — delete this module together with
``GuardedAsyncIOScheduler`` once APScheduler 3.11.4 is adopted.
"""

import asyncio
import logging
import time
from collections.abc import Iterator
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from chibi.services.scheduler import ChibiScheduler, GuardedAsyncIOScheduler, StdioScheduler
from chibi.utils.app import SingletonMeta

APSCHEDULER_LOGGER = "apscheduler.scheduler"


@pytest.fixture
def apscheduler_records() -> Iterator[list[str]]:
    """Capture formatted records emitted on the ``apscheduler.scheduler`` logger.

    The scheduler log bridge (``chibi.config.logging``) disables propagation on
    the ``apscheduler`` logger, so pytest's ``caplog`` — which listens at the
    root — sees nothing once the bridge is installed. This fixture attaches a
    collecting handler directly to the emitting logger instead, which works
    both with and without the bridge.
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    logger = logging.getLogger(APSCHEDULER_LOGGER)
    handler = _Collector(level=logging.ERROR)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _make_settings(tmp_path) -> SimpleNamespace:
    """Build minimal scheduler settings backed by a temporary SQLite job store."""
    return SimpleNamespace(redis=None, local_data_path=str(tmp_path), scheduler_misfire_grace_time=3600)


async def _wait_until(predicate, timeout: float = 10.0, poll_interval: float = 0.05) -> None:
    """Poll a predicate until it becomes True or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(poll_interval)
    raise AssertionError("Condition was not met within the timeout")


class TestWakeupGuardSemantics:
    """Direct tests of the guard semantics, mirroring the upstream regression test."""

    def _make_guarded_scheduler(self) -> GuardedAsyncIOScheduler:
        scheduler = GuardedAsyncIOScheduler()
        scheduler._eventloop = MagicMock()
        return scheduler

    async def test_wakeup_rearms_timer_on_process_jobs_error(self, apscheduler_records: list[str]) -> None:
        """An exception in _process_jobs() is logged and the timer is re-armed with jobstore_retry_interval."""
        scheduler = self._make_guarded_scheduler()
        scheduler._process_jobs = MagicMock(side_effect=RuntimeError("Boom"))

        # Bypass the run_in_event_loop decorator to run the wakeup body synchronously.
        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        scheduler._eventloop.call_later.assert_called_once_with(scheduler.jobstore_retry_interval, scheduler.wakeup)
        captured = "\n".join(apscheduler_records)
        assert "Error processing jobs" in captured
        assert "RuntimeError: Boom" in captured

    async def test_wakeup_still_rearms_on_success(self) -> None:
        """A successful wakeup keeps the normal re-arm behaviour with the computed wait_seconds."""
        scheduler = self._make_guarded_scheduler()
        scheduler._process_jobs = MagicMock(return_value=42)

        GuardedAsyncIOScheduler.wakeup.__wrapped__(scheduler)

        scheduler._eventloop.call_later.assert_called_once_with(42, scheduler.wakeup)

    async def test_is_subclass_of_asyncio_scheduler(self) -> None:
        """The guard is a drop-in AsyncIOScheduler subclass (trivially removable)."""
        assert issubclass(GuardedAsyncIOScheduler, AsyncIOScheduler)


class TestWakeupGuardWiring:
    """Both scheduler flavors must use the guarded implementation."""

    def test_chibi_scheduler_uses_guard(self, tmp_path) -> None:
        SingletonMeta._instances.pop(ChibiScheduler, None)
        try:
            with patch("chibi.services.scheduler.application_settings", _make_settings(tmp_path)):
                assert isinstance(ChibiScheduler()._scheduler, GuardedAsyncIOScheduler)
        finally:
            SingletonMeta._instances.pop(ChibiScheduler, None)

    def test_stdio_scheduler_uses_guard(self, tmp_path) -> None:
        SingletonMeta._instances.pop(StdioScheduler, None)
        try:
            with patch("chibi.services.scheduler.application_settings", _make_settings(tmp_path)):
                assert isinstance(StdioScheduler()._scheduler, GuardedAsyncIOScheduler)
        finally:
            SingletonMeta._instances.pop(StdioScheduler, None)


# Module-level sink: job kwargs are pickled through the persistent job store, so a
# per-test list passed as a kwarg would be deserialized into a *different* object.
_probe_sink: list[datetime] = []


async def _guard_probe_job(**kwargs) -> None:
    """Record that the job actually fired after the recovery (serializable textual reference)."""
    _probe_sink.append(datetime.now())


class TestWakeupGuardRecovery:
    """End-to-end recovery: a jobstore failure mid-wakeup must not silence the scheduler.

    Note: in APScheduler 3.11.x, ``jobstore.get_due_jobs()`` failures are already
    caught inside ``_process_jobs()``; the unguarded jobstore calls that escape
    to ``wakeup()`` (and thus hit the guard) are ``jobstore.update_job()`` and
    ``jobstore.get_next_run_time()``. The transient failure is therefore
    simulated on ``update_job`` — the exact path described by upstream #1132
    and by the Chibi incident (a Redis hiccup in ``jobstore.update_job()``).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scheduler_cls", [ChibiScheduler, StdioScheduler])
    async def test_scheduler_recovers_after_jobstore_failure(
        self, scheduler_cls: type, tmp_path, apscheduler_records: list[str]
    ) -> None:
        SingletonMeta._instances.pop(scheduler_cls, None)
        try:
            with patch("chibi.services.scheduler.application_settings", _make_settings(tmp_path)):
                scheduler = scheduler_cls()
                _probe_sink.clear()

                # A fast recurring job so several wakeup cycles occur within the test.
                scheduler.add_job(
                    _guard_probe_job,
                    trigger="interval",
                    seconds=0.3,
                    id="system:guard_probe",
                    replace_existing=True,
                )
                scheduler.start()
                inner = scheduler._scheduler
                # Speed up the guard's failure retry loop for the test.
                inner.jobstore_retry_interval = 0.2
                jobstore = inner._jobstores["default"]
                real_update_job = jobstore.update_job
                boom = MagicMock(side_effect=RuntimeError("Simulated Redis hiccup"))
                try:
                    jobstore.update_job = boom
                    # Wait for the wakeup cycle that hits the jobstore failure.
                    await _wait_until(lambda: boom.call_count > 0, timeout=5)

                    # The guard must have logged the escaped exception...
                    captured = "\n".join(apscheduler_records)
                    assert "Error processing jobs" in captured
                    assert "Simulated Redis hiccup" in captured
                    # ...and re-armed the wakeup timer instead of leaving it disarmed.
                    assert getattr(inner, "_timeout", None) is not None

                    # Restore the job store and let the re-armed timer fire again.
                    jobstore.update_job = real_update_job
                    await _wait_until(lambda: len(_probe_sink) >= 2, timeout=10)
                    assert scheduler.state != 0  # scheduler is still running
                finally:
                    scheduler.shutdown(wait=False)

                # The job fired both in the failing wakeup window and afterwards.
                assert len(_probe_sink) >= 2
        finally:
            SingletonMeta._instances.pop(scheduler_cls, None)
