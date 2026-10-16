import inspect
import logging
import sys
from typing import Any

from loguru import logger

config: dict[Any, Any] = {
    "handlers": [
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
        },
    ],
}
logger.configure(**config)

SCHEDULER_LOGGER_NAME = "apscheduler"


class StdlibToLoguruHandler(logging.Handler):
    """Forward stdlib ``logging`` records into loguru sinks.

    Records keep their stdlib level (mapped onto the matching loguru level) and
    their call site, so bridged lines are rendered by the standard loguru sinks
    with the usual ``YYYY-MM-DD HH:MM:SS | LEVEL | message`` shape. The stdlib
    logger name is embedded in the message (``[apscheduler.scheduler] ...``) so
    a plain ``grep -i scheduler`` over the logs can match records whose message
    text alone would not mention the scheduler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk up past the stdlib logging internals so loguru attributes the
        # record to the original call site instead of this handler.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, "[{}] {}", record.name, record.getMessage())


def install_scheduler_log_bridge(level: str | None = None) -> None:
    """Route APScheduler stdlib logging records into the loguru sinks.

    APScheduler 3.x emits everything through stdlib ``logging`` on the
    ``apscheduler.*`` logger family. Chibi configures only loguru, so without
    this bridge those records fall through to ``logging.lastResort`` — a bare
    stderr handler with a WARNING threshold, no timestamp and no level — and
    every INFO/DEBUG record is discarded entirely. This installs a single
    :class:`StdlibToLoguruHandler` on the ``apscheduler`` logger and disables
    propagation so records are emitted exactly once, through loguru, and never
    additionally through ``lastResort``.

    The bridge is scoped narrowly to the ``apscheduler.*`` family: loggers from
    other libraries keep behaving exactly as before.

    Args:
        level: Floor level for the ``apscheduler`` logger. ``None`` (default)
            resolves it from the ``scheduler_debug_logs`` setting: DEBUG when
            the flag is on, INFO otherwise. DEBUG surfaces per-wakeup chatter
            (``Next wakeup is due at ...``, ``Added job ...``); INFO is enough
            to see starts, scheduling and missed-run warnings.

    Idempotent: calling it again re-applies the level but never installs a
    second handler, so records are never duplicated.
    """
    if level is None:
        from chibi.config.app import application_settings

        level = "DEBUG" if application_settings.scheduler_debug_logs else "INFO"
    scheduler_logger = logging.getLogger(SCHEDULER_LOGGER_NAME)
    scheduler_logger.setLevel(level.upper())
    scheduler_logger.propagate = False
    if not any(isinstance(handler, StdlibToLoguruHandler) for handler in scheduler_logger.handlers):
        scheduler_logger.addHandler(StdlibToLoguruHandler())


def use_stderr_logging() -> None:
    """Route loguru output to stderr and drop the stdout sink.

    Used by the stdio JSONL runner: stdout is the JSONL protocol channel and
    must carry protocol frames only, so log lines would corrupt the protocol.
    Terminal and Telegram modes are unaffected — they never call this and
    keep configuring their own sinks.

    Lines follow the plain shape ``YYYY-MM-DD HH:MM:SS | LEVEL | message``
    (local time, no file/line references) so downstream consumers such as
    the TUI diagnostics view can parse the level field directly.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
