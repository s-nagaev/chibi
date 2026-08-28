import sys
from typing import Any

from loguru import logger

config: dict[Any, Any] = {
    "handlers": [
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": "<lvl>{level}</lvl>\t| <green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> |  <lvl>{message}</lvl>",
        },
    ],
}
logger.configure(**config)


def use_stderr_logging() -> None:
    """Route loguru output to stderr and drop the stdout sink.

    Used by the IDE stdio runner: stdout is the JSONL protocol channel and
    must carry protocol frames only, so log lines would corrupt the protocol.
    Terminal and Telegram modes are unaffected — they never call this and
    keep configuring their own sinks.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
