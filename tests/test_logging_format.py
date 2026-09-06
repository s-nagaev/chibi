"""Tests for the backend log line format across user-facing sinks."""

import re
import sys
from collections.abc import Iterator

import pytest
from loguru import logger

from chibi.config.app import configure_stderr_sink
from chibi.config.logging import use_stderr_logging
from chibi.runners.terminal import setup_logging

LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+ \| ")
ANSI_CODES = re.compile(r"\x1b\[[0-9;]*m")
FILE_LINE_NOISE = re.compile(r"\.py:\d+")


def assert_plain_log_line(line: str) -> None:
    """Assert one log line matches ``YYYY-MM-DD HH:MM:SS | LEVEL | message`` with no file/line noise."""
    plain = ANSI_CODES.sub("", line)
    assert LOG_LINE.match(plain), plain
    assert FILE_LINE_NOISE.search(plain) is None, plain


@pytest.fixture()
def restore_stderr_logging() -> Iterator[None]:
    """Restore a plain stderr loguru sink after a test reconfigured handlers."""
    yield
    logger.remove()
    logger.add(sys.stderr, level="INFO")


@pytest.mark.usefixtures("restore_stderr_logging")
def test_stderr_logging_shape(capsys: pytest.CaptureFixture[str]) -> None:
    """use_stderr_logging() emits plain ``datetime | LEVEL | message`` lines to stderr."""
    use_stderr_logging()
    logger.info("hello world")
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert lines, err
    for line in lines:
        assert_plain_log_line(line)
    assert any(line.endswith("hello world") for line in lines), err


@pytest.mark.usefixtures("restore_stderr_logging")
def test_backend_stderr_sink_shape(capfd: pytest.CaptureFixture[str]) -> None:
    """The backend sink installed at import emits plain lines and preserves message and user context."""
    logger.remove()
    configure_stderr_sink()
    logger.info("backend hello")
    logger.bind(user_id=42).info("scoped hello")
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert lines, err
    for line in lines:
        assert_plain_log_line(line)
    assert any("backend hello" in line for line in lines), err
    assert any("[42] scoped hello" in line for line in lines), err


@pytest.mark.usefixtures("restore_stderr_logging")
def test_terminal_runner_sink_shape(capfd: pytest.CaptureFixture[str]) -> None:
    """setup_logging() (terminal runner) emits plain ``datetime | LEVEL | message`` lines to stderr."""
    setup_logging()
    logger.warning("terminal hello")
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert lines, err
    for line in lines:
        assert_plain_log_line(line)
    assert any(line.endswith("terminal hello") for line in lines), err
