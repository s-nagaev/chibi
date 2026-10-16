"""Tests for traceback diagnostics handling across Chibi's loguru sinks."""

import re
import sys
from collections.abc import Callable, Iterator
from typing import NoReturn

import pytest
from loguru import logger

from chibi.config.app import configure_stderr_sink
from chibi.config.logging import configure_stdout_sink, traceback_diagnostics_enabled, use_stderr_logging
from chibi.runners.terminal import setup_logging

CANARY = "sk-canary-credential-value"
ANSI_CODES = re.compile(r"\x1b\[[0-9;]*m")
SINK_SETUPS: list[Callable[[], None]] = [
    configure_stdout_sink,
    use_stderr_logging,
    configure_stderr_sink,
    setup_logging,
]


@pytest.fixture()
def restore_default_sink() -> Iterator[None]:
    """Restore a plain stderr loguru sink after a test reconfigured handlers."""
    yield
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def _raise_with_canary_local(value: str) -> NoReturn:
    """Raise an exception whose frame locals contain the canary value.

    Args:
        value: Secret-like string stored in a local variable of this frame.
    """
    canary_local = f"Bearer {value}"
    assert canary_local
    raise RuntimeError("boom")


def _log_exception_with_canary() -> None:
    """Log an exception whose traceback frames hold a canary local."""
    try:
        _raise_with_canary_local(CANARY)
    except RuntimeError as exc:
        logger.opt(exception=exc).error("canary probe")


def _captured_output(capfd: pytest.CaptureFixture[str]) -> str:
    """Return the ANSI-stripped stdout+stderr captured since the last read.

    Args:
        capfd: The pytest capture fixture to read from.

    Returns:
        Combined, ANSI-stripped captured output.
    """
    captured = capfd.readouterr()
    return ANSI_CODES.sub("", captured.out + captured.err)


@pytest.mark.parametrize(
    "value",
    ["1", "true", "yes", "y", "ok", "on", "TRUE", "Yes", "ON", "Ok", "yEs"],
)
def test_loguru_truthy_spellings_enable_diagnostics(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Every loguru truthy spelling of LOGURU_DIAGNOSE enables rendering, in any case."""
    monkeypatch.setenv("LOGURU_DIAGNOSE", value)
    assert traceback_diagnostics_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "n", "nok", "off", "", "maybe", "enabled", " True "])
def test_other_values_disable_diagnostics(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Values outside loguru's truthy set disable rendering."""
    monkeypatch.setenv("LOGURU_DIAGNOSE", value)
    assert traceback_diagnostics_enabled() is False


def test_unset_variable_disables_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset LOGURU_DIAGNOSE disables rendering (loguru's own default is True)."""
    monkeypatch.delenv("LOGURU_DIAGNOSE", raising=False)
    assert traceback_diagnostics_enabled() is False


@pytest.mark.usefixtures("restore_default_sink")
def test_sink_configuration_reads_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Re-configuring a sink after LOGURU_DIAGNOSE changes actually changes the behaviour."""
    monkeypatch.delenv("LOGURU_DIAGNOSE", raising=False)
    use_stderr_logging()
    _log_exception_with_canary()
    assert CANARY not in _captured_output(capfd)

    monkeypatch.setenv("LOGURU_DIAGNOSE", "1")
    use_stderr_logging()
    _log_exception_with_canary()
    assert CANARY in _captured_output(capfd)


@pytest.mark.parametrize("setup_sink", SINK_SETUPS, ids=[fn.__name__ for fn in SINK_SETUPS])
@pytest.mark.usefixtures("restore_default_sink")
def test_sinks_hide_locals_by_default(setup_sink: Callable[[], None], capfd: pytest.CaptureFixture[str]) -> None:
    """With diagnostics disabled no sink renders canary locals, but the traceback is still logged."""
    setup_sink()
    _log_exception_with_canary()
    output = _captured_output(capfd)
    assert CANARY not in output
    assert "RuntimeError: boom" in output


@pytest.mark.parametrize("setup_sink", SINK_SETUPS, ids=[fn.__name__ for fn in SINK_SETUPS])
@pytest.mark.usefixtures("restore_default_sink")
def test_sinks_render_locals_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
    setup_sink: Callable[[], None],
    capfd: pytest.CaptureFixture[str],
) -> None:
    """With LOGURU_DIAGNOSE enabled the canary local is rendered (the hiding test is not vacuous)."""
    monkeypatch.setenv("LOGURU_DIAGNOSE", "1")
    setup_sink()
    _log_exception_with_canary()
    assert CANARY in _captured_output(capfd)
