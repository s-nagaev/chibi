"""Tests for the Telegram runner startup behavior."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from chibi.config import application_settings
from chibi.runners.telegram import run_chibi


@pytest.fixture
def restore_client_setting() -> Iterator[None]:
    """Restore the original client value after the test."""
    original = application_settings.client
    yield
    application_settings.client = original


class TestTelegramRunnerClient:
    """Tests for the client value set by the Telegram runner."""

    def test_run_chibi_sets_client_to_telegram(self, restore_client_setting: Iterator[None]) -> None:
        """`run_chibi` marks the process as serving the telegram client."""
        with (
            patch("chibi.runners.telegram.ChibiBot") as bot_cls,
            patch("chibi.runners.telegram.log_application_settings"),
            patch("chibi.runners.telegram.telegram_setting_pre_start_check"),
            patch("chibi.runners.telegram.telegram_security_pre_start_check"),
        ):
            application_settings.client = "tui"

            run_chibi()

        assert application_settings.client == "telegram"
        bot_cls.return_value.run.assert_called_once()
