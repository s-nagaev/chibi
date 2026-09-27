"""Tests for ApplicationSettings."""

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from chibi.config.app import ApplicationSettings, ClientType


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove ambient CLIENT overrides so defaults are observable."""
    monkeypatch.delenv("CLIENT", raising=False)
    yield monkeypatch


class TestClientSetting:
    """Tests for the ApplicationSettings.client field."""

    @pytest.mark.usefixtures("clean_environment")
    def test_client_defaults_to_telegram(self) -> None:
        """Without any override the process is assumed to serve the telegram client."""
        assert ApplicationSettings().client == "telegram"

    @pytest.mark.parametrize("client", ["telegram", "tui", "vscode", "pycharm", "neovim"])
    def test_client_accepts_every_declared_value(self, client: ClientType) -> None:
        """Every ClientType literal is a valid value for the field."""
        assert ApplicationSettings(client=client).client == client

    @pytest.mark.parametrize("client", ["ide", "slack", "TUI", ""])
    def test_client_rejects_unknown_values(self, client: str) -> None:
        """Values outside ClientType fail validation."""
        with pytest.raises(ValidationError):
            ApplicationSettings(client=client)

    @pytest.mark.usefixtures("clean_environment")
    def test_client_is_readable_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The field is bound to the CLIENT environment variable like other settings."""
        monkeypatch.setenv("CLIENT", "tui")

        assert ApplicationSettings().client == "tui"


class TestSchedulerSettings:
    """Tests for the scheduler-related ApplicationSettings fields."""

    def test_scheduler_defaults(self) -> None:
        """Scheduler fields expose the documented defaults."""
        settings = ApplicationSettings()
        assert settings.scheduler_tool_enabled is True
        assert settings.scheduler_notify_enabled is False
        assert settings.scheduler_agent_commands_enabled is False
        assert settings.scheduler_command_timeout_max == 900
        assert settings.scheduler_misfire_grace_time == 3600
        assert settings.scheduler_failure_notify is True

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("SCHEDULER_TOOL_ENABLED", "false"),
            ("SCHEDULER_NOTIFY_ENABLED", "true"),
            ("SCHEDULER_AGENT_COMMANDS_ENABLED", "true"),
            ("SCHEDULER_FAILURE_NOTIFY", "false"),
        ],
    )
    def test_scheduler_bool_fields_are_env_driven(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, expected: str
    ) -> None:
        """Boolean scheduler fields are bound to their environment variables."""
        monkeypatch.setenv(env_var, expected)
        settings = ApplicationSettings()
        field_name = env_var.lower()
        assert getattr(settings, field_name) == (expected == "true")

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("SCHEDULER_COMMAND_TIMEOUT_MAX", 1200),
            ("SCHEDULER_MISFIRE_GRACE_TIME", 60),
        ],
    )
    def test_scheduler_int_fields_are_env_driven(
        self, monkeypatch: pytest.MonkeyPatch, env_var: str, expected: int
    ) -> None:
        """Integer scheduler fields are bound to their environment variables."""
        monkeypatch.setenv(env_var, str(expected))
        settings = ApplicationSettings()
        assert getattr(settings, env_var.lower()) == expected
