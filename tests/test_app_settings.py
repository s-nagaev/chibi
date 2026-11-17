"""Tests for ApplicationSettings."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from chibi.config.app import ApplicationSettings, ClientType, _get_application_settings


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
        assert settings.scheduler_agent_commands_enabled is True
        assert settings.scheduler_command_timeout_max == 900
        assert settings.scheduler_misfire_grace_time == 3600
        assert settings.scheduler_failure_notify is True

    @pytest.mark.parametrize(
        ("env_var", "expected"),
        [
            ("SCHEDULER_TOOL_ENABLED", "false"),
            ("SCHEDULER_NOTIFY_ENABLED", "true"),
            ("SCHEDULER_AGENT_COMMANDS_ENABLED", "true"),
            ("SCHEDULER_AGENT_COMMANDS_ENABLED", "false"),
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


class TestLocalDataPathFallback:
    """Tests for the host fallback of the local data path."""

    @pytest.fixture(autouse=True)
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
        """Bind ``HOME`` and the working directory to ``tmp_path`` for one test.

        ``_get_application_settings`` merges ``~/chibi-bot/settings`` into
        ``os.environ``, so the process environment is snapshotted and restored
        around every test. The cached settings object is dropped on entry and on
        exit, so each test resolves the configuration from scratch.
        """
        environment = dict(os.environ)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LOCAL_DATA_PATH", raising=False)
        monkeypatch.setenv("CHIBI_ENV", "dev")
        _get_application_settings.cache_clear()
        yield
        os.environ.clear()
        os.environ.update(environment)
        _get_application_settings.cache_clear()

    def test_host_run_without_configuration_falls_back_to_home_data_dir(self, tmp_path: Path) -> None:
        """A bare host install resolves to ``~/chibi-bot/data`` and the directory is created."""
        settings = _get_application_settings()

        expected = tmp_path / "chibi-bot" / "data"
        assert settings.local_data_path == expected.as_posix()
        assert expected.is_dir()

    def test_environment_value_wins_over_the_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An explicit environment variable is never replaced by the host fallback."""
        configured = tmp_path / "explicit-data"
        monkeypatch.setenv("LOCAL_DATA_PATH", str(configured))

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
        assert not (tmp_path / "chibi-bot" / "data").exists()

    def test_working_directory_dotenv_value_wins_over_the_fallback(self, tmp_path: Path) -> None:
        """``LOCAL_DATA_PATH`` from the working directory ``.env`` disables the fallback."""
        configured = tmp_path / "dotenv-data"
        (tmp_path / ".env").write_text(f"LOCAL_DATA_PATH={configured}\n")

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
        assert not (tmp_path / "chibi-bot" / "data").exists()

    def test_settings_file_value_wins_over_the_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``LOCAL_DATA_PATH`` merged from ``~/chibi-bot/settings`` disables the fallback."""
        monkeypatch.setenv("CHIBI_ENV", "no-dev")
        configured = tmp_path / "settings-data"
        settings_dir = tmp_path / "chibi-bot"
        settings_dir.mkdir()
        (settings_dir / "settings").write_text(f"LOCAL_DATA_PATH={configured}\n")

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
        assert not (settings_dir / "data").exists()

    @pytest.mark.parametrize("runtime", ["docker", "podman"])
    def test_container_runtime_keeps_the_image_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime: str
    ) -> None:
        """Container runtimes keep the image path ``/app/data``."""
        monkeypatch.setattr(ApplicationSettings, "runtime_environment", property(lambda self: runtime))

        settings = _get_application_settings()

        assert settings.local_data_path == "/app/data"
        assert not (tmp_path / "chibi-bot" / "data").exists()
class TestSettingsFileLoading:
    """The ``~/chibi-bot/settings`` file is merged into the environment in every mode."""

    @pytest.fixture(autouse=True)
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
        """Bind ``HOME`` to ``tmp_path`` and snapshot the environment around one test.

        ``_get_application_settings`` merges the settings file into
        ``os.environ``, so the process environment is snapshotted and restored
        around every test. The cached settings object is dropped on entry and
        on exit, so each test resolves the configuration from scratch.
        """
        environment = dict(os.environ)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("LOCAL_DATA_PATH", raising=False)
        monkeypatch.delenv("CHIBI_SETTINGS_FILE", raising=False)
        monkeypatch.setenv("CHIBI_ENV", "dev")
        _get_application_settings.cache_clear()
        yield
        os.environ.clear()
        os.environ.update(environment)
        _get_application_settings.cache_clear()

    def test_dev_mode_reads_the_settings_file(self, tmp_path: Path) -> None:
        """In dev mode the settings file exists and its values land in the environment."""
        settings_dir = tmp_path / "chibi-bot"
        settings_dir.mkdir()
        configured = tmp_path / "dev-mode-data"
        (settings_dir / "settings").write_text(f"LOCAL_DATA_PATH={configured}\n")

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
        assert os.environ["LOCAL_DATA_PATH"] == str(configured)

    def test_chibi_settings_file_override_points_at_a_custom_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``CHIBI_SETTINGS_FILE`` redirects the settings file away from the default path."""
        custom = tmp_path / "custom-settings"
        configured = tmp_path / "custom-data"
        custom.write_text(f"LOCAL_DATA_PATH={configured}\n")
        monkeypatch.setenv("CHIBI_SETTINGS_FILE", str(custom))

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
        assert not (tmp_path / "chibi-bot" / "settings").exists()

    def test_process_environment_values_absent_from_the_file_survive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Variables set in the process environment but missing from the file are kept."""
        settings_dir = tmp_path / "chibi-bot"
        settings_dir.mkdir()
        (settings_dir / "settings").write_text("SCHEDULER_TOOL_ENABLED=false\n")
        monkeypatch.setenv("SCHEDULER_FAILURE_NOTIFY", "false")

        settings = _get_application_settings()

        assert os.environ["SCHEDULER_FAILURE_NOTIFY"] == "false"
        assert settings.scheduler_failure_notify is False
        assert settings.scheduler_tool_enabled is False

    def test_dev_mode_process_environment_wins_over_the_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """In dev mode a variable set in the process environment beats the file value."""
        settings_dir = tmp_path / "chibi-bot"
        settings_dir.mkdir()
        (settings_dir / "settings").write_text(f"LOCAL_DATA_PATH={tmp_path / 'file-data'}\n")
        monkeypatch.setenv("LOCAL_DATA_PATH", str(tmp_path / "env-data"))

        settings = _get_application_settings()

        assert settings.local_data_path == str(tmp_path / "env-data")
        assert os.environ["LOCAL_DATA_PATH"] == str(tmp_path / "env-data")

    def test_no_dev_mode_keeps_the_historical_file_precedence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Outside dev a variable present in both the file and the environment resolves to the file value."""
        monkeypatch.setenv("CHIBI_ENV", "no-dev")
        settings_dir = tmp_path / "chibi-bot"
        settings_dir.mkdir()
        configured = tmp_path / "file-data"
        (settings_dir / "settings").write_text(f"LOCAL_DATA_PATH={configured}\n")
        monkeypatch.setenv("LOCAL_DATA_PATH", str(tmp_path / "env-data"))

        settings = _get_application_settings()

        assert settings.local_data_path == str(configured)
