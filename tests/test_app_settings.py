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

    def test_client_defaults_to_telegram(self, clean_environment: Iterator[pytest.MonkeyPatch]) -> None:
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

    def test_client_is_readable_from_environment(
        self, clean_environment: Iterator[pytest.MonkeyPatch], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The field is bound to the CLIENT environment variable like other settings."""
        monkeypatch.setenv("CLIENT", "tui")

        assert ApplicationSettings().client == "tui"
