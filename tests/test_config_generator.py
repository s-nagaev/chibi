"""Tests for config generator."""

import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from chibi import config_generator


class TestGenerateDefaultConfig:
    """Tests for generate_default_config."""

    @pytest.fixture
    def temp_dirs(self, tmp_path: Path) -> dict[str, Path]:
        """Return patched paths for config generator."""
        return {
            "chibi_bot_dir": tmp_path / "chibi-bot",
            "data_dir": tmp_path / "chibi-bot" / "data",
            "skills_dir": tmp_path / "chibi-bot" / "skills",
            "home_dir": tmp_path / "chibi-bot" / "home",
            "config_path": tmp_path / "chibi-bot" / "settings",
        }

    @patch("chibi.config_generator.print")
    def test_writes_utf8_file(self, mock_print, temp_dirs, monkeypatch):
        """Generated config file must be written with encoding='utf-8'."""
        original_open = builtins.open
        captured_encoding: str | None = None

        def spy_open(*args, **kwargs):
            nonlocal captured_encoding
            if args and args[0] == config_generator.CONFIG_PATH:
                positional_encoding = args[3] if len(args) > 3 else None
                captured_encoding = kwargs.get("encoding", positional_encoding)
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)

        dirs = temp_dirs
        with patch.object(config_generator, "CHIBI_BOT_DIR", dirs["chibi_bot_dir"]):
            with patch.object(config_generator, "DATA_DIR", dirs["data_dir"]):
                with patch.object(config_generator, "SKILLS_DIR", dirs["skills_dir"]):
                    with patch.object(config_generator, "HOME_DIR", dirs["home_dir"]):
                        with patch.object(config_generator, "CONFIG_PATH", dirs["config_path"]):
                            config_generator.generate_default_config()

        assert dirs["config_path"].exists()
        assert captured_encoding == "utf-8"

    @patch("chibi.config_generator.print")
    def test_does_not_overwrite_existing(self, mock_print, temp_dirs):
        """Existing config file should not be overwritten."""
        dirs = temp_dirs
        dirs["config_path"].parent.mkdir(parents=True, exist_ok=True)
        dirs["config_path"].write_text("existing", encoding="utf-8")

        with patch.object(config_generator, "CHIBI_BOT_DIR", dirs["chibi_bot_dir"]):
            with patch.object(config_generator, "DATA_DIR", dirs["data_dir"]):
                with patch.object(config_generator, "SKILLS_DIR", dirs["skills_dir"]):
                    with patch.object(config_generator, "HOME_DIR", dirs["home_dir"]):
                        with patch.object(config_generator, "CONFIG_PATH", dirs["config_path"]):
                            config_generator.generate_default_config()

        assert dirs["config_path"].read_text(encoding="utf-8") == "existing"
