"""Tests for generated-media save helpers in chibi.services.providers.tools.utils."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from chibi.config import gpt_settings
from chibi.config.app import ApplicationSettings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.utils import resolve_media_destination, save_media_to_disk


@pytest.fixture
def filesystem_access_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the filesystem_access gate for the duration of a test."""
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)


@pytest.fixture
def filesystem_access_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the filesystem_access gate for the duration of a test."""
    monkeypatch.setattr(gpt_settings, "filesystem_access", False)


@pytest.fixture
def fresh_app_settings(monkeypatch: pytest.MonkeyPatch) -> type[ApplicationSettings]:
    """Return an ApplicationSettings class isolated from ambient env and .env files."""
    monkeypatch.delenv("MEDIA_OUTPUT_DIR", raising=False)
    return ApplicationSettings


class TestFilesystemAccessGate:
    """Saving is gated by gpt_settings.filesystem_access."""

    async def test_save_raises_when_filesystem_access_disabled(
        self, tmp_path: Path, filesystem_access_disabled
    ) -> None:
        with pytest.raises(PermissionError, match="FILESYSTEM_ACCESS"):
            await save_media_to_disk(
                b"data",
                tool_name="generate_image",
                prompt="a cat",
                ext="png",
                save_path=str(tmp_path / "out.png"),
            )

    async def test_no_file_written_when_gate_blocks(self, tmp_path: Path, filesystem_access_disabled) -> None:
        target = tmp_path / "out.png"
        with pytest.raises(PermissionError):
            await save_media_to_disk(
                b"data", tool_name="generate_image", prompt="a cat", ext="png", save_path=str(target)
            )
        assert not target.exists()

    async def test_default_dir_not_created_when_gate_blocks(self, tmp_path: Path, filesystem_access_disabled) -> None:
        fallback = tmp_path / "media"
        with pytest.raises(PermissionError):
            await save_media_to_disk(
                b"data", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(fallback)
            )
        assert not fallback.exists()

    async def test_save_works_when_filesystem_access_enabled(self, tmp_path: Path, filesystem_access_enabled) -> None:
        saved = await save_media_to_disk(
            b"data", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(tmp_path)
        )
        assert Path(saved).read_bytes() == b"data"


class TestExplicitFilePath:
    """An explicit file path is honoured as-is."""

    async def test_explicit_file_path_written_exactly_there(self, tmp_path: Path, filesystem_access_enabled) -> None:
        target = tmp_path / "deep" / "nested" / "art.png"
        saved = await save_media_to_disk(
            b"payload", tool_name="generate_image", prompt="a cat", ext="png", save_path=str(target)
        )
        assert Path(saved) == target.resolve()
        assert target.read_bytes() == b"payload"

    async def test_explicit_file_path_overwrites_existing_file(self, tmp_path: Path, filesystem_access_enabled) -> None:
        target = tmp_path / "art.png"
        target.write_bytes(b"old")
        await save_media_to_disk(b"new", tool_name="generate_image", prompt="a cat", ext="png", save_path=str(target))
        assert target.read_bytes() == b"new"


class TestDirectoryPath:
    """An existing directory receives a derived filename."""

    @pytest.mark.parametrize("precreate", [True, False])
    async def test_directory_path_gets_derived_file(
        self, tmp_path: Path, filesystem_access_enabled, precreate: bool
    ) -> None:
        directory = tmp_path / "images"
        if precreate:
            directory.mkdir()
        saved = await save_media_to_disk(
            b"payload", tool_name="generate_image", prompt="A Big Red Cat!!!", ext="png", save_path=str(directory)
        )
        path = Path(saved)
        assert path.parent == directory.resolve()
        assert path.name.startswith("generate_image-a-big-red-cat-")
        assert path.name.endswith(".png")
        assert path.read_bytes() == b"payload"

    async def test_path_without_suffix_and_not_existing_is_treated_as_directory(
        self, tmp_path: Path, filesystem_access_enabled
    ) -> None:
        directory = tmp_path / "newdir"
        saved = await save_media_to_disk(
            b"payload", tool_name="text_to_speech", prompt="hello", ext="mp3", save_path=str(directory)
        )
        assert Path(saved).parent == directory.resolve()
        assert directory.is_dir()


class TestFallbackDir:
    """When save_path is absent, the caller-provided default dir is used."""

    async def test_absent_path_uses_fallback_dir(self, tmp_path: Path, filesystem_access_enabled) -> None:
        fallback = tmp_path / "media"
        saved = await save_media_to_disk(
            b"payload", tool_name="generate_video", prompt="sunset", ext="mp4", fallback_dir=str(fallback)
        )
        path = Path(saved)
        assert path.parent == fallback.resolve()
        assert path.name.startswith("generate_video-sunset-")
        assert fallback.is_dir()

    async def test_absent_path_without_fallback_dir_raises(self, filesystem_access_enabled) -> None:
        with pytest.raises(ValueError, match="fallback_dir"):
            await save_media_to_disk(b"payload", tool_name="generate_image", prompt="a cat", ext="png")

    async def test_fallback_dir_creation_is_lazy(self, tmp_path: Path, filesystem_access_enabled) -> None:
        fallback = tmp_path / "media"
        assert not fallback.exists()
        await save_media_to_disk(
            b"payload", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(fallback)
        )
        assert fallback.exists()


class TestCollisionHandling:
    """Derived filenames never silently overwrite existing files."""

    async def test_collision_gets_numeric_suffix_and_original_is_untouched(
        self, tmp_path: Path, filesystem_access_enabled
    ) -> None:
        fixed_now = datetime(2026, 10, 13, 12, 0, 0)
        with patch("chibi.services.providers.tools.utils.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = fixed_now
            first = await save_media_to_disk(
                b"first", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(tmp_path)
            )
            second = await save_media_to_disk(
                b"second", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(tmp_path)
            )
            third = await save_media_to_disk(
                b"third", tool_name="generate_image", prompt="a cat", ext="png", fallback_dir=str(tmp_path)
            )

        assert first != second != third
        assert Path(first).read_bytes() == b"first"
        assert Path(second).read_bytes() == b"second"
        assert Path(third).read_bytes() == b"third"
        assert Path(second).stem == f"{Path(first).stem}-1"
        assert Path(third).stem == f"{Path(first).stem}-2"

    async def test_resolve_returns_first_free_variant(self, tmp_path: Path) -> None:
        existing = tmp_path / "generate_image-a-cat-20261013-120000.png"
        existing.write_bytes(b"taken")
        with patch("chibi.services.providers.tools.utils.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime(2026, 10, 13, 12, 0, 0)
            resolved = resolve_media_destination(None, "generate_image", "a cat", "png", fallback_dir=str(tmp_path))

        assert resolved.name == "generate_image-a-cat-20261013-120000-1.png"
        assert existing.read_bytes() == b"taken"


class TestContentSources:
    """The writer accepts raw bytes and URLs (download mocked, no network)."""

    async def test_url_input_downloads_and_saves(self, tmp_path: Path, filesystem_access_enabled) -> None:
        with patch(
            "chibi.services.providers.tools.utils.download", new=AsyncMock(return_value=b"audio-bytes")
        ) as mock_download:
            saved = await save_media_to_disk(
                "https://example.com/song.mp3",
                tool_name="generate_music_via_suno",
                prompt="a song",
                ext="mp3",
                fallback_dir=str(tmp_path),
            )
        mock_download.assert_awaited_once_with("https://example.com/song.mp3")
        assert Path(saved).read_bytes() == b"audio-bytes"

    async def test_failed_download_raises_and_creates_nothing(self, tmp_path: Path, filesystem_access_enabled) -> None:
        fallback = tmp_path / "media"
        with patch("chibi.services.providers.tools.utils.download", new=AsyncMock(return_value=None)):
            with pytest.raises(ToolException, match="example.com"):
                await save_media_to_disk(
                    "https://example.com/song.mp3",
                    tool_name="generate_music_via_suno",
                    prompt="a song",
                    ext="mp3",
                    fallback_dir=str(fallback),
                )
        assert not fallback.exists()


class TestMediaOutputDirSetting:
    """The media_output_dir setting resolves lazily with ~ and env vars expanded."""

    def test_default_is_none(self, fresh_app_settings) -> None:
        assert fresh_app_settings(_env_file=None).media_output_dir is None

    def test_resolved_default_is_working_dir_media(self, fresh_app_settings) -> None:
        settings = fresh_app_settings(_env_file=None)
        expected = Path("~/chibi/media").expanduser()
        assert settings.resolved_media_output_dir == str(expected)

    def test_media_output_dir_is_env_driven(self, fresh_app_settings, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDIA_OUTPUT_DIR", "/data/custom-media")
        settings = fresh_app_settings(_env_file=None)
        assert settings.media_output_dir == "/data/custom-media"
        assert settings.resolved_media_output_dir == "/data/custom-media"

    def test_tilde_is_expanded(self, fresh_app_settings) -> None:
        settings = fresh_app_settings(_env_file=None, media_output_dir="~/media-out")
        assert settings.resolved_media_output_dir == str(Path("~/media-out").expanduser())

    def test_env_vars_are_expanded(self, fresh_app_settings, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDIA_ROOT", "/data/media-root")
        settings = fresh_app_settings(_env_file=None, media_output_dir="$MEDIA_ROOT/generated")
        assert settings.resolved_media_output_dir == "/data/media-root/generated"

    def test_expansion_happens_on_use_not_at_import(self, fresh_app_settings, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = fresh_app_settings(_env_file=None, media_output_dir="${MEDIA_ROOT}/generated")
        assert settings.media_output_dir == "${MEDIA_ROOT}/generated"
        monkeypatch.setenv("MEDIA_ROOT", "/late")
        assert settings.resolved_media_output_dir == "/late/generated"

    def test_working_dir_env_vars_are_expanded_in_default(
        self, fresh_app_settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHIBI_WD", "/work/dir")
        settings = fresh_app_settings(_env_file=None, working_dir="$CHIBI_WD")
        assert settings.resolved_media_output_dir == "/work/dir/media"


class TestSupportsBinaryMediaCapability:
    """Runner capability flag: delivery-capable interfaces are marked explicitly."""

    def test_base_interface_cannot_deliver_binary_media(self) -> None:
        from chibi.services.interface import UserInterface

        assert UserInterface.supports_binary_media is False

    def test_telegram_family_can_deliver_binary_media(self) -> None:
        from chibi.services.interface import TelegramInterface
        from chibi.services.scheduler_interface import SchedulerInterface

        assert TelegramInterface.supports_binary_media is True
        assert SchedulerInterface.supports_binary_media is True

    def test_non_delivery_runners_cannot_deliver_binary_media(self) -> None:
        from chibi.runners.ide_transport import IDEInterface
        from chibi.services.scheduler_interface import StdioSchedulerInterface
        from chibi.services.terminal_interface import TerminalInterface

        assert IDEInterface.supports_binary_media is False
        assert TerminalInterface.supports_binary_media is False
        assert StdioSchedulerInterface.supports_binary_media is False


class TestPromptSlug:
    """Prompts are slugified for derived filenames."""

    async def test_slug_from_special_characters(self, tmp_path: Path, filesystem_access_enabled) -> None:
        saved = await save_media_to_disk(
            b"payload",
            tool_name="generate_image",
            prompt="  A  BIG / red cat!  ",
            ext="png",
            fallback_dir=str(tmp_path),
        )
        assert "generate_image-a-big-red-cat-" in Path(saved).name

    async def test_empty_prompt_falls_back_to_media(self, tmp_path: Path, filesystem_access_enabled) -> None:
        saved = await save_media_to_disk(
            b"payload", tool_name="generate_image", prompt="!!!", ext="png", fallback_dir=str(tmp_path)
        )
        assert "generate_image-media-" in Path(saved).name
