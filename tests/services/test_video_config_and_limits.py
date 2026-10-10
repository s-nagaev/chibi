"""Unit tests for video-generation config, user limits and generate_video plumbing."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from pydantic import ValidationError

from chibi.config import gpt_settings
from chibi.config.gpt import GPTSettings
from chibi.exceptions import NoProviderSelectedError
from chibi.models import User, VideoMeta
from chibi.schemas.app import VideoResult
from chibi.services import user as user_service
from chibi.storage.local import LocalStorage


@pytest.fixture(autouse=True)
def _clean_video_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from operator env / .env values for the video settings."""
    monkeypatch.delenv("VIDEO_GENERATIONS_LIMIT", raising=False)
    monkeypatch.delenv("VIDEO_GENERATIONS_WHITELIST", raising=False)
    monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 0, raising=False)
    monkeypatch.setattr(gpt_settings, "video_generations_whitelist_raw", None, raising=False)


class TestVideoConfig:
    def test_limit_defaults_to_zero(self) -> None:
        settings = GPTSettings(_env_file=None)
        assert settings.video_generations_monthly_limit == 0

    def test_limit_parsed_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIDEO_GENERATIONS_LIMIT", "7")
        settings = GPTSettings(_env_file=None)
        assert settings.video_generations_monthly_limit == 7

    def test_limit_must_be_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIDEO_GENERATIONS_LIMIT", "not-a-number")
        with pytest.raises(ValidationError):
            GPTSettings(_env_file=None)

    def test_whitelist_defaults_to_empty(self) -> None:
        settings = GPTSettings(_env_file=None)
        assert settings.video_generations_whitelist_raw is None
        assert settings.video_generations_whitelist == []

    def test_whitelist_parsed_from_env_and_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIDEO_GENERATIONS_WHITELIST", " 11 , 22 ,33")
        settings = GPTSettings(_env_file=None)
        assert settings.video_generations_whitelist == ["11", "22", "33"]


class TestVideoLimits:
    def test_limit_disabled_means_never_reached(self) -> None:
        user = User(id=1, videos=[VideoMeta(expire_at=1e12), VideoMeta(expire_at=1e12)])
        assert gpt_settings.video_generations_monthly_limit == 0
        assert user.has_reached_video_limits is False

    def test_limit_not_reached_below_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 3, raising=False)
        user = User(id=1, videos=[VideoMeta(expire_at=1e12)])
        assert user.has_reached_video_limits is False

    def test_limit_reached_at_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 2, raising=False)
        user = User(id=1, videos=[VideoMeta(expire_at=1e12), VideoMeta(expire_at=1e12)])
        assert user.has_reached_video_limits is True

    def test_whitelisted_user_bypasses_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 1, raising=False)
        monkeypatch.setattr(gpt_settings, "video_generations_whitelist_raw", "42", raising=False)
        user = User(id=42, videos=[VideoMeta(expire_at=1e12)])
        assert user.has_reached_video_limits is False

    def test_videos_field_backward_compatible_default(self) -> None:
        # Simulates a user persisted before the videos field existed.
        user = User.model_validate({"id": 123})
        assert user.videos == []
        assert user.images == []


class TestCountVideo:
    async def test_count_video_appends_and_persists(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))

        await db.count_video(1)

        user = await db.get_or_create_user(user_id=1)
        assert len(user.videos) == 1
        assert user.videos[0].expire_at > 0

    async def test_expired_videos_are_pruned_on_local_storage_read(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        await db.count_video(1)

        user = await db.get_or_create_user(user_id=1)
        user.videos.append(VideoMeta(expire_at=0))  # long expired
        await db.save_user(user)

        refreshed = await db.get_or_create_user(user_id=1)
        assert len(refreshed.videos) == 1
        assert all(video.expire_at > 0 for video in refreshed.videos)


def _make_video_provider(name: str) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.get_videos = AsyncMock(return_value=VideoResult(video=b"mp4-bytes"))
    return provider


class TestGenerateVideo:
    def _interface(self, user_id: int = 1, thread_id: int = 0) -> MagicMock:
        interface = MagicMock()
        interface.user_id = user_id
        interface.storage_id = user_id
        interface.thread_id = thread_id
        return interface

    async def test_falls_back_to_active_video_provider(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        interface = self._interface()
        sentinel = _make_video_provider("Alibaba")

        with patch.object(User, "providers", new_callable=PropertyMock) as providers_mock:
            providers_mock.return_value.get.return_value = sentinel
            with (
                patch.object(User, "get_active_video_provider", return_value=sentinel) as get_active,
                patch.object(User, "get_active_video_model", return_value=None),
            ):
                result = await user_service.generate_video.__wrapped__(db, interface=interface, prompt="a cat")

        assert result.video == b"mp4-bytes"
        get_active.assert_called_once_with(thread_id=0)
        sentinel.get_videos.assert_awaited_once_with(prompt="a cat", model=None, duration=None)
        user = await db.get_or_create_user(user_id=1)
        assert len(user.videos) == 1

    async def test_explicit_provider_name_skips_active_resolution(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        interface = self._interface()
        sentinel = _make_video_provider("MiniMax")

        with patch.object(User, "providers", new_callable=PropertyMock) as providers_mock:
            providers_mock.return_value.get.return_value = sentinel
            result = await user_service.generate_video.__wrapped__(
                db, interface=interface, prompt="a dog", model="H3", provider_name="MiniMax", duration=5
            )

        assert result.video == b"mp4-bytes"
        providers_mock.return_value.get.assert_called_once_with("MiniMax")
        sentinel.get_videos.assert_awaited_once_with(prompt="a dog", model="H3", duration=5)

    async def test_raises_when_no_provider_available(self, tmp_path) -> None:
        db = LocalStorage(storage_path=str(tmp_path))
        interface = self._interface()

        with patch.object(User, "providers", new_callable=PropertyMock) as providers_mock:
            providers_mock.return_value.get.return_value = None
            with patch.object(User, "get_active_video_provider", side_effect=NoProviderSelectedError):
                with pytest.raises(NoProviderSelectedError):
                    await user_service.generate_video.__wrapped__(db, interface=interface, prompt="a cat")

        user = await db.get_or_create_user(user_id=1)
        assert user.videos == []

    async def test_whitelist_enforcement_matches_image_flow(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parity with generate_image: the service-level check compares the raw int user_id
        against the string whitelist, so whitelisted users are still counted here — the
        whitelist is enforced by ``has_reached_video_limits`` (which compares str(id)),
        exactly like the image flow (media.py tool-level check)."""
        monkeypatch.setattr(gpt_settings, "video_generations_whitelist_raw", "1", raising=False)
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 1, raising=False)
        db = LocalStorage(storage_path=str(tmp_path))
        interface = self._interface(user_id=1)
        sentinel = _make_video_provider("ZhipuAI")

        with patch.object(User, "providers", new_callable=PropertyMock) as providers_mock:
            providers_mock.return_value.get.return_value = sentinel
            with (
                patch.object(User, "get_active_video_provider", return_value=sentinel),
                patch.object(User, "get_active_video_model", return_value="cogvideox-3"),
            ):
                await user_service.generate_video.__wrapped__(db, interface=interface, prompt="a bird")

        # Counted (same as images), but the user can never be blocked by the limit.
        user = await db.get_or_create_user(user_id=1)
        assert len(user.videos) == 1
        assert user.has_reached_video_limits is False
        sentinel.get_videos.assert_awaited_once_with(prompt="a bird", model="cogvideox-3", duration=None)


class TestUserHasReachedVideosGenerationLimit:
    async def test_service_helper_reflects_user_limits(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 1, raising=False)
        db = LocalStorage(storage_path=str(tmp_path))

        assert await user_service.user_has_reached_videos_generation_limit.__wrapped__(db, user_id=9) is False

        await db.count_video(9)
        assert await user_service.user_has_reached_videos_generation_limit.__wrapped__(db, user_id=9) is True
