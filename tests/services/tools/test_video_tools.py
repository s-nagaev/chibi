"""Unit tests for the video-generation tools: registration, limit check,
provider fallback, thumbnail constraints and the size-guard delivery."""

import io
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from chibi.config import gpt_settings
from chibi.models import User
from chibi.schemas.app import ModelChangeSchema, VideoResult
from chibi.services.providers.tools import GenerateVideoTool, GetAvailableVideoGenerationModelsTool
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import RegisteredChibiTools
from chibi.services.providers.tools.video import (
    THUMBNAIL_MAX_BYTES,
    THUMBNAIL_MAX_PIXELS,
    VIDEO_SIZE_GUARD_BYTES,
    prepare_video_thumbnail,
    sync_video_tool_registration,
)
from chibi.storage.local import LocalStorage


@contextmanager
def _sync_env(
    ready: bool,
    public_mode: bool = False,
    tools_whitelist: list[str] | None = None,
) -> Iterator[dict]:
    """Run sync_video_tool_registration with a controlled environment."""
    providers_mock = MagicMock()
    providers_mock.return_value.video_generation_ready = {"alibaba"} if ready else {}
    settings = MagicMock()
    settings.public_mode = public_mode
    settings.tools_whitelist = tools_whitelist or []
    with (
        patch("chibi.services.providers.RegisteredProviders", providers_mock),
        patch("chibi.services.providers.tools.video.gpt_settings", settings),
        patch.object(RegisteredChibiTools, "tools_map", dict(RegisteredChibiTools.tools_map)),
    ):
        sync_video_tool_registration()
        yield RegisteredChibiTools.tools_map


def _jpeg_bytes(width: int, height: int, color: str = "red") -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def mock_interface() -> MagicMock:
    interface = MagicMock()
    interface.user_id = 12345
    interface.chat_id = 6789
    interface.send_video = AsyncMock()
    interface.send_document = AsyncMock()
    return interface


@pytest.fixture
def mock_kwargs(mock_interface: MagicMock) -> dict:
    return {"interface": mock_interface, "user_id": 12345}


def _video_result(
    video: bytes = b"mp4-bytes",
    thumbnail: bytes | None = None,
    duration: int | None = None,
) -> VideoResult:
    return VideoResult(video=video, thumbnail=thumbnail, duration=duration)


class TestRegistration:
    def test_tools_exist_with_expected_names(self) -> None:
        assert GetAvailableVideoGenerationModelsTool.name == "get_available_video_generation_models"
        assert GenerateVideoTool.name == "generate_video"

    def test_generate_video_runs_in_background_by_default(self) -> None:
        assert GenerateVideoTool.run_in_background_by_default is True

    def test_generate_video_definition_requires_only_prompt(self) -> None:
        from typing import Any, cast

        function = cast(dict[str, Any], GenerateVideoTool.definition["function"])
        assert function["parameters"]["required"] == ["prompt"]
        properties = cast(dict[str, Any], function["parameters"]["properties"])
        assert set(properties) == {"provider", "video_model", "prompt", "duration"}

    def test_sync_registers_tools_when_a_provider_is_video_ready(self) -> None:
        with _sync_env(ready=True) as tools_map:
            assert "generate_video" in tools_map
            assert "get_available_video_generation_models" in tools_map

    def test_sync_deregisters_tools_without_a_video_ready_provider(self) -> None:
        with patch.object(RegisteredChibiTools, "tools_map", dict(RegisteredChibiTools.tools_map)) as seeded:
            seeded["generate_video"] = GenerateVideoTool
            seeded["get_available_video_generation_models"] = GetAvailableVideoGenerationModelsTool
            with _sync_env(ready=False) as tools_map:
                assert "generate_video" not in tools_map
                assert "get_available_video_generation_models" not in tools_map

    def test_sync_in_public_mode_keeps_tools_registered_even_without_a_ready_provider(self) -> None:
        with patch.object(RegisteredChibiTools, "tools_map", dict(RegisteredChibiTools.tools_map)) as seeded:
            seeded.pop("generate_video", None)
            seeded.pop("get_available_video_generation_models", None)
            with _sync_env(ready=False, public_mode=True) as tools_map:
                assert "generate_video" in tools_map
                assert "get_available_video_generation_models" in tools_map

    def test_sync_respects_tools_whitelist(self) -> None:
        with patch.object(RegisteredChibiTools, "tools_map", dict(RegisteredChibiTools.tools_map)) as seeded:
            seeded["generate_video"] = GenerateVideoTool
            seeded["get_available_video_generation_models"] = GetAvailableVideoGenerationModelsTool
            with _sync_env(ready=True, tools_whitelist=["text_to_speech", "generate_image"]) as tools_map:
                assert "generate_video" not in tools_map
                assert "get_available_video_generation_models" not in tools_map

    def test_sync_keeps_tools_outside_the_whitelist_scope_untouched(self) -> None:
        with patch.object(RegisteredChibiTools, "tools_map", {"text_to_speech": "sentinel"}):
            with _sync_env(ready=True, tools_whitelist=["text_to_speech", "generate_image"]) as tools_map:
                assert tools_map == {"text_to_speech": "sentinel"}

    def _fresh_interpreter_tools_map(self, env_overrides: dict[str, str]) -> list[str]:
        """Import the providers package in a fresh interpreter and return the tool names.

        Guards the actual wiring of sync_video_tool_registration() at the end
        of chibi.services.providers.__init__: deleting the call must turn these
        tests red.
        """
        env = {
            key: value for key, value in os.environ.items() if not key.endswith("_KEY") and not key.endswith("_TOKEN")
        }
        env["PUBLIC_MODE"] = "false"
        env.update(env_overrides)
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-c",
                "import chibi.services.providers; "
                "from chibi.services.providers.tools.tool import RegisteredChibiTools; "
                "import json; print(json.dumps(sorted(RegisteredChibiTools.tools_map)))",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=True,
        )
        return json.loads(result.stdout)

    def test_fresh_interpreter_without_keys_has_no_video_tools(self) -> None:
        tools = self._fresh_interpreter_tools_map({})
        assert "generate_video" not in tools
        assert "get_available_video_generation_models" not in tools

    def test_fresh_interpreter_with_video_key_registers_video_tools(self) -> None:
        tools = self._fresh_interpreter_tools_map({"ALIBABA_API_KEY": "dummy"})
        assert "generate_video" in tools
        assert "get_available_video_generation_models" in tools

    def test_fresh_interpreter_in_public_mode_registers_video_tools(self) -> None:
        tools = self._fresh_interpreter_tools_map({"ALIBABA_API_KEY": "dummy", "PUBLIC_MODE": "true"})
        assert "generate_video" in tools
        assert "get_available_video_generation_models" in tools

    def test_fresh_interpreter_whitelist_excludes_video_tools(self) -> None:
        tools = self._fresh_interpreter_tools_map(
            {"ALIBABA_API_KEY": "dummy", "TOOLS_WHITELIST": "text_to_speech,generate_image"}
        )
        assert "generate_video" not in tools
        assert "get_available_video_generation_models" not in tools


class TestGetAvailableVideoGenerationModelsTool:
    async def test_returns_available_models(self, mock_kwargs: dict) -> None:
        models = [
            ModelChangeSchema(provider="Alibaba", name="wan2.6-t2v", image_generation=False, video_generation=True)
        ]
        with patch(
            "chibi.services.providers.tools.video.get_models_available", new=AsyncMock(return_value=models)
        ) as mock_get:
            result = await GetAvailableVideoGenerationModelsTool.function(**mock_kwargs)

        mock_get.assert_awaited_once_with(user_id=12345, video_generation=True)
        assert result["available_models"][0]["provider"] == "Alibaba"
        assert result["available_models"][0]["name"] == "wan2.6-t2v"

    async def test_raises_without_user_id(self) -> None:
        with pytest.raises(ToolException):
            await GetAvailableVideoGenerationModelsTool.function()


class TestGenerateVideoLimitCheck:
    async def test_limit_reached_raises_and_does_not_generate(
        self, mock_kwargs: dict, mock_interface: MagicMock
    ) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=True),
            ),
            patch("chibi.services.providers.tools.video.generate_video", new=AsyncMock()) as mock_generate,
        ):
            with pytest.raises(ToolException, match="monthly limit"):
                await GenerateVideoTool.function(prompt="a cat", **mock_kwargs)

        mock_generate.assert_not_awaited()
        mock_interface.send_video.assert_not_awaited()
        mock_interface.send_document.assert_not_awaited()

    async def test_limit_not_reached_generates(self, mock_kwargs: dict) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result()),
            ) as mock_generate,
        ):
            result = await GenerateVideoTool.function(prompt="a cat", **mock_kwargs)

        assert result == {"detail": "Video was successfully generated and sent to user."}
        mock_generate.assert_awaited_once()


class TestGenerateVideoProviderFallback:
    async def test_without_provider_args_resolves_via_active_provider(self, mock_kwargs: dict) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result()),
            ) as mock_generate,
        ):
            await GenerateVideoTool.function(prompt="a cat", **mock_kwargs)

        kwargs = mock_generate.await_args.kwargs if mock_generate.await_args else {}
        assert kwargs["provider_name"] is None
        assert kwargs["model"] is None

    async def test_provider_and_model_are_passed_through(self, mock_kwargs: dict) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result()),
            ) as mock_generate,
        ):
            await GenerateVideoTool.function(
                prompt="a cat", provider="Alibaba", video_model="wan2.6-t2v", duration=5, **mock_kwargs
            )

        kwargs = mock_generate.await_args.kwargs if mock_generate.await_args else {}
        assert kwargs["provider_name"] == "Alibaba"
        assert kwargs["model"] == "wan2.6-t2v"
        assert kwargs["duration"] == 5


class TestThumbnailConstraints:
    def test_small_valid_jpeg_passes_through_within_constraints(self) -> None:
        thumbnail = _jpeg_bytes(200, 150)
        result = prepare_video_thumbnail(thumbnail)
        assert result is not None
        assert len(result) < THUMBNAIL_MAX_BYTES

    def test_oversized_jpeg_is_downscaled_to_320px(self) -> None:
        from PIL import Image

        thumbnail = _jpeg_bytes(1024, 768)
        result = prepare_video_thumbnail(thumbnail)
        assert result is not None
        assert len(result) < THUMBNAIL_MAX_BYTES
        with Image.open(io.BytesIO(result)) as image:
            assert image.format == "JPEG"
            assert image.width <= THUMBNAIL_MAX_PIXELS
            assert image.height <= THUMBNAIL_MAX_PIXELS

    def test_large_image_stays_below_size_limit(self) -> None:
        thumbnail = _jpeg_bytes(2000, 2000)
        result = prepare_video_thumbnail(thumbnail)
        assert result is not None
        assert len(result) < THUMBNAIL_MAX_BYTES

    def test_invalid_bytes_are_dropped(self) -> None:
        assert prepare_video_thumbnail(b"not-an-image") is None

    def test_empty_bytes_are_dropped(self) -> None:
        assert prepare_video_thumbnail(b"") is None


class TestSizeGuardDelivery:
    async def test_video_within_limit_is_sent_as_video(self, mock_kwargs: dict, mock_interface: MagicMock) -> None:
        thumbnail = _jpeg_bytes(300, 200)
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result(video=b"mp4", thumbnail=thumbnail, duration=5)),
            ),
        ):
            await GenerateVideoTool.function(prompt="a cat on a bike", **mock_kwargs)

        mock_interface.send_video.assert_awaited_once()
        mock_interface.send_document.assert_not_awaited()
        kwargs = mock_interface.send_video.await_args.kwargs
        assert kwargs["video"] == b"mp4"
        assert kwargs["duration"] == 5
        assert kwargs["thumbnail"] is not None
        assert kwargs["filename"].endswith(".mp4")

    async def test_video_over_limit_is_sent_as_document(self, mock_kwargs: dict, mock_interface: MagicMock) -> None:
        big_video = b"x" * (VIDEO_SIZE_GUARD_BYTES + 1)
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result(video=big_video, duration=10)),
            ),
        ):
            await GenerateVideoTool.function(prompt="a cat on a bike", **mock_kwargs)

        mock_interface.send_document.assert_awaited_once()
        mock_interface.send_video.assert_not_awaited()
        kwargs = mock_interface.send_document.await_args.kwargs
        assert kwargs["document"] == big_video
        assert "file" in kwargs["caption"].lower()

    async def test_thumbnail_violating_constraints_is_dropped(
        self, mock_kwargs: dict, mock_interface: MagicMock
    ) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result(video=b"mp4", thumbnail=b"broken")),
            ),
        ):
            await GenerateVideoTool.function(prompt="a cat", **mock_kwargs)

        kwargs = mock_interface.send_video.await_args.kwargs
        assert kwargs["thumbnail"] is None

    async def test_filename_is_sanitised_from_the_prompt(self, mock_kwargs: dict, mock_interface: MagicMock) -> None:
        with (
            patch(
                "chibi.services.providers.tools.video.user_has_reached_videos_generation_limit",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "chibi.services.providers.tools.video.generate_video",
                new=AsyncMock(return_value=_video_result()),
            ),
        ):
            await GenerateVideoTool.function(prompt='a "cat"/on\na 🚲 bike?', **mock_kwargs)

        filename = mock_interface.send_video.await_args.kwargs["filename"]
        assert filename.endswith(".mp4")
        assert "/" not in filename
        assert "\\" not in filename
        assert '"' not in filename
        assert "\n" not in filename
        assert "?" not in filename
        assert "🚲" not in filename


def _stub_database_provider(db: LocalStorage) -> MagicMock:
    stub = MagicMock()
    stub.get_database = AsyncMock(return_value=db)
    return stub


class TestFullToolFlowWithMockedProvider:
    """E2E-style tool flow: the real limit check and ``generate_video`` service run
    against a LocalStorage instance, with the provider mocked at the ``get_videos``
    boundary. Verifies that limit checking, usage counting and delivery (video vs
    document) work together, not just in isolation."""

    @pytest.fixture
    def e2e_env(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(gpt_settings, "video_generations_whitelist_raw", None, raising=False)
        db = LocalStorage(storage_path=str(tmp_path))
        provider = MagicMock()
        provider.name = "Alibaba"
        provider.get_videos = AsyncMock(
            return_value=_video_result(video=b"mp4-bytes", thumbnail=_jpeg_bytes(300, 200), duration=5)
        )
        with (
            patch("chibi.storage.database._db_provider", _stub_database_provider(db)),
            patch.object(User, "providers", new_callable=PropertyMock) as providers_mock,
            patch.object(User, "get_active_video_provider", return_value=provider),
            patch.object(User, "get_active_video_model", return_value="wan2.6-t2v"),
        ):
            providers_mock.return_value.get.return_value = provider
            yield db, provider

    async def test_flow_delivers_video_and_counts_usage(
        self, e2e_env, mock_kwargs: dict, mock_interface: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, provider = e2e_env
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 5, raising=False)

        result = await GenerateVideoTool.function(prompt="a cat on a bike", **mock_kwargs)

        assert result == {"detail": "Video was successfully generated and sent to user."}
        provider.get_videos.assert_awaited_once_with(prompt="a cat on a bike", model="wan2.6-t2v", duration=None)
        mock_interface.send_video.assert_awaited_once()
        mock_interface.send_document.assert_not_awaited()
        kwargs = mock_interface.send_video.await_args.kwargs
        assert kwargs["video"] == b"mp4-bytes"
        assert kwargs["duration"] == 5
        assert kwargs["thumbnail"] is not None

        user = await db.get_or_create_user(user_id=12345)
        assert len(user.videos) == 1

    async def test_flow_sends_oversized_video_as_document(
        self, e2e_env, mock_kwargs: dict, mock_interface: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, provider = e2e_env
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 5, raising=False)
        big_video = b"x" * (VIDEO_SIZE_GUARD_BYTES + 1)
        provider.get_videos.return_value = _video_result(video=big_video, duration=10)

        await GenerateVideoTool.function(prompt="a big cat", **mock_kwargs)

        mock_interface.send_document.assert_awaited_once()
        mock_interface.send_video.assert_not_awaited()
        kwargs = mock_interface.send_document.await_args.kwargs
        assert kwargs["document"] == big_video
        assert "file" in kwargs["caption"].lower()

        user = await db.get_or_create_user(user_id=12345)
        assert len(user.videos) == 1

    async def test_flow_blocks_generation_when_limit_reached(
        self, e2e_env, mock_kwargs: dict, mock_interface: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db, provider = e2e_env
        monkeypatch.setattr(gpt_settings, "video_generations_monthly_limit", 1, raising=False)
        await db.count_video(12345)

        with pytest.raises(ToolException, match="monthly limit"):
            await GenerateVideoTool.function(prompt="a cat", **mock_kwargs)

        provider.get_videos.assert_not_awaited()
        mock_interface.send_video.assert_not_awaited()
        mock_interface.send_document.assert_not_awaited()

        user = await db.get_or_create_user(user_id=12345)
        assert len(user.videos) == 1  # only the pre-seeded generation
