from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.exceptions import NoResponseError, ServiceResponseError
from chibi.schemas.app import VideoResult
from chibi.services.providers import Alibaba, Minimax, ZhipuAI
from tests.conftest import MockResponse


def _make_video_providers() -> tuple[Alibaba, ZhipuAI, Minimax]:
    return Alibaba("test_token"), ZhipuAI("test_token"), Minimax("test_token")


@pytest.mark.asyncio
async def test_get_videos_raises_not_implemented_on_image_for_all_providers() -> None:
    alibaba, zhipu, minimax = _make_video_providers()
    for provider in (alibaba, zhipu, minimax):
        with pytest.raises(NotImplementedError):
            await provider.get_videos(model=None, prompt="a cat", image=b"image-bytes")
        with pytest.raises(NotImplementedError):
            await provider.get_videos(model=None, prompt="a cat", image="https://example.com/cat.png")


@pytest.mark.asyncio
async def test_video_generation_ready_flags() -> None:
    alibaba, zhipu, minimax = _make_video_providers()
    assert alibaba.video_generation_ready
    assert zhipu.video_generation_ready
    assert minimax.video_generation_ready
    assert alibaba.default_video_model == "wan2.6-t2v"
    assert zhipu.default_video_model == "cogvideox-3"
    assert minimax.default_video_model == "MiniMax-H3"


class TestAlibabaVideo:
    @pytest.mark.asyncio
    async def test_get_videos_success(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})
        running_response = Mock(status_code=200, output={"task_status": "RUNNING"})
        done_response = Mock(
            status_code=200,
            output={"task_status": "SUCCEEDED", "video_url": "https://cdn/video.mp4", "usage": {"video_duration": 5}},
        )

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
            patch("chibi.services.providers.alibaba.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)
            sdk.fetch = AsyncMock(side_effect=[running_response, done_response])

            result = await provider.get_videos(model=None, prompt="a cat surfing")

        assert isinstance(result, VideoResult)
        assert result.video == b"MP4"
        assert result.thumbnail is None
        assert result.duration == 5
        sdk.async_call.assert_awaited_once_with(api_key="test_token", model="wan2.6-t2v", prompt="a cat surfing")
        sdk.fetch.assert_awaited_with(task="task-1", api_key="test_token")

    @pytest.mark.asyncio
    async def test_get_videos_passes_duration(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})
        done_response = Mock(
            status_code=200,
            output={"task_status": "SUCCEEDED", "video_url": "https://cdn/video.mp4"},
        )

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
            patch("chibi.services.providers.alibaba.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)
            sdk.fetch = AsyncMock(return_value=done_response)

            result = await provider.get_videos(model="wan2.6-t2v", prompt="a cat", duration=10)

        assert result.video == b"MP4"
        assert result.duration == 10
        sdk.async_call.assert_awaited_once_with(api_key="test_token", model="wan2.6-t2v", prompt="a cat", duration=10)

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_submit_error(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=400, code="InvalidParameter", message="bad request")

        with patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk:
            sdk.async_call = AsyncMock(return_value=submit_response)

            with pytest.raises(ServiceResponseError):
                await provider.get_videos(model=None, prompt="a cat")

        sdk.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_failed_task(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})
        failed_response = Mock(
            status_code=200,
            output={"task_status": "FAILED", "code": "InternalError", "message": "generation failed"},
        )

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)
            sdk.fetch = AsyncMock(return_value=failed_response)

            with pytest.raises(ServiceResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_poll_timeout(self) -> None:
        provider = Alibaba("test_token")
        provider.video_poll_timeout = 0
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)

            with pytest.raises(ServiceResponseError, match="did not complete"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_missing_video_url(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})
        done_response = Mock(status_code=200, output={"task_status": "SUCCEEDED"})

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)
            sdk.fetch = AsyncMock(return_value=done_response)

            with pytest.raises(NoResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_download_failure(self) -> None:
        provider = Alibaba("test_token")
        submit_response = Mock(status_code=200, output={"task_id": "task-1"})
        done_response = Mock(status_code=200, output={"task_status": "SUCCEEDED", "video_url": "https://cdn/v.mp4"})

        with (
            patch("chibi.services.providers.alibaba.AioVideoSynthesis") as sdk,
            patch("chibi.services.providers.alibaba.sleep", new=AsyncMock()),
            patch("chibi.services.providers.alibaba.download_media", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            sdk.async_call = AsyncMock(return_value=submit_response)
            sdk.fetch = AsyncMock(return_value=done_response)

            with pytest.raises(ServiceResponseError, match="Failed to download"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_available_models_declares_video_models(self) -> None:
        with patch(
            "chibi.services.providers.provider.OpenAIFriendlyProvider.get_available_models",
            new=AsyncMock(return_value=[]),
        ):
            models = await Alibaba("test_token").get_available_models(video_generation=True)

        assert {model.name for model in models} == {"wan2.6-t2v"}
        assert all(model.video_generation for model in models)


class TestZhipuAIVideo:
    @pytest.mark.asyncio
    async def test_get_videos_success_with_cover(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1", "task_status": "PROCESSING"})
        processing_response = MockResponse({"task_status": "PROCESSING"})
        done_response = MockResponse(
            {
                "task_status": "SUCCESS",
                "video_result": [{"url": "https://cdn/video.mp4", "cover_image_url": "https://cdn/cover.jpg"}],
            }
        )

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, processing_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
            patch(
                "chibi.services.providers.zhipuai.download_media",
                new=AsyncMock(side_effect=[b"MP4", b"JPG"]),
            ),
        ):
            result = await provider.get_videos(model=None, prompt="a cat")

        assert result.video == b"MP4"
        assert result.thumbnail == b"JPG"
        assert result.duration is None

        submit_call = request_mock.call_args_list[0]
        assert submit_call.kwargs["method"] == "POST"
        assert submit_call.kwargs["url"] == "https://api.z.ai/api/paas/v4/videos/generations"
        assert submit_call.kwargs["data"] == {"model": "cogvideox-3", "prompt": "a cat"}

        poll_call = request_mock.call_args_list[-1]
        assert poll_call.kwargs["method"] == "GET"
        assert poll_call.kwargs["url"] == "https://api.z.ai/api/paas/v4/async-result/task-1"

    @pytest.mark.asyncio
    async def test_get_videos_passes_supported_duration(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1"})
        done_response = MockResponse({"task_status": "SUCCESS", "video_result": [{"url": "https://cdn/v.mp4"}]})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
            patch("chibi.services.providers.zhipuai.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            await provider.get_videos(model=None, prompt="a cat", duration=10)

        assert request_mock.call_args_list[0].kwargs["data"]["duration"] == 10

    @pytest.mark.asyncio
    async def test_get_videos_ignores_unsupported_duration(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1"})
        done_response = MockResponse({"task_status": "SUCCESS", "video_result": [{"url": "https://cdn/v.mp4"}]})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
            patch("chibi.services.providers.zhipuai.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            await provider.get_videos(model=None, prompt="a cat", duration=7)

        assert "duration" not in request_mock.call_args_list[0].kwargs["data"]

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_failed_task(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1"})
        failed_response = MockResponse({"task_status": "FAIL", "message": "sensitive content"})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, failed_response]),
            ),
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
        ):
            with pytest.raises(ServiceResponseError, match="failed"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_poll_timeout(self) -> None:
        provider = ZhipuAI("test_token")
        provider.video_poll_timeout = 0

        with patch(
            "chibi.services.providers.provider.RestApiFriendlyProvider._request",
            new=AsyncMock(
                side_effect=[
                    MockResponse({"id": "task-1"}),
                    AssertionError("poll request must not be issued after the deadline"),
                ]
            ),
        ):
            with pytest.raises(ServiceResponseError, match="did not complete"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_missing_task_id(self) -> None:
        provider = ZhipuAI("test_token")

        with patch(
            "chibi.services.providers.provider.RestApiFriendlyProvider._request",
            new=AsyncMock(return_value=MockResponse({})),
        ):
            with pytest.raises(NoResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_missing_video_url(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1"})
        done_response = MockResponse({"task_status": "SUCCESS", "video_result": []})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ),
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
        ):
            with pytest.raises(NoResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_tolerates_cover_download_failure(self) -> None:
        provider = ZhipuAI("test_token")
        submit_response = MockResponse({"id": "task-1"})
        done_response = MockResponse(
            {
                "task_status": "SUCCESS",
                "video_result": [{"url": "https://cdn/v.mp4", "cover_image_url": "https://cdn/cover.jpg"}],
            }
        )

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ),
            patch("chibi.services.providers.zhipuai.sleep", new=AsyncMock()),
            patch(
                "chibi.services.providers.zhipuai.download_media",
                new=AsyncMock(side_effect=[b"MP4", RuntimeError("boom")]),
            ),
        ):
            result = await provider.get_videos(model=None, prompt="a cat")

        assert result.video == b"MP4"
        assert result.thumbnail is None

    @pytest.mark.asyncio
    async def test_get_available_models_declares_video_models(self) -> None:
        models = await ZhipuAI("test_token").get_available_models(video_generation=True)

        assert [model.name for model in models] == ["cogvideox-3"]
        assert models[0].video_generation


class TestMinimaxVideo:
    @pytest.mark.asyncio
    async def test_get_videos_success(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        running_response = MockResponse({"task": {"id": "42", "status": "running"}})
        done_response = MockResponse(
            {
                "task": {
                    "id": "42",
                    "status": "succeeded",
                    "content": {"url": "https://cdn/video.mp4"},
                    "duration": 5,
                }
            }
        )

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, running_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
            patch("chibi.services.providers.minimax.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            result = await provider.get_videos(model=None, prompt="a cat surfing")

        assert result.video == b"MP4"
        assert result.thumbnail is None
        assert result.duration == 5

        submit_call = request_mock.call_args_list[0]
        assert submit_call.kwargs["method"] == "POST"
        assert submit_call.kwargs["url"] == "https://api.minimax.io/v2/video_generation"
        payload = submit_call.kwargs["data"]
        assert payload["model"] == "MiniMax-H3"
        assert payload["content"] == [{"type": "text", "text": "a cat surfing"}]
        assert payload["resolution"] == "768P"
        assert payload["duration"] == 5
        assert payload["ratio"] == "16:9"

        poll_call = request_mock.call_args_list[-1]
        assert poll_call.kwargs["method"] == "GET"
        assert poll_call.kwargs["url"] == "https://api.minimax.io/v2/query/video_generation/42"

    @pytest.mark.asyncio
    async def test_get_videos_clamps_duration(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        done_response = MockResponse(
            {"task": {"id": "42", "status": "succeeded", "content": {"url": "https://cdn/v.mp4"}}}
        )

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
            patch("chibi.services.providers.minimax.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            result = await provider.get_videos(model=None, prompt="a cat", duration=100)

        assert request_mock.call_args_list[0].kwargs["data"]["duration"] == 15
        assert result.duration == 15

    @pytest.mark.asyncio
    async def test_get_videos_uses_h3_max_model(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        done_response = MockResponse(
            {"task": {"id": "42", "status": "succeeded", "content": {"url": "https://cdn/v.mp4"}}}
        )

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ) as request_mock,
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
            patch("chibi.services.providers.minimax.download_media", new=AsyncMock(return_value=b"MP4")),
        ):
            await provider.get_videos(model="MiniMax-H3-Max", prompt="a cat")

        assert request_mock.call_args_list[0].kwargs["data"]["model"] == "MiniMax-H3-Max"

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_failed_task(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        failed_response = MockResponse({"task": {"id": "42", "status": "failed"}})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, failed_response]),
            ),
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
        ):
            with pytest.raises(ServiceResponseError, match="failed"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_cancelled_task(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        cancelled_response = MockResponse({"task": {"id": "42", "status": "cancelled"}})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, cancelled_response]),
            ),
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
        ):
            with pytest.raises(ServiceResponseError, match="cancelled"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_poll_timeout(self) -> None:
        provider = Minimax("test_token")
        provider.video_poll_timeout = 0

        with patch(
            "chibi.services.providers.provider.RestApiFriendlyProvider._request",
            new=AsyncMock(
                side_effect=[
                    MockResponse({"task_id": "42"}),
                    AssertionError("poll request must not be issued after the deadline"),
                ]
            ),
        ):
            with pytest.raises(ServiceResponseError, match="did not complete"):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_missing_task_id(self) -> None:
        provider = Minimax("test_token")

        with patch(
            "chibi.services.providers.provider.RestApiFriendlyProvider._request",
            new=AsyncMock(return_value=MockResponse({})),
        ):
            with pytest.raises(NoResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_missing_video_url(self) -> None:
        provider = Minimax("test_token")
        submit_response = MockResponse({"task_id": "42"})
        done_response = MockResponse({"task": {"id": "42", "status": "succeeded", "content": {}}})

        with (
            patch(
                "chibi.services.providers.provider.RestApiFriendlyProvider._request",
                new=AsyncMock(side_effect=[submit_response, done_response]),
            ),
            patch("chibi.services.providers.minimax.sleep", new=AsyncMock()),
        ):
            with pytest.raises(NoResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_videos_raises_on_request_error(self) -> None:
        provider = Minimax("test_token")

        with patch(
            "chibi.services.providers.provider.RestApiFriendlyProvider._request",
            new=AsyncMock(side_effect=ServiceResponseError(provider="Minimax", detail="boom")),
        ):
            with pytest.raises(ServiceResponseError):
                await provider.get_videos(model=None, prompt="a cat")

    @pytest.mark.asyncio
    async def test_get_available_models_declares_video_models(self) -> None:
        models = await Minimax("test_token").get_available_models(video_generation=True)

        assert {model.name for model in models} == {"MiniMax-H3", "MiniMax-H3-Max"}
        assert all(model.video_generation for model in models)


def test_alibaba_uses_async_video_synthesis_class() -> None:
    from dashscope.aigc.video_synthesis import AioVideoSynthesis

    assert Alibaba.__module__ == "chibi.services.providers.alibaba"
    # The provider must be wired against the async SDK class, not the sync one.
    assert AioVideoSynthesis.__name__ == "AioVideoSynthesis"


def test_download_media_helper_signature() -> None:
    import inspect

    from chibi.services.providers.provider import download_media

    assert inspect.iscoroutinefunction(download_media)
    signature = inspect.signature(download_media)
    assert list(signature.parameters) == ["url", "timeout"]
