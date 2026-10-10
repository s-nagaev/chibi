from asyncio import sleep
from time import monotonic
from typing import Any
from urllib.parse import urljoin

from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import NoModelSelectedError, NoResponseError, ServiceResponseError
from chibi.schemas.app import ModelChangeSchema, VideoResult
from chibi.services.providers.provider import OpenAIFriendlyProvider, RestApiFriendlyProvider, download_media


class ZhipuAI(OpenAIFriendlyProvider, RestApiFriendlyProvider):
    api_key = gpt_settings.zhipuai_key
    chat_ready = True
    moderation_ready = True
    video_generation_ready = True
    vision_ready = False

    name = "ZhipuAI"
    model_name_keywords = ["glm"]
    base_url = "https://api.z.ai/api/paas/v4/"
    default_model = "glm-5"
    default_image_model = "glm-image"
    default_video_model = "cogvideox-3"
    default_moderation_model = "glm-4-32b-0414-128k"
    default_vision_model = "GLM-4.6V-FlashX"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def get_model_display_name(self, model_name: str) -> str:
        return model_name.upper()

    async def get_available_models(
        self, image_generation: bool = False, video_generation: bool = False
    ) -> list[ModelChangeSchema]:
        if image_generation:
            return [
                ModelChangeSchema(
                    provider=self.name, name="glm-image", display_name="GLM Image", image_generation=True
                ),
                ModelChangeSchema(
                    provider=self.name, name="cogview-4-250304", display_name="CogView 4", image_generation=True
                ),
            ]
        if video_generation:
            return [
                ModelChangeSchema(
                    provider=self.name,
                    name="cogvideox-3",
                    display_name="CogVideoX 3",
                    image_generation=False,
                    video_generation=True,
                ),
            ]
        models = await super().get_available_models(image_generation=False, video_generation=video_generation)

        additional_models = [
            ModelChangeSchema(
                provider=self.name, name="glm-4-32b-0414-128k", display_name="GLM 4 32B", image_generation=False
            ),
            ModelChangeSchema(
                provider=self.name, name="glm-4.7-flash", display_name="GLM 4.7 Flash", image_generation=False
            ),
            ModelChangeSchema(
                provider=self.name, name="glm-4.7-flashx", display_name="GLM 4.7 FlashX", image_generation=False
            ),
        ]
        models.extend(additional_models)
        return models

    async def get_images(self, prompt: str, model: str | None = None) -> list[str]:
        model = model or self.default_image_model
        url = "https://api.z.ai/api/paas/v4/images/generations"
        response = await self._request(
            method="POST",
            url=url,
            data={
                "model": model,
                "prompt": prompt,
                "size": "1728x960",
            },
        )
        response_data = response.json()
        data = response_data.get("data")
        if not data:
            raise NoResponseError(provider=self.name, model=model, detail="Server returned no data")
        image_url = data[0].get("url")

        return [
            image_url,
        ]

    async def get_videos(
        self,
        model: str | None,
        prompt: str,
        image: bytes | str | None = None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> VideoResult:
        """Generate a text-to-video clip via the ZhipuAI video API.

        Args:
            model: Video model name, or None for the provider default.
            prompt: Video generation prompt.
            image: Optional image input; text-to-video only, so must be None.
            duration: Optional clip length in seconds (5 or 10); ignored otherwise.
            kwargs: Additional provider-specific options.

        Returns:
            The fully downloaded video generation result.

        Raises:
            NotImplementedError: If an image input is provided.
            NoModelSelectedError: If no video model is selected.
        """
        if image is not None:
            raise NotImplementedError(f"{self.name} video generation supports text-to-video only for now")

        model = model or self.default_video_model
        if not model:
            raise NoModelSelectedError(provider=self.name, detail="No video generation model selected")

        payload: dict[str, Any] = {"model": model, "prompt": prompt}
        if duration is not None and duration in (5, 10):
            payload["duration"] = duration

        response = await self._request(method="POST", url=urljoin(self.base_url, "videos/generations"), data=payload)
        task_id = response.json().get("id")
        if not task_id:
            raise NoResponseError(provider=self.name, model=model, detail="Server returned no task id")

        deadline = monotonic() + self.video_poll_timeout
        data: dict[str, Any] = {}
        while True:
            if monotonic() > deadline:
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=f"Video generation task {task_id} did not complete within {self.video_poll_timeout} s",
                )
            await sleep(self.video_poll_interval)
            poll_response = await self._request(method="GET", url=urljoin(self.base_url, f"async-result/{task_id}"))
            data = poll_response.json()
            task_status = str(data.get("task_status", "")).upper()
            if task_status == "SUCCESS":
                break
            if task_status == "FAIL":
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=f"Video generation task {task_id} failed: {data.get('message') or 'unknown error'}",
                )

        video_result = data.get("video_result") or []
        video_url = video_result[0].get("url") if video_result else None
        if not video_url:
            raise NoResponseError(
                provider=self.name, model=model, detail=f"Completed task {task_id} returned no video URL"
            )

        try:
            video = await download_media(url=video_url)
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name, model=model, detail=f"Failed to download the generated video: {e}"
            ) from e

        thumbnail: bytes | None = None
        cover_image_url = video_result[0].get("cover_image_url")
        if cover_image_url:
            try:
                thumbnail = await download_media(url=cover_image_url)
            except Exception as e:
                logger.warning(f"Failed to download the video cover image for {self.name}/{model}: {e}")

        return VideoResult(video=video, thumbnail=thumbnail, duration=duration)
