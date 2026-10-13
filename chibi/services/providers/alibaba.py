from asyncio import sleep
from http import HTTPStatus
from time import monotonic
from typing import Any

import dashscope
from dashscope.aigc.image_generation import AioImageGeneration
from dashscope.aigc.video_synthesis import AioVideoSynthesis
from dashscope.api_entities.dashscope_response import Choice, ImageGenerationResponse, Message
from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import NoModelSelectedError, NoResponseError, ServiceResponseError
from chibi.schemas.app import ModelChangeSchema, VideoResult
from chibi.services.providers.provider import OpenAIFriendlyProvider, download_media

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


class Alibaba(OpenAIFriendlyProvider):
    api_key = gpt_settings.alibaba_key
    chat_ready = True
    image_generation_ready = True
    video_generation_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    name = "Alibaba"
    model_name_keywords_exclude = ["tts", "stt", "image", "embed"]
    default_model = "qwen3.7-plus"
    default_moderation_model = "qwen3.8-flash"

    default_image_model = "qwen-image-plus"
    default_video_model = "wan2.6-t2v"
    default_vision_model = "qwen3-vl-flash"

    @staticmethod
    def _get_image_url(choice: Choice) -> str | None:
        if isinstance(choice.message.content, str):
            return choice.message.content
        return choice.message.content[0].get("image")

    async def get_images(
        self, prompt: str, model: str | None = None, images: list[tuple[bytes, str]] | None = None
    ) -> list[str]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate.
            model: Image model name. Defaults to the provider's default image model.
            images: Optional image input as (bytes, mime type) tuples. Not supported
                by this provider.

        Returns:
            List of image URLs.

        Raises:
            ServiceResponseError: If image input is provided, which this provider
                does not support, or if the provider returns an error response.
        """
        model = model or self.default_model
        if images:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="This provider does not support image input for image generation.",
            )
        message = Message(role="user", content=[{"text": prompt}])
        number_of_images = 1 if "qwen" in model or "z-image" in model else gpt_settings.image_n_choices
        response: ImageGenerationResponse = await AioImageGeneration.call(
            api_key=self.token,
            model=model,
            messages=[message],
            n=number_of_images,
            size=gpt_settings.image_size_alibaba,
            prompt_extend=True,
            watermark=False,
        )

        if response.status_code != HTTPStatus.OK:
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=(
                    f"Unexpected response status code: {response.status_code}. "
                    f"Response code: {response.code}. Message: {response.message}"
                ),
            )
        image_urls: list[str] = []
        for choice in response.output.choices:
            if url := self._get_image_url(choice):
                image_urls.append(url)
        return image_urls

    async def get_videos(
        self,
        model: str | None,
        prompt: str,
        image: bytes | str | None = None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> VideoResult:
        """Generate a text-to-video clip via the DashScope async API.

        Args:
            model: Video model name, or None for the provider default.
            prompt: Video generation prompt.
            image: Optional image input; text-to-video only, so must be None.
            duration: Optional clip length in seconds, if the model supports it.
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

        extra_kwargs: dict[str, Any] = {}
        if duration is not None:
            extra_kwargs["duration"] = duration

        started = monotonic()
        response = await AioVideoSynthesis.async_call(api_key=self.token, model=model, prompt=prompt, **extra_kwargs)
        if response.status_code != HTTPStatus.OK:
            logger.error(
                f"[Video] {self.name}: failed to submit {model} video generation task "
                f"(HTTP {response.status_code}): {response.message}"
            )
            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail=(
                    f"Failed to submit the video generation task. "
                    f"Response code: {response.code}. Message: {response.message}"
                ),
            )

        task_id = response.output["task_id"]
        logger.info(f"[Video] {self.name}: submitted {model} task {task_id}")

        deadline = monotonic() + self.video_poll_timeout
        poll_attempts = 0
        while True:
            if monotonic() > deadline:
                logger.error(
                    f"[Video] {self.name}: task {task_id} did not complete within "
                    f"{self.video_poll_timeout}s after {poll_attempts} poll attempts"
                )
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=f"Video generation task {task_id} did not complete within {self.video_poll_timeout} s",
                )
            await sleep(self.video_poll_interval)
            response = await AioVideoSynthesis.fetch(task=task_id, api_key=self.token)
            if response.status_code != HTTPStatus.OK:
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=(
                        f"Failed to fetch the video generation task {task_id}. "
                        f"Response code: {response.code}. Message: {response.message}"
                    ),
                )
            task_status = response.output["task_status"]
            poll_attempts += 1
            logger.debug(
                f"[Video] {self.name}: task {task_id} poll #{poll_attempts} "
                f"status={task_status} elapsed={monotonic() - started:.0f}s"
            )
            if task_status == "SUCCEEDED":
                break
            if task_status in ("FAILED", "CANCELED", "UNKNOWN"):
                logger.error(
                    f"[Video] {self.name}: task {task_id} finished with status {task_status} "
                    f"({response.output.get('message')})"
                )
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=(
                        f"Video generation task {task_id} finished with status {task_status}. "
                        f"Code: {response.output.get('code')}. Message: {response.output.get('message')}"
                    ),
                )

        video_url = response.output.get("video_url")
        if not video_url:
            raise NoResponseError(
                provider=self.name, model=model, detail=f"Succeeded task {task_id} returned no video URL"
            )
        try:
            video = await download_media(url=video_url)
        except Exception as e:
            raise ServiceResponseError(
                provider=self.name, model=model, detail=f"Failed to download the generated video: {e}"
            ) from e

        usage = response.output.get("usage") or {}
        reported_duration = usage.get("video_duration")
        clip_duration = int(reported_duration) if reported_duration else duration
        logger.info(
            f"[Video] {self.name}: task {task_id} completed in {monotonic() - started:.0f}s "
            f"duration={clip_duration} size={len(video)} bytes"
        )
        return VideoResult(
            video=video,
            thumbnail=None,
            duration=clip_duration,
        )

    async def get_available_models(
        self, image_generation: bool = False, video_generation: bool = False
    ) -> list[ModelChangeSchema]:
        models = await super().get_available_models(
            image_generation=image_generation, video_generation=video_generation
        )

        if image_generation:
            wan_models = [
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.6-t2i",
                    display_name="Wan 2.6",
                    image_generation=True,
                ),
            ]

            models += wan_models

        if video_generation:
            wan_video_models = [
                ModelChangeSchema(
                    provider=self.name,
                    name="wan2.6-t2v",
                    display_name="Wan 2.6 T2V",
                    image_generation=False,
                    video_generation=True,
                ),
            ]

            models += wan_video_models

        return models
