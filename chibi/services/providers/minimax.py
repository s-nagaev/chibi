from asyncio import sleep
from time import monotonic
from typing import Any
from urllib.parse import urljoin

from anthropic import AsyncClient
from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError, NoModelSelectedError, NoResponseError, ServiceResponseError
from chibi.schemas.app import ModelChangeSchema, VideoResult
from chibi.services.providers.provider import AnthropicFriendlyProvider, download_media


class Minimax(AnthropicFriendlyProvider):
    api_key = gpt_settings.minimax_api_key
    chat_ready = True
    tts_ready = True
    video_generation_ready = True
    moderation_ready = True

    name = "Minimax"
    base_url = "https://api.minimax.io/anthropic"
    default_model = "MiniMax-M3"
    default_moderation_model = "MiniMax-M2.5"
    model_name_keywords = ["MiniMax"]

    base_tts_url = "https://api.minimax.io/v1/"
    base_video_url = "https://api.minimax.io/v2/"
    default_tts_model = "speech-2.8-turbo"
    default_tts_voice = "Korean_HaughtyLady"
    default_image_model = "image-01"
    default_video_model = "MiniMax-H3"
    default_music_model = "music-3.0"

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    @property
    def client(self) -> AsyncClient:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token, base_url=self.base_url)
        return self._client

    async def get_available_models(
        self, image_generation: bool = False, video_generation: bool = False
    ) -> list[ModelChangeSchema]:
        if image_generation:
            image_models = [
                ModelChangeSchema(provider=self.name, name="image-01", display_name="Image-01", image_generation=True)
            ]

            return self.filter_and_return_list_of_models(models=image_models, image_generation=image_generation)

        if video_generation:
            video_models = self._declared_video_models()
            return self.filter_and_return_list_of_models(models=video_models, video_generation=video_generation)

        models = await super().get_available_models(video_generation=video_generation)
        if not models:
            # Get models endpoint sometimes returns empty list, so we need a hacky fallback here
            supported_models = [
                "MiniMax-M3",
                "MiniMax-M2.7",
                "MiniMax-M2.7-highspeed",
                "MiniMax-M2.5",
                "MiniMax-M2.5-highspeed",
            ]
            models = [
                ModelChangeSchema(
                    provider=self.name,
                    name=model_name,
                    display_name=model_name,
                    image_generation=False,
                )
                for model_name in supported_models
            ]
        return self.filter_and_return_list_of_models(
            models=models, image_generation=image_generation, video_generation=video_generation
        )

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.tts_voice
        model = model or self.tts_model

        logger.info(f"Recording a voice message with model {model}...")

        url = f"{self.base_tts_url}t2a_v2"

        data = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice,
                "emotion": "happy",
                "speed": 1.2,
            },
        }
        try:
            response = await self._request(method="POST", url=url, data=data)
        except Exception as e:
            logger.error(f"Failed to get available models for provider {self.name} due to exception: {e}")
            return bytes()
        response_data = response.json()["data"]
        return bytes.fromhex(response_data["audio"])

    async def generate_music(self, prompt: str, model: str | None = None) -> bytes:
        model = model or self.default_music_model

        logger.info(f"Generating music with model {model}...")

        url = f"{self.base_tts_url}music_generation"

        data = {
            "model": model,
            "prompt": prompt,
            "is_instrumental": True,
        }
        response = await self._request(method="POST", url=url, data=data)
        response_data = response.json()["data"]
        return bytes.fromhex(response_data["audio"])

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
                does not support.
        """
        if images:
            raise ServiceResponseError(
                provider=self.name,
                model=model or self.default_image_model,
                detail="This provider does not support image input for image generation.",
            )
        url = "https://api.minimax.io/v1/image_generation"
        response = await self._request(
            method="POST",
            url=url,
            data={
                "model": model,
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "response_format": "url",
                "n": gpt_settings.image_n_choices,
                "prompt_optimizer": True,
            },
        )
        response_data = response.json()
        images_urls = response_data.get("data", {}).get("image_urls", [])
        return images_urls

    async def _normalize_video_model(self, model: str) -> str:
        """Resolve a video model shorthand to a declared video model name.

        Args:
            model: Requested video model name, possibly a display-name shorthand
                such as ``H3`` instead of ``MiniMax-H3``.

        Returns:
            The declared video model name accepted by the /v2 API.

        Raises:
            ServiceResponseError: If the model does not match any declared video model.
        """
        declared = [m.name for m in self._declared_video_models()]
        if model in declared:
            return model
        lowered = model.lower()
        for name in declared:
            if name.lower().endswith(lowered):
                return name
        supported = ", ".join(declared)
        raise ServiceResponseError(
            provider=self.name,
            model=model,
            detail=f"Unsupported video model {model!r}. Supported models: {supported}",
        )

    def _declared_video_models(self) -> list[ModelChangeSchema]:
        """Build the full catalog of video models supported by the MiniMax /v2 API.

        Returns:
            The declared video model schemas, unfiltered by model whitelists.
        """
        return [
            ModelChangeSchema(
                provider=self.name,
                name="MiniMax-H3",
                display_name="MiniMax H3",
                image_generation=False,
                video_generation=True,
            ),
            ModelChangeSchema(
                provider=self.name,
                name="MiniMax-H3-Max",
                display_name="MiniMax H3 Max",
                image_generation=False,
                video_generation=True,
            ),
        ]

    async def get_videos(
        self,
        model: str | None,
        prompt: str,
        image: bytes | str | None = None,
        duration: int | None = None,
        **kwargs: Any,
    ) -> VideoResult:
        """Generate a text-to-video clip via the MiniMax Hailuo video API.

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
            ServiceResponseError: If the requested video model is not supported.
        """
        if image is not None:
            raise NotImplementedError(f"{self.name} video generation supports text-to-video only for now")

        model = model or self.default_video_model
        if not model:
            raise NoModelSelectedError(provider=self.name, detail="No video generation model selected")
        model = await self._normalize_video_model(model)

        # The V2 API requires resolution, duration and (for text-to-video) a
        # non-adaptive ratio.
        requested_duration = int(duration) if duration is not None else 5
        requested_duration = max(4, min(15, requested_duration))
        payload = {
            "model": model,
            "content": [{"type": "text", "text": prompt}],
            "resolution": "768P",
            "duration": requested_duration,
            "ratio": "16:9",
        }

        started = monotonic()
        response = await self._request(
            method="POST", url=urljoin(self.base_video_url, "video_generation"), data=payload
        )
        task_id = response.json().get("task_id")
        if not task_id:
            raise NoResponseError(provider=self.name, model=model, detail="Server returned no task id")

        logger.info(f"[Video] {self.name}: submitted {model} task {task_id} (duration={requested_duration}s)")

        deadline = monotonic() + self.video_poll_timeout
        task: dict[str, Any] = {}
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
            poll_response = await self._request(
                method="GET", url=urljoin(self.base_video_url, f"query/video_generation/{task_id}")
            )
            task = poll_response.json().get("task") or {}
            status = task.get("status")
            poll_attempts += 1
            logger.debug(
                f"[Video] {self.name}: task {task_id} poll #{poll_attempts} "
                f"status={status} elapsed={monotonic() - started:.0f}s"
            )
            if status == "succeeded":
                break
            if status in ("failed", "cancelled"):
                logger.error(f"[Video] {self.name}: task {task_id} finished with status {status}")
                raise ServiceResponseError(
                    provider=self.name,
                    model=model,
                    detail=f"Video generation task {task_id} finished with status {status}",
                )

        video_url = (task.get("content") or {}).get("url")
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

        reported_duration = task.get("duration")
        clip_duration = int(reported_duration) if reported_duration else requested_duration
        logger.info(
            f"[Video] {self.name}: task {task_id} completed in {monotonic() - started:.0f}s "
            f"duration={clip_duration} size={len(video)} bytes"
        )
        return VideoResult(
            video=video,
            thumbnail=None,
            duration=clip_duration,
        )
