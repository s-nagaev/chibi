from asyncio import sleep

from loguru import logger
from tenacity import retry, retry_if_result, stop_after_attempt, wait_fixed

from chibi.config import gpt_settings
from chibi.exceptions import ServiceResponseError
from chibi.schemas.app import ModelChangeSchema
from chibi.schemas.suno import SunoGetGenerationDetailsSchema
from chibi.services.providers.constants.suno import POLLING_ATTEMPTS_MAX, POLLING_ATTEMPTS_WAIT_BETWEEN
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.services.providers.utils import suno_task_still_processing


class Suno(RestApiFriendlyProvider):
    name = "Suno"

    api_key = gpt_settings.suno_key
    chat_ready = False
    music_ready = True
    base_url = "https://api.sunoapi.org"
    default_model = "V5"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        return []

    @retry(
        wait=wait_fixed(POLLING_ATTEMPTS_WAIT_BETWEEN),
        stop=stop_after_attempt(POLLING_ATTEMPTS_MAX),
        retry=retry_if_result(suno_task_still_processing),
    )
    async def poll_result(self, task_id: int | str) -> SunoGetGenerationDetailsSchema:
        logger.info(f"[Suno] Checking status of the task #{task_id}...")
        url = f"{self.base_url}/api/v1/generate/record-info"
        params = {"taskId": str(task_id)}
        response = await self._request(method="GET", url=url, params=params)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("code") != 200:
            raise ServiceResponseError(
                provider=self.name,
                detail=(
                    f"Unsuccessful status code response while checking status of the task {task_id}: {response_data}"
                ),
            )
        return SunoGetGenerationDetailsSchema.model_validate(response_data)

    async def order_music_generation(
        self,
        prompt: str,
        instrumental_only: bool = False,
        model: str = default_model,
    ) -> int:
        url = f"{self.base_url}/api/v1/generate"

        response = await self._request(
            method="POST",
            url=url,
            data={
                "customMode": False,
                "instrumental": instrumental_only,
                "model": model,
                "prompt": prompt,
                "callBackUrl": False,
            },
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("code") != 200:
            raise ServiceResponseError(provider=self.name, detail=f"Unsuccessful status code response: {response_data}")

        task_id = response_data.get("data", {}).get("taskId")
        if not task_id:
            raise ServiceResponseError(provider=self.name, detail=f"Unsuccessful status code response: {response_data}")
        return task_id

    async def order_music_generation_advanced_mode(
        self,
        prompt: str,
        style: str,
        title: str,
        instrumental_only: bool = False,
        model: str = default_model,
        negative_tags: str | None = None,
        vocal_gender: str | None = None,
        style_weight: float = 0.5,
        weirdness_constraint: float = 0.5,
    ) -> int:
        url = f"{self.base_url}/api/v1/generate"
        response = await self._request(
            method="POST",
            url=url,
            data={
                "customMode": True,
                "instrumental": instrumental_only,
                "model": model,
                "prompt": prompt,
                "callBackUrl": False,
                "style": style,
                "title": title,
                "negativeTags": negative_tags,
                "vocalGender": vocal_gender,
                "styleWeight": style_weight,
                "weirdnessConstraint": weirdness_constraint,
            },
        )
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("code") != 200:
            raise ServiceResponseError(provider=self.name, detail=f"Unsuccessful status code response: {response_data}")

        task_id = response_data.get("data", {}).get("taskId")
        if not task_id:
            raise ServiceResponseError(provider=self.name, detail=f"Unsuccessful status code response: {response_data}")
        return task_id

    async def generate_music(
        self,
        prompt: str,
        instrumental_only: bool = False,
        model: str = default_model,
    ) -> SunoGetGenerationDetailsSchema:
        task_id = await self.order_music_generation(prompt=prompt, instrumental_only=instrumental_only, model=model)
        await sleep(POLLING_ATTEMPTS_WAIT_BETWEEN)
        music_data = await self.poll_result(task_id=task_id)
        return music_data
