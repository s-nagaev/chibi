from chibi.config import gpt_settings
from chibi.exceptions import NoResponseError
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider, RestApiFriendlyProvider


class ZhipuAI(OpenAIFriendlyProvider, RestApiFriendlyProvider):
    api_key = gpt_settings.zai_key
    chat_ready = True
    moderation_ready = True

    name = "ZhipuAI"
    model_name_keywords = ["glm"]
    base_url = "https://api.z.ai/api/paas/v4/"
    default_model = "glm-5"
    default_image_model = "glm-image"
    default_moderation_model = "glm-4-32b-0414-128k"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def get_model_display_name(self, model_name: str) -> str:
        return model_name.upper()

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        if image_generation:
            return [
                ModelChangeSchema(provider=self.name, name="glm-image", image_generation=True),
                ModelChangeSchema(provider=self.name, name="cogview-4-250304", image_generation=True),
            ]
        return await super().get_available_models(image_generation=False)

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
