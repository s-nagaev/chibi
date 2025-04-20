from openai.types.chat import ChatCompletionMessageParam

from chibi.config import gpt_settings
from chibi.schemas.app import ChatResponseSchema
from chibi.schemas.mistralai import ChatCompletionSchema, GetModelsResponseSchema
from chibi.services.providers.provider import RestApiFriendlyProvider


class MistralAI(RestApiFriendlyProvider):
    name = "MistralAI"
    model_name_keywords = ["mistral", "mixtral", "ministral"]
    model_name_keywords_exclude = ["embed", "moderation", "ocr"]
    default_model = "mistral-large-latest"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    async def _get_chat_completion_response(
        self, messages: list[ChatCompletionMessageParam], model: str, system_prompt: str | None = None
    ) -> ChatResponseSchema:
        url = "https://api.mistral.ai/v1/chat/completions"

        system_message = {"role": "system", "content": system_prompt}
        dialog = [system_message] + [dict(m) for m in messages]
        data = {"model": model, "messages": dialog, "safe_prompt": False}
        response = await self._request(method="POST", url=url, data=data)

        if response.status_code != 200:
            raise Exception  # TODO

        response_data = ChatCompletionSchema(**response.json())
        choices = response_data.choices
        if choices:
            answer_data = choices[0]
            answer = answer_data.message.content
            usage = response_data.usage
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage)

    async def get_available_models(self, image_generation: bool = False) -> list[str]:
        if image_generation:
            return []

        url = "https://api.mistral.ai/v1/models"
        response = await self._request(method="GET", url=url)
        response_data = GetModelsResponseSchema(**response.json())

        if gpt_settings.models_whitelist:
            allowed_model_names = [
                model.id for model in response_data.data if model.id in gpt_settings.models_whitelist
            ]
        else:
            allowed_model_names = [model.id for model in response_data.data if self.is_chat_ready_model(model.id)]

        return sorted(allowed_model_names)
