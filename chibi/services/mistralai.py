from httpx import HTTPStatusError

from chibi.config import gpt_settings
from chibi.schemas.app import ChatResponseSchema
from chibi.schemas.mistralai import ChatCompletionSchema, GetModelsResponseSchema
from chibi.services.provider import Provider
from chibi.types import ChatCompletionMessageSchema


class MistralAI(Provider):
    name = "MistralAI"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        timeout: int = gpt_settings.timeout,
    ) -> ChatResponseSchema:
        url = "https://api.mistral.ai/v1/chat/completions"

        system_message = {"role": "system", "content": gpt_settings.assistant_prompt}
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

    async def get_available_models(self) -> list[str]:
        url = "https://api.mistral.ai/v1/models"

        response = await self._request(method="GET", url=url)

        response_data = GetModelsResponseSchema(**response.json())

        if gpt_settings.models_whitelist:
            allowed_model_names = [
                model.id for model in response_data.data if model.id in gpt_settings.models_whitelist
            ]
        else:
            allowed_model_names = [model.id for model in response_data.data]

        return sorted(allowed_model_names)

    async def api_key_is_valid(self) -> bool:
        try:
            await self.get_available_models()
        except HTTPStatusError:
            return False
        except Exception:
            raise
        return True
