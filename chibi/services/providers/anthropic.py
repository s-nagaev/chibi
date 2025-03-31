from chibi.config import gpt_settings
from chibi.schemas.anthropic import ChatCompletionSchema
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.providers.provider import RestApiFriendlyProvider
from chibi.types import ChatCompletionMessageSchema


class Anthropic(RestApiFriendlyProvider):
    name = "Anthropic"
    model_name_keywords = ["claude"]
    default_model = "claude-3-7-sonnet-20250219"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
        }

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        system_prompt: str,
    ) -> ChatResponseSchema:
        url = "https://api.anthropic.com/v1/messages"

        data = {"model": model, "messages": messages, "system": system_prompt, "max_tokens": self.max_tokens}
        response = await self._request(method="POST", url=url, data=data)

        response_data = ChatCompletionSchema(**response.json())
        choices = response_data.content
        if choices:
            answer_data = choices[0]
            answer = answer_data.text
            usage = UsageSchema(
                completion_tokens=response_data.usage.output_tokens,
                prompt_tokens=response_data.usage.input_tokens,
                total_tokens=response_data.usage.output_tokens + response_data.usage.input_tokens,
            )
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage)

    async def get_available_models(self, image_generation: bool = False) -> list[str]:
        if image_generation:
            return []

        url = "https://api.anthropic.com/v1/models"
        response = await self._request(method="GET", url=url)
        response_data = response.json().get("data", [])
        all_models = [model.get("id") for model in response_data if model.get("id") and model.get("type") == "model"]

        if gpt_settings.models_whitelist:
            allowed_model_names = [model for model in all_models if model in gpt_settings.models_whitelist]
        else:
            allowed_model_names = all_models

        return sorted(allowed_model_names)
