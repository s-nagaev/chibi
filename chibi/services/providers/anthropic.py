from httpx import HTTPStatusError

from chibi.config import gpt_settings
from chibi.schemas.anthropic import ChatCompletionSchema
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.services.providers.provider import Provider
from chibi.types import ChatCompletionMessageSchema


class Anthropic(Provider):
    name = "Anthropic"

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
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        system_prompt: str = gpt_settings.assistant_prompt,
        timeout: int = gpt_settings.timeout,
    ) -> ChatResponseSchema:
        url = "https://api.anthropic.com/v1/messages"

        data = {"model": model, "messages": messages, "system": system_prompt, "max_tokens": max_tokens}
        response = await self._request(method="POST", url=url, data=data)

        response_data = ChatCompletionSchema(**response.json())
        choices = response_data.content
        if choices:
            answer_data = choices[0]
            answer = answer_data.text
            print(response_data)
            usage = UsageSchema(
                completion_tokens=response_data.usage.output_tokens,
                prompt_tokens=response_data.usage.input_tokens,
                total_tokens=response_data.usage.output_tokens + response_data.usage.input_tokens,
            )
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage)

    async def get_available_models(self) -> list[str]:
        all_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307",
            "claude-2.1'",
            "claude-2.0",
            "claude-instant-1.2",
        ]

        if gpt_settings.models_whitelist:
            allowed_model_names = [model for model in all_models if model in gpt_settings.models_whitelist]
        else:
            allowed_model_names = all_models

        return sorted(allowed_model_names)

    async def api_key_is_valid(self) -> bool:
        try:
            await self.get_chat_response(
                messages=[{"role": "user", "content": "ping"}], max_tokens=5, model="claude-3-haiku-20240307"
            )
        except HTTPStatusError:
            return False
        except Exception:
            raise
        return True
