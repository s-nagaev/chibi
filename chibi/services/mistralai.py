import httpx
from httpx import HTTPStatusError, Response
from httpx._types import RequestData
from openai.types.chat import ChatCompletionMessageParam

from chibi.config import gpt_settings
from chibi.schemas.mistralai import (
    ChatCompletionSchema,
    GetModelsResponseSchema,
    UsageSchema,
)
from chibi.services.provider import Provider


class MistralAI(Provider):
    async def _request(self, method: str, url: str, data: RequestData | None = None) -> Response:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        transport = httpx.AsyncHTTPTransport(retries=3, proxy=gpt_settings.proxy)

        async with httpx.AsyncClient(
            transport=transport, timeout=gpt_settings.timeout, proxy=gpt_settings.proxy
        ) as client:
            response = await client.request(method=method, url=url, json=data, headers=headers)
        return response

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        timeout: int = gpt_settings.timeout,
    ) -> tuple[str, UsageSchema | None]:
        url = "https://api.mistral.ai/v1/chat/completions"

        system_message = {"role": "system", "content": gpt_settings.assistant_prompt}
        dialog = [system_message] + [dict(m) for m in messages]
        data = {"model": model, "messages": dialog, "safe_prompt": False}
        response = await self._request(method="POST", url=url, data=data)

        if response.status_code != 200:
            return "", None

        response_data = ChatCompletionSchema(**response.json())
        choices = response_data.choices
        if not choices:
            return "", None

        answer = choices[0]
        usage = response_data.usage
        return answer.message.content, usage

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
