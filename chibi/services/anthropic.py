import httpx
from httpx import HTTPStatusError, Response
from httpx._types import RequestData
from openai.types.chat import ChatCompletionMessageParam

from chibi.config import gpt_settings
from chibi.schemas.anthropic import ChatCompletionSchema, UsageSchema
from chibi.services.provider import Provider


class Anthropic(Provider):
    async def _request(self, method: str, url: str, data: RequestData | None = None) -> Response:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
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
        url = "https://api.anthropic.com/v1/messages"

        data = {"model": model, "messages": messages, "system": gpt_settings.assistant_prompt, "max_tokens": max_tokens}
        response = await self._request(method="POST", url=url, data=data)

        response_data = ChatCompletionSchema(**response.json())
        choices = response_data.content
        if not choices:
            return "", None

        answer = choices[0]
        usage = response_data.usage
        return answer.text, usage

    async def get_available_models(self) -> list[str]:
        all_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
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
