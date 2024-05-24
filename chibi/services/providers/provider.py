import abc
from abc import ABC

import httpx
from httpx import Response
from httpx._types import RequestData
from loguru import logger

from chibi.config import gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NotAuthorizedError,
    ServiceRateLimitError,
    ServiceResponseError,
)
from chibi.schemas.app import ChatResponseSchema
from chibi.types import ChatCompletionMessageSchema


class Provider(ABC):
    name: str

    def __init__(self, token: str, user=None) -> None:
        self.token = token
        self.active_model = user.model if user else None

    @property
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    async def _request(self, method: str, url: str, data: RequestData | None = None) -> Response:
        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        transport = httpx.AsyncHTTPTransport(retries=gpt_settings.retries, proxy=gpt_settings.proxy)

        try:
            async with httpx.AsyncClient(
                transport=transport, timeout=gpt_settings.timeout, proxy=gpt_settings.proxy
            ) as client:
                response = await client.request(method=method, url=url, json=data, headers=self._headers)
        except Exception as e:
            logger.error(f"An error occurred while calling the {self.name} API: {e}")
            raise ServiceResponseError(provider=self.name, detail=str(e))

        if response.status_code == 200:
            return response

        logger.error(
            f"Unexpected response from {self.name} API. Status code: {response.status_code}. " f"Data: {response.text}"
        )
        if response.status_code == 401:
            raise NotAuthorizedError(provider=self.name)
        if response.status_code == 429:
            raise ServiceRateLimitError(provider=self.name)
        raise ServiceResponseError(provider=self.name)

    @abc.abstractmethod
    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str,
        temperature: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        system_prompt: str,
        timeout: int,
    ) -> ChatResponseSchema:
        ...

    async def get_chat_response(
        self,
        messages: list[ChatCompletionMessageSchema],
        model: str | None = None,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        system_prompt: str = gpt_settings.assistant_prompt,
        timeout: int = gpt_settings.timeout,
    ) -> ChatResponseSchema:
        model = model or self.active_model
        if not model:
            raise ValueError("No active model provided!")

        return await self._get_chat_completion_response(
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            presence_penalty=presence_penalty,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout=timeout,
        )

    @abc.abstractmethod
    async def get_available_models(self) -> list[str]:
        ...

    @abc.abstractmethod
    async def api_key_is_valid(self) -> bool:
        ...
