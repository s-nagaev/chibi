from abc import ABC

from chibi.config import gpt_settings


class Provider(ABC):
    def __init__(self, token: str, user=None) -> None:
        self.token = token
        self.active_model = user.model if user else None

    async def _get_chat_completion_response(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        presence_penalty: float,
        frequency_penalty: float,
        timeout: int,
    ) -> tuple[str, dict[str, int] | None]:
        ...

    async def get_chat_response(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        timeout: int = gpt_settings.timeout,
    ) -> tuple[str, dict[str, int] | None]:
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
            timeout=timeout,
        )

    async def get_available_models(self) -> list[str]:
        ...

    async def api_key_is_valid(self) -> bool:
        ...
