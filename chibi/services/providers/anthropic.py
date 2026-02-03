from anthropic import AsyncClient

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError
from chibi.services.providers.provider import AnthropicFriendlyProvider


class Anthropic(AnthropicFriendlyProvider):
    api_key = gpt_settings.anthropic_key
    chat_ready = True
    moderation_ready = True

    name = "Anthropic"
    model_name_keywords = ["claude"]
    default_model = "claude-sonnet-4-5-20250929"
    default_moderation_model = "claude-haiku-4-5-20251001"

    def __init__(self, token: str) -> None:
        self._client: AsyncClient | None = None
        super().__init__(token=token)

    @property
    def client(self) -> AsyncClient:
        if self._client:
            return self._client

        if not self.token:
            raise NoApiKeyProvidedError(provider=self.name)

        self._client = AsyncClient(api_key=self.token)
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
        }
