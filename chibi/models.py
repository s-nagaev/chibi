import time

from pydantic import BaseModel, Field

from chibi.config import gpt_settings
from chibi.exceptions import NoApiKeyProvidedError
from chibi.services.anthropic import Anthropic
from chibi.services.mistralai import MistralAI
from chibi.services.openai import OpenAI
from chibi.services.provider import Provider


class Message(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    role: str
    content: str
    expire_at: float | None = None


class ImageMeta(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    expire_at: float


class User(BaseModel):
    id: int
    anthropic_token: str | None = gpt_settings.anthropic_key
    openai_token: str | None = gpt_settings.openai_key
    mistralai_token: str | None = gpt_settings.mistralai_key
    gpt_model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    images: list[ImageMeta] = Field(default_factory=list)

    @property
    def api_token(self) -> str | None:
        return None

    @api_token.setter
    def api_token(self, value: str) -> None:
        if value.startswith("sk-"):
            self.openai_token = value
        self.mistralai_token = value

    @property
    def model(self) -> str:
        """Get user's preferred model.

        If user never set the preferred model, returns the default one.
        If user's preferred model is not compatible with the models whitelist, returns the default one.
        If none of the previous conditions are met, returns user's preferred model.

        Returns:
            The GPT model name.
        """
        if not self.gpt_model:
            return gpt_settings.model_default

        if not gpt_settings.models_whitelist:
            return self.gpt_model

        if self.gpt_model in gpt_settings.models_whitelist:
            return self.gpt_model

        return gpt_settings.model_default

    @property
    def active_provider(self) -> Provider:
        if ("mistral" in self.model or "mixtral" in self.model) and self.mistralai_token is not None:
            return MistralAI(token=self.mistralai_token, user=self)
        if "claude" in self.model and self.anthropic_token:
            return Anthropic(token=self.anthropic_token, user=self)
        if "gpt" in self.model and self.openai_token:
            return OpenAI(token=self.openai_token, user=self)
        raise NoApiKeyProvidedError(provider="Any")

    @property
    def available_providers(self) -> list[Provider]:
        providers: list[Provider] = []
        if self.anthropic_token:
            providers.append(Anthropic(user=self, token=self.anthropic_token))
        if self.mistralai_token:
            providers.append(MistralAI(user=self, token=self.mistralai_token))
        if self.openai_token:
            providers.append(OpenAI(user=self, token=self.openai_token))
        return providers

    @property
    def openai(self) -> OpenAI | None:
        if self.openai_token:
            return OpenAI(token=self.openai_token, user=self)
        return None

    async def get_available_models(self) -> list[str]:
        models = []
        for provider in self.available_providers:
            models.extend(await provider.get_available_models())
        return models

    @property
    def has_reached_image_limits(self) -> bool:
        if not gpt_settings.image_generations_monthly_limit:
            return False
        if str(self.id) in gpt_settings.image_generations_whitelist:
            return False
        return len(self.images) >= gpt_settings.image_generations_monthly_limit
