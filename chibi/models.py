import time
from typing import Any

from pydantic import BaseModel, Field

from chibi.config import gpt_settings
from chibi.exceptions import (
    NoApiKeyProvidedError,
    NoModelSelectedError,
    NoProviderSelectedError,
)
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers import registered_providers
from chibi.services.providers.provider import Provider


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
    alibaba_token: str | None = gpt_settings.alibaba_key
    anthropic_token: str | None = gpt_settings.anthropic_key
    deepseek_token: str | None = gpt_settings.deepseek_key
    gemini_token: str | None = gpt_settings.gemini_key
    mistralai_token: str | None = gpt_settings.mistralai_key
    openai_token: str | None = gpt_settings.openai_key
    tokens: dict[str, str] = {}
    messages: list[Message] = Field(default_factory=list)
    images: list[ImageMeta] = Field(default_factory=list)
    gpt_model: str | None = None  # Deprecated
    selected_gpt_model_name: str | None = None
    selected_gpt_provider_name: str | None = None
    selected_image_model_name: str | None = None
    selected_image_provider_name: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("gpt_model", None) and not kwargs.get("selected_gpt_model_name", None):
            kwargs["selected_gpt_model_name"] = kwargs["gpt_model"]
        super().__init__(**kwargs)

    def get_token(self, provider_name: str) -> str | None:
        if gpt_settings.public_mode:
            return self.tokens.get(provider_name, None)
        return getattr(gpt_settings, f"{provider_name.lower()}_key", None)

    @property
    def model(self) -> str:
        """Get user's active model.

        Returns:
            The GPT model name.
        """
        if self.selected_gpt_model_name:
            return self.selected_gpt_model_name

        if self.selected_gpt_provider_name:
            return self.active_gpt_provider.default_model

        if gpt_settings.model_default:
            return gpt_settings.model_default

        raise NoModelSelectedError

    @property
    def active_image_provider(self) -> Provider:
        if not self.selected_image_provider_name:
            raise NoProviderSelectedError
        token = self.get_token(self.selected_image_provider_name)
        if not token:
            raise NoApiKeyProvidedError(provider="Unset", detail="No API key provided")
        provider = registered_providers[self.selected_image_provider_name]
        return provider(token=token, user=self)

    @property
    def active_gpt_provider(self) -> Provider:
        if not self.selected_gpt_provider_name:
            for provider_name, provider in registered_providers.items():
                if provider.is_chat_ready_model(self.model) and (token := self.get_token(provider_name)):
                    self.selected_gpt_provider_name = provider_name
                    return provider(token=token, user=self)
            raise NoProviderSelectedError

        token = self.get_token(self.selected_gpt_provider_name)
        if not token:
            raise NoApiKeyProvidedError(provider="Unset", detail="No API key provided")
        provider = registered_providers[self.selected_gpt_provider_name]
        return provider(token=token, user=self)

    @property
    def available_providers(self) -> list[Provider]:
        providers_with_token_set: list[Provider] = []

        if not gpt_settings.public_mode:
            for provider_name, personal_provider in registered_providers.items():
                if token := self.get_token(provider_name):
                    providers_with_token_set.append(personal_provider(user=self, token=token))
            return providers_with_token_set

        for provider_name, token in self.tokens.items():
            provider = registered_providers.get(provider_name, None)
            if provider is None:
                continue
            providers_with_token_set.append(provider(user=self, token=token))
        return providers_with_token_set

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        models = []
        for provider in self.available_providers:
            models.extend(
                [
                    ModelChangeSchema(provider=provider.name, name=model, image_generation=image_generation)
                    for model in await provider.get_available_models(image_generation=image_generation)
                ]
            )
        return models

    @property
    def has_reached_image_limits(self) -> bool:
        if not gpt_settings.image_generations_monthly_limit:
            return False
        if str(self.id) in gpt_settings.image_generations_whitelist:
            return False
        return len(self.images) >= gpt_settings.image_generations_monthly_limit
