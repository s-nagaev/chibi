from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from chibi.config.telegram import telegram_settings
from chibi.constants import IMAGE_ASPECT_RATIO_LITERAL, IMAGE_SIZE_LITERAL


class GPTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    alibaba_key: str | None = Field(alias="ALIBABA_API_KEY", default=None)
    anthropic_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)
    deepseek_key: str | None = Field(alias="DEEPSEEK_API_KEY", default=None)
    gemini_key: str | None = Field(alias="GEMINI_API_KEY", default=None)
    grok_key: str | None = Field(alias="GROK_API_KEY", default=None)
    mistralai_key: str | None = Field(alias="MISTRALAI_API_KEY", default=None)
    openai_key: str | None = Field(alias="OPENAI_API_KEY", default=None)

    assistant_prompt: str = Field(
        default=f"You're helpful and friendly assistant. Your name is {telegram_settings.bot_name}",
    )
    frequency_penalty: float = Field(default=0)
    image_generations_monthly_limit: int = Field(alias="IMAGE_GENERATIONS_LIMIT", default=0)
    image_n_choices: int = Field(default=1)
    image_quality: Literal["standard", "hd"] = Field(default="standard")
    image_size: IMAGE_SIZE_LITERAL = Field(default="1024x1024")
    image_aspect_ratio: IMAGE_ASPECT_RATIO_LITERAL = Field(default="16:9")
    max_conversation_age_minutes: int = Field(default=60)
    max_history_tokens: int = Field(default=10240)
    max_tokens: int = Field(default=4096)
    presence_penalty: float = Field(default=0)
    proxy: str | None = Field(default=None)
    public_mode: bool = Field(default=False)
    retries: int = Field(default=3)
    temperature: float = Field(default=1)
    timeout: int = Field(default=600)
    models_whitelist_raw: str | None = Field(alias="MODELS_WHITELIST", default=None)
    image_generations_whitelist_raw: str | None = Field(alias="IMAGE_GENERATIONS_WHITELIST", default=None)
    model_default: str | None = Field(default=None)

    @property
    def models_whitelist(self) -> list[str]:
        return [x.strip() for x in self.models_whitelist_raw.split(",")] if self.models_whitelist_raw else []

    @property
    def image_generations_whitelist(self) -> list[str]:
        return (
            [x.strip() for x in self.image_generations_whitelist_raw.split(",")]
            if self.image_generations_whitelist_raw
            else []
        )

    @property
    def messages_ttl(self) -> int:
        return self.max_conversation_age_minutes * 60


@lru_cache()
def _get_gpt_settings() -> GPTSettings:
    return GPTSettings()


gpt_settings = _get_gpt_settings()
