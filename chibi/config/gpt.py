from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseSettings, Field

from chibi.config.telegram import telegram_settings


class GPTSettings(BaseSettings):
    api_key: str | None = Field(env="OPENAI_API_KEY", default=None)  # Deprecated
    openai_key: str | None = Field(env="OPENAI_API_KEY", default=None)
    anthropic_key: str | None = Field(env="ANTHROPIC_API_KEY", default=None)
    mistralai_key: str | None = Field(env="MISTRALAI_API_KEY", default=None)

    assistant_prompt: str = Field(
        env="ASSISTANT_PROMPT",
        default=f"You're helpful and friendly assistant. Your name is {telegram_settings.bot_name}",
    )
    dall_e_model: Literal["dall-e-2", "dall-e-3"] = Field(env="DALL_E_MODEL", default="dall-e-3")
    frequency_penalty: float = Field(env="FREQUENCY_PENALTY", default=0)
    image_generations_monthly_limit: int = Field(env="IMAGE_GENERATIONS_LIMIT", default=0)
    image_generations_whitelist: list[str] = Field(env="IMAGE_GENERATIONS_WHITELIST", default_factory=list)
    image_n_choices: int = Field(env="OPENAI_IMAGE_N_CHOICES", default=1)
    image_quality: Literal["standard", "hd"] = Field(env="IMAGE_QUALITY", default="standard")
    image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = Field(
        env="IMAGE_SIZE", default="1024x1024"
    )
    max_conversation_age_minutes: int = Field(env="MAX_CONVERSATION_AGE_MINUTES", default=60)
    max_history_tokens: int = Field(env="MAX_HISTORY_TOKENS", default=1800)
    max_tokens: int = Field(env="MAX_TOKENS", default=1000)
    model_default: str = Field(env="MODEL_DEFAULT", default="gpt-3.5-turbo")
    models_whitelist: list[str] = Field(env="MODELS_WHITELIST", default_factory=list)
    presence_penalty: float = Field(env="PRESENCE_PENALTY", default=0)
    proxy: str | None = Field(env="PROXY", default=None)
    public_mode: bool = Field(env="PUBLIC_MODE", default=False)
    retries: int = Field(env="RETRIES", default=3)
    temperature: float = Field(env="TEMPERATURE", default=1)
    timeout: int = Field(env="TIMEOUT", default=240)

    class Config:
        env_file = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name in ("image_generations_whitelist", "models_whitelist"):
                return raw_val.split(",")
            if field_name == "gpt4_whitelist":
                return [str(username).strip().strip("@") for username in raw_val.split(",")]
            return cls.json_loads(raw_val)  # type: ignore

    @property
    def messages_ttl(self) -> int:
        return self.max_conversation_age_minutes * 60


@lru_cache()
def _get_gpt_settings() -> GPTSettings:
    return GPTSettings()


gpt_settings = _get_gpt_settings()
