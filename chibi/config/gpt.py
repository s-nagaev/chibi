from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseSettings, Field

from chibi.config.telegram import telegram_settings


class GPTSettings(BaseSettings):
    api_key: Optional[str] = Field(env="OPENAI_API_KEY", default=None)

    assistant_prompt: str = Field(
        env="ASSISTANT_PROMPT",
        default=f"You're helpful and friendly assistant. Your name is {telegram_settings.bot_name}",
    )
    frequency_penalty: float = Field(env="OPENAI_FREQUENCY_PENALTY", default=0)
    gpt4_enabled: bool = Field(env="GPT4_ENABLED", default=True)
    gpt4_whitelist: Optional[list[str]] = Field(env="GPT4_WHITELIST", default=None)
    image_n_choices: int = Field(env="OPENAI_IMAGE_N_CHOICES", default=4)
    image_size: str = Field(env="IMAGE_SIZE", default="512x512")
    max_conversation_age_minutes: int = Field(env="MAX_CONVERSATION_AGE_MINUTES", default=60)
    max_history_tokens: int = Field(env="MAX_HISTORY_TOKENS", default=1800)
    max_tokens: int = Field(env="MAX_TOKENS", default=1000)
    model_default: str = Field(env="MODEL_DEFAULT", default="gpt-3.5-turbo")
    presence_penalty: float = Field(env="OPENAI_PRESENCE_PENALTY", default=0)
    proxy: Optional[str] = Field(env="PROXY", default=None)
    temperature: float = Field(env="OPENAI_TEMPERATURE", default=1)
    timeout: int = Field(env="TIMEOUT", default=15)

    class Config:
        env_file = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
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
