import logging
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)


class GPTSettings(BaseSettings):
    api_key: str = Field(env="OPENAI_API_KEY")

    assistant_prompt: str = Field(
        env="ASSISTANT_PROMPT",
        default="You're helpful and friendly assistant. Your name is Chibi",
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
            return cls.json_loads(raw_val)

    @property
    def messages_ttl(self) -> int:
        return self.max_conversation_age_minutes * 60


class TelegramSettings(BaseSettings):
    token: str = Field(env="TELEGRAM_BOT_TOKEN")

    bot_name: str = Field(env="BOT_NAME", default="Chibi")
    groups_whitelist: Optional[list[int]] = Field(env="GROUPS_WHITELIST", default=None)
    message_for_disallowed_users: str = Field(
        env="MESSAGE_FOR_DISALLOWED_USERS",
        default="You're not allowed to interact with me, sorry. Contact my owner first, please.",
    )
    proxy: Optional[str] = Field(env="PROXY", default=None)
    users_whitelist: Optional[list[str]] = Field(env="USERS_WHITELIST", default=None)

    class Config:
        env_file = ".env"

        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            if field_name == "users_whitelist":
                return [str(username).strip().strip("@") for username in raw_val.split(",")]
            if field_name == "groups_whitelist":
                return [int(group_id) for group_id in raw_val.split(",")]
            return cls.json_loads(raw_val)


class ApplicationSettings(BaseSettings):
    redis: Optional[str] = Field(env="REDIS", default=None)
    local_data_path: str = Field(env="LOCAL_DATA_PATH", default="/app/data")

    class Config:
        env_file = ".env"


@lru_cache()
def _get_gpt_settings() -> GPTSettings:
    logger.info("Loading config settings from the environment...")
    return GPTSettings()


@lru_cache()
def _get_telegram_settings() -> TelegramSettings:
    logger.info("Loading config settings from the environment...")
    return TelegramSettings()


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    logger.info("Loading config settings from the environment...")
    return ApplicationSettings()


gpt_settings = _get_gpt_settings()
telegram_settings = _get_telegram_settings()
application_settings = _get_application_settings()
