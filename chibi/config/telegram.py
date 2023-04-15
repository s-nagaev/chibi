from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseSettings, Field


class TelegramSettings(BaseSettings):
    token: str = Field(env="TELEGRAM_BOT_TOKEN")

    allow_bots: bool = Field(env="ALLOW_BOTS", default=False)
    answer_direct_messages_only: bool = Field(env="ANSWER_DIRECT_MESSAGES_ONLY", default=True)
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
            return cls.json_loads(raw_val)  # type: ignore


@lru_cache()
def _get_telegram_settings() -> TelegramSettings:
    return TelegramSettings()


telegram_settings = _get_telegram_settings()
