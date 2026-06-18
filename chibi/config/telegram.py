from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        extra="ignore",
    )
    token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")

    telegram_base_url: str = Field(default="https://api.telegram.org/bot")
    telegram_base_file_url: str = Field(default="https://api.telegram.org/file/bot")
    allow_bots: bool = Field(default=False)
    answer_direct_messages_only: bool = Field(default=True)
    bot_name: str = Field(default="Chibi")
    message_for_disallowed_users: str = Field(
        default="You're not allowed to interact with me, sorry. Contact my owner first, please.",
    )
    proxy: str | None = Field(default=None)
    groups_whitelist_raw: str | None = Field(alias="GROUPS_WHITELIST", default=None)

    users_whitelist_raw: str | None = Field(alias="USERS_WHITELIST", default=None)

    @property
    def groups_whitelist(self) -> list[int]:
        return [int(x.strip()) for x in self.groups_whitelist_raw.split(",")] if self.groups_whitelist_raw else []

    @property
    def users_whitelist(self) -> list[str]:
        return (
            [str(x).strip().strip("@") for x in self.users_whitelist_raw.split(",")] if self.users_whitelist_raw else []
        )


@lru_cache()
def _get_telegram_settings() -> TelegramSettings:
    return TelegramSettings()


telegram_settings = _get_telegram_settings()
