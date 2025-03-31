from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        extra="ignore",
    )
    redis: str | None = Field(default=None)
    redis_password: str | None = Field(default=None)
    local_data_path: str = Field(default="/app/data")
    log_prompt_data: bool = Field(default=False)
    hide_models: bool = Field(default=False)
    hide_imagine: bool = Field(default=False)


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    return ApplicationSettings()


application_settings = _get_application_settings()
