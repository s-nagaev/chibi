from functools import lru_cache

from pydantic import BaseSettings, Field


class ApplicationSettings(BaseSettings):
    redis: str | None = Field(env="REDIS", default=None)
    redis_password: str | None = Field(env="REDIS_PASSWORD", default=None)
    local_data_path: str = Field(env="LOCAL_DATA_PATH", default="/app/data")
    log_prompt_data: bool = Field(env="LOG_PROMPT_DATA", default=False)

    class Config:
        env_file = ".env"


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    return ApplicationSettings()


application_settings = _get_application_settings()
