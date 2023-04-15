from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class ApplicationSettings(BaseSettings):
    redis: Optional[str] = Field(env="REDIS", default=None)
    redis_password: Optional[str] = Field(env="REDIS_PASSWORD", default=None)
    local_data_path: str = Field(env="LOCAL_DATA_PATH", default="/app/data")

    class Config:
        env_file = ".env"


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    return ApplicationSettings()


application_settings = _get_application_settings()
