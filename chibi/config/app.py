from functools import lru_cache
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """
    Application settings loaded from environment or .env file.

    Attributes:
        redis: Redis connection URL.
        redis_password: Password for Redis.
        aws_region: AWS region for DynamoDB.
        aws_access_key_id: AWS access key ID.
        aws_secret_access_key: AWS secret access key.
        ddb_users_table: DynamoDB table name for users.
        ddb_messages_table: DynamoDB table name for messages.
        local_data_path: Filesystem path for local storage.
        log_prompt_data: Whether to log prompt data.
        hide_models: Hide model options in UI.
        hide_imagine: Hide imagine commands.
        heartbeat_url: URL for heartbeat check.
        heartbeat_frequency_call: Interval between heartbeat calls.
        heartbeat_retry_calls: Number of retries for heartbeat.
        heartbeat_proxy: Proxy URL for heartbeat.
    """

    model_config = SettingsConfigDict(
        env_file=(".env",),
        extra="ignore",
    )

    # Redis settings
    redis: str | None = Field(default=None)
    redis_password: str | None = Field(default=None)

    # DynamoDB settings
    aws_region: str | None = Field(default=None)
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    ddb_users_table: str | None = Field(default=None)
    ddb_messages_table: str | None = Field(default=None)

    # Local storage settings
    local_data_path: str = Field(default="/app/data")

    # Other settings
    log_prompt_data: bool = Field(default=False)
    hide_models: bool = Field(default=False)
    hide_imagine: bool = Field(default=False)
    heartbeat_url: str | None = Field(default=None)
    heartbeat_frequency_call: int = Field(default=30)
    heartbeat_retry_calls: int = Field(default=3)
    heartbeat_proxy: str | None = Field(default=None)

    @property
    def storage_backend(self) -> Literal["local", "redis", "dynamodb"]:
        if self.redis:
            return "redis"
        if self.aws_access_key_id and self.aws_secret_access_key:
            return "dynamodb"
        return "local"


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    return ApplicationSettings()


logger.level("TOOL", no=20, color="<light-blue>")
logger.level("THINK", no=20, color="<light-magenta>")
logger.level("CALL", no=20, color="<magenta>")
logger.level("CHECK", no=20, color="<light-red>")
logger.level("SUBAGENT", no=20, color="<cyan>")
logger.level("DELEGATE", no=20, color="<blue>")


application_settings = _get_application_settings()
