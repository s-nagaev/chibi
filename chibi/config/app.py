import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ClientType = Literal["telegram", "tui", "vscode", "pycharm", "neovim", "terminal"]


class ApplicationSettings(BaseSettings):
    """
    Application settings loaded from environment or .env file.

    Attributes:
        redis: Redis connection URL.
        redis_password: Password for Redis.
        local_data_path: Filesystem path for local storage. Defaults to the image
            path ``/app/data`` inside a container and to ``~/chibi-bot/data`` on a host runtime that
            configures it nowhere else.
        log_prompt_data: Whether to log prompt data.
        hide_models: Hide model options in UI.
        hide_imagine: Hide imagine commands.
        heartbeat_url: URL for heartbeat check.
        heartbeat_frequency_call: Interval between heartbeat calls.
        heartbeat_retry_calls: Number of retries for heartbeat.
        heartbeat_proxy: Proxy URL for heartbeat.
        client: Client frontend served by this process (telegram, tui, vscode, pycharm, neovim, terminal).
        scheduler_tool_enabled: Whether the agent scheduling tool is enabled.
        scheduler_notify_enabled: Whether the `notify` scheduled action type is allowed.
        scheduler_agent_commands_enabled: Whether shell commands are allowed in scheduled jobs. Enabled by
            default: the agent can schedule itself and run arbitrary commands through its
            terminal tool anyway, so gating only scheduled commands adds no security. Every command is still
            pre-moderated before it is accepted and re-moderated before every run.
        scheduler_command_timeout_max: Upper limit for a scheduled shell command timeout in seconds.
        scheduler_misfire_grace_time: Default grace period in seconds for catching up missed job runs.
        scheduler_failure_notify: Whether the user is notified when a scheduled job fails.
        scheduler_debug_logs: Lower the APScheduler log floor of the stdlib→loguru bridge to DEBUG
            (per-wakeup chatter like ``Next wakeup is due at ...`` and ``Added job ...`` becomes
            visible). Default False keeps an INFO floor; toggling the env var only requires a
            process restart, not a rebuild.
    """

    model_config = SettingsConfigDict(
        env_file=(".env",),
        extra="ignore",
    )

    # Redis settings
    redis: str | None = Field(default=None)
    redis_password: str | None = Field(default=None)

    # Local storage settings
    local_data_path: str = Field(default="/app/data")

    # MCP settings
    enable_mcp_sse: bool = Field(default=True)
    enable_mcp_stdio: bool = Field(default=False)
    enable_mcp_rest: bool = Field(default=False)

    # HeartBeat settings
    heartbeat_url: str | None = Field(default=None)
    heartbeat_frequency_call: int = Field(default=60, ge=30)
    heartbeat_retry_calls: int = Field(default=3)
    heartbeat_proxy: str | None = Field(default=None)

    # InfluxDB settings
    influxdb_url: str | None = Field(default=None)
    influxdb_token: str | None = Field(default=None)
    influxdb_org: str | None = Field(default=None)
    influxdb_bucket: str | None = Field(default=None)

    # ChromaDB settings
    enable_infinite_context: bool = Field(default=False)
    chroma_host: str = Field(default="")
    chroma_port: int = Field(default=8000)
    chroma_history_retention_days: int = Field(default=180)
    memory_search_limit: int = Field(default=5)
    embedding_function: Literal["LOCAL", "OPENAI", "GEMINI", "MISTRALAI", "JINA"] = Field(default="LOCAL")
    embedding_model: str | None = Field(default=None)
    # Batch settings for context retrieval (by token count)
    batch_token_limit: int = Field(default=30000)

    # Interface
    hide_models: bool = Field(default=False)
    hide_imagine: bool = Field(default=False)
    client: ClientType = Field(default="telegram")

    # Other settings
    log_prompt_data: bool = Field(default=False)

    # Scheduler settings
    scheduler_tool_enabled: bool = Field(default=True)
    scheduler_notify_enabled: bool = Field(default=False)
    # Enabled by default: the agent can already run arbitrary commands via its terminal tool,
    # so blocking only *scheduled* commands adds no real protection. Premoderation stays.
    scheduler_agent_commands_enabled: bool = Field(default=True)
    scheduler_command_timeout_max: int = Field(default=900)
    scheduler_misfire_grace_time: int = Field(default=3600)
    scheduler_failure_notify: bool = Field(default=True)
    scheduler_debug_logs: bool = Field(default=False)

    # Agent settings
    home_dir: str = Field(default="~/chibi")  # AI agent's home directory
    working_dir: str = Field(default="~/chibi")  # AI agent's CWD
    skills_dir: str = Field(default=(Path(".") / "skills").absolute().as_posix())  # absolute path to dir with skills

    selected_file_storage: Literal["local", "telegram"] = Field(default="telegram", alias="FILE_STORAGE")

    @property
    def is_influx_configured(self) -> bool:
        return all((self.influxdb_url, self.influxdb_token, self.influxdb_org, self.influxdb_bucket))

    @property
    def storage_backend(self) -> Literal["local", "redis"]:
        if self.redis:
            return "redis"
        return "local"

    @property
    def runtime_environment(self) -> Literal["host", "docker", "podman"]:
        if Path("/.dockerenv").exists():
            return "docker"
        if Path("run/.containerenv").is_file():
            return "podman"
        return "host"

    @property
    def is_chroma_configured(self) -> bool:
        return self.enable_infinite_context

    @property
    def running_in_container(self) -> bool:
        return self.runtime_environment in ("docker", "podman")


def _local_data_path_is_configured() -> bool:
    """Report whether any settings source provides a local data path.

    The environment carries the process variables and the values merged from
    ``~/chibi-bot/settings``; the working directory ``.env`` file is read by
    pydantic-settings and is inspected separately here.

    Returns:
        True when ``LOCAL_DATA_PATH`` is present in the environment or in the
        working directory ``.env`` file.
    """
    for source in (os.environ, dotenv_values(".env")):
        if any(key.lower() == "local_data_path" for key in source):
            return True
    return False


def _apply_host_local_data_fallback(settings: ApplicationSettings) -> None:
    """Point local storage at ``~/chibi-bot/data`` when running on a host.

    The field default ``/app/data`` only exists inside the container image, so a
    bare host install (``pip install chibi-bot``) fails with ``Read-only file
    system: '/app/data'`` as soon as the SQLite job store or the local storage
    backend writes there. Container runtimes keep the image path and an explicit
    ``LOCAL_DATA_PATH`` always wins. The fallback directory is the one the TUI
    installers already provision, and it is created when missing.

    Args:
        settings: Settings object to adjust in place.
    """
    if settings.running_in_container or _local_data_path_is_configured():
        return
    data_path = Path.home() / "chibi-bot" / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    settings.local_data_path = data_path.as_posix()
    logger.info("LOCAL_DATA_PATH is not configured, using {}", data_path)


@lru_cache()
def _get_application_settings() -> ApplicationSettings:
    """Resolve the application settings for the current process.

    ``CHIBI_ENV=dev`` isolates a development run from production
    configuration: the settings file is never read, so neither
    ``~/chibi-bot/settings`` nor a ``CHIBI_SETTINGS_FILE`` path can inject
    values into the process environment and outrank the working directory
    ``.env``. Outside dev the file keeps its historical precedence over the
    process environment.

    Returns:
        The resolved application settings, cached for the process lifetime.
    """
    if os.getenv("CHIBI_ENV", "no-dev").lower() != "dev":
        # The settings file is a production source only. A dev run reads it
        # never, even though stdio model discovery once relied on it: discovery
        # now runs from the dev configuration alone and, like any other failed
        # model listing, degrades gracefully when no provider is configured.
        settings_path = Path(os.getenv("CHIBI_SETTINGS_FILE") or Path.home() / "chibi-bot" / "settings")
        if settings_path.exists():
            overrides = {k: v for k, v in dotenv_values(settings_path).items() if v is not None}
            os.environ.update(overrides)
    settings = ApplicationSettings()
    _apply_host_local_data_fallback(settings)
    return settings


def add_user_context(record) -> bool:
    user_id = record["extra"].get("user_id")
    record["extra"]["user_id"] = f"[{user_id}] " if user_id else ""
    return True


def configure_stderr_sink() -> None:
    """Install the backend stderr loguru sink.

    Every emitted line follows the plain shape ``YYYY-MM-DD HH:MM:SS | LEVEL |
    message`` (local time, no file/line references), optionally prefixed by the
    bound user id. Terminal and stdio runners install their own sinks instead.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "{extra[user_id]}"
        "<level>{message}</level>",
        filter=add_user_context,
    )


configure_stderr_sink()
logger.level("TOOL", no=20, color="<light-blue>")
logger.level("THINK", no=20, color="<light-magenta>")
logger.level("CALL", no=20, color="<magenta>")
logger.level("CHECK", no=20, color="<light-red>")
logger.level("MODERATOR", no=20, color="<light-red>")
logger.level("SUBAGENT", no=20, color="<cyan>")
logger.level("DELEGATE", no=20, color="<blue>")


application_settings = _get_application_settings()
