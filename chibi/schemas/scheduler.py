"""Schemas for the internal job scheduler."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class JobInfo(BaseModel):
    """Summary information about a scheduled job."""

    job_id: str = Field(description="Fully qualified job identifier, e.g. `system:retention_cleanup`.")
    trigger: str = Field(description="String representation of the APScheduler trigger.")
    next_run_time: datetime | None = Field(
        default=None, description="Next scheduled run time, or None when the job is paused."
    )
    kwargs_summary: str | None = Field(
        default=None, description="Truncated single-line summary of the job keyword arguments."
    )


class SelfActionPayload(BaseModel):
    """Action that wakes the agent itself with a scheduler trigger."""

    type: Literal["self"] = Field(description="Action type discriminator: wake the agent in the fixed thread.")
    trigger_text: str = Field(description="Text injected into the thread as the third-type scheduler trigger message.")


class NotifyActionPayload(BaseModel):
    """Action that sends a static message to the user without waking the agent."""

    type: Literal["notify"] = Field(description="Action type discriminator: static user notification, zero LLM usage.")
    message: str = Field(description="Static message text delivered to the chat fixed in the job context.")


class CommandActionPayload(BaseModel):
    """Action that runs a shell command and reports the result to the user."""

    type: Literal["command"] = Field(description="Action type discriminator: run a pre-moderated shell command.")
    command: str = Field(description="Shell command to execute on every job run.")
    cwd: str | None = Field(default=None, description="Working directory for the command, or None for the default.")
    timeout_seconds: int = Field(
        default=60,
        gt=0,
        description=(
            "Per-job command timeout in seconds. The dispatcher clamps it from above with "
            "`scheduler_command_timeout_max`."
        ),
    )


AgentJobActionPayload = Annotated[
    SelfActionPayload | NotifyActionPayload | CommandActionPayload,
    Field(discriminator="type"),
]


class AgentJobPayload(BaseModel):
    """Full payload of an agent-created scheduler job stored in the job kwargs."""

    job_id: str = Field(description="Fully qualified job identifier, e.g. `agent:134604548:daily-report`.")
    user_id: int = Field(
        description=(
            "Integer id of the job owner (namespace owner `agent:{user_id}:*`). Positive for Telegram users; "
            "IDE/stdio sessions use the negative reserved `IDE_STORAGE_ID`."
        )
    )
    thread_id: int = Field(
        description=(
            "Message thread the job lives in (0 for non-threaded chats). Mandatory for all jobs. "
            "For IDE/stdio sessions this is the client-minted session thread id."
        )
    )
    storage_id: int = Field(
        description=(
            "Storage key of the conversation the job belongs to (user_id for private chats, "
            "chat_id for groups/forum topics). IDE/stdio sessions use the negative reserved `IDE_STORAGE_ID`."
        )
    )
    chat_id: int = Field(
        description=(
            "Chat the job delivers its output to. Positive for Telegram chats; "
            "IDE/stdio sessions use the negative reserved `IDE_STORAGE_ID`."
        )
    )
    title: str = Field(description="Human-readable job name shown to the user.")
    action: AgentJobActionPayload = Field(description="Action performed on every job run.")
