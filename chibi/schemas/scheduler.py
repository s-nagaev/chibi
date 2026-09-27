"""Schemas for the internal job scheduler."""

from datetime import datetime

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
