from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, HttpUrl
from pydantic.alias_generators import to_camel


class SunoBaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class SunoTrackSchema(SunoBaseModel):
    id: str
    audio_url: HttpUrl | str = ""
    source_audio_url: HttpUrl | None = None
    stream_audio_url: HttpUrl | None = None
    source_stream_audio_url: HttpUrl | None = None
    image_url: HttpUrl | None = None
    source_image_url: HttpUrl | None = None
    prompt: str
    model_name: str
    title: str
    tags: str
    create_time: int | None = None  # Milliseconds timestamp
    duration: float | None = None


class TaskResponseSchema(SunoBaseModel):
    task_id: str
    suno_data: list[SunoTrackSchema]


class TaskDataSchema(SunoBaseModel):
    task_id: str


class StartedTaskDataSchema(TaskDataSchema):
    parent_music_id: str | None = None
    param: str
    response: TaskResponseSchema | None = None
    status: str
    type: str
    operation_type: str
    error_code: int | None = None
    error_message: str | None = None
    create_time: int | None = None

    @property
    def parsed_param(self) -> dict[str, Any]:
        try:
            return json.loads(self.param)
        except (ValueError, TypeError):
            return {}


class SunoAPIResponseSchema(SunoBaseModel):
    code: int
    msg: str

    @property
    def is_success(self) -> bool:
        return self.code == 200


class SunoGetGenerationRequestSchema(SunoAPIResponseSchema):
    data: TaskDataSchema | None = None


class SunoGetGenerationDetailsSchema(SunoAPIResponseSchema):
    data: StartedTaskDataSchema | None = None

    @property
    def is_in_progress(self) -> bool:
        if not self.data:
            return True
        return self.data.status not in (
            "SUCCESS",
            "CREATE_TASK_FAILED",
            "GENERATE_AUDIO_FAILED",
            "SENSITIVE_WORD_ERROR",
        )
