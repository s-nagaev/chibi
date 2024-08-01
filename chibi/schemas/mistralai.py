from pydantic import BaseModel, Field


class ModelPermissionSchema(BaseModel):
    id: str
    object: str
    created: int
    allow_create_engine: bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: str | None
    is_blocking: bool


class ModelDataSchema(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class GetModelsResponseSchema(BaseModel):
    object: str
    data: list[ModelDataSchema]


class MessageSchema(BaseModel):
    role: str
    content: str
    tool_calls: str | None = Field(None, alias="tool_calls")


class ChoiceSchema(BaseModel):
    index: int
    message: MessageSchema
    finish_reason: str
    logprobs: str | None = Field(None, alias="logprobs")


class MistralaiUsageSchema(BaseModel):
    prompt_tokens: int
    total_tokens: int
    completion_tokens: int


class ChatCompletionSchema(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChoiceSchema]
    usage: MistralaiUsageSchema
