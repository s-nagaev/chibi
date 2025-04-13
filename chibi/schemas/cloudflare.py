from pydantic import BaseModel


class UsageSchema(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AnswerItemSchema(BaseModel):
    response: str
    usage: UsageSchema


class ErroItemSchema(BaseModel):
    code: int
    message: str


class ChatCompletionResponseSchema(BaseModel):
    success: bool
    errors: list[ErroItemSchema]
    result: AnswerItemSchema


class PriceSchema(BaseModel):
    unit: str
    price: float
    currency: str


class PropertySchema(BaseModel):
    property_id: str
    value: str | list[PriceSchema] | None


class TaskSchema(BaseModel):
    id: str
    name: str
    description: str


class ModelDescriptionSchema(BaseModel):
    id: str
    source: int
    name: str
    description: str
    task: TaskSchema
    created_at: str
    tags: list[str]
    properties: list[PropertySchema]


class ResultInfoSchema(BaseModel):
    count: int
    page: int
    per_page: int
    total_count: int


class ModelsSearchResponseSchema(BaseModel):
    success: bool
    result: list[ModelDescriptionSchema]
    errors: list[ErroItemSchema]
    result_info: ResultInfoSchema
