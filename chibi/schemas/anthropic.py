from pydantic import BaseModel


class ContentItemSchema(BaseModel):
    type: str
    text: str | None = None


class UsageSchema(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatCompletionSchema(BaseModel):
    id: str
    type: str
    role: str
    content: list[ContentItemSchema]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: UsageSchema
