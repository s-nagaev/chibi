from pydantic import BaseModel


class ContentItemSchema(BaseModel):
    type: str


class TextItemSchema(ContentItemSchema):
    text: str | None = None


class ToolCallItemSchema(ContentItemSchema):
    id: str | None = None
    name: str | None = None
    input: dict[str, str] | None = None


class AnthropicUsageSchema(BaseModel):
    input_tokens: int
    output_tokens: int
