from pydantic import BaseModel


class UsageSchema(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatResponseSchema(BaseModel):
    answer: str
    provider: str
    model: str
    usage: UsageSchema | None
