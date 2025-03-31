from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel


class UsageSchema(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatResponseSchema(BaseModel):
    answer: str
    provider: str
    model: str
    usage: UsageSchema | CompletionUsage | None


class ModelChangeSchema(BaseModel):
    provider: str
    name: str
    image_generation: bool
