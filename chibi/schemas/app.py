from typing import TYPE_CHECKING

from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    pass


class UsageSchema(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
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
    display_name: str = ""
    image_generation: bool

    @model_validator(mode="after")
    def set_display_name_if_none(self) -> "ModelChangeSchema":
        if not self.display_name:
            self.display_name = self.name
        return self
