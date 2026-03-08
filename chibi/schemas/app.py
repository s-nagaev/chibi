from typing import TYPE_CHECKING

from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pydantic import BaseModel, Field, model_validator

from chibi.config import telegram_settings

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


class MetricTagsSchema(UsageSchema):
    user_id: int
    user_name: str | None = None
    provider: str
    model: str
    bot: str = telegram_settings.bot_name


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


class ModeratorsAnswer(BaseModel):
    status: str | None = None
    verdict: str
    reason: str | None = None


class VisionResultSchema(BaseModel):
    short_description: str = Field(description="Image short description, up to 100 characters")
    full_description: str = Field(description="Image full description")
    text: str | None = Field(
        default=None,
        description=(
            "The text extracted from the image, must be filled if image is document and can be "
            "omitted if image does not contain any text"
        ),
    )
