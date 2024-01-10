import time
from typing import Optional

from pydantic import BaseModel, Field

from chibi.config import gpt_settings


class Message(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    role: str
    content: str
    expire_at: Optional[float] = None


class ImageMeta(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    expire_at: float


class User(BaseModel):
    id: int
    api_token: Optional[str]
    gpt_model: Optional[str]
    messages: list[Message] = Field(default_factory=list)
    images: list[ImageMeta] = Field(default_factory=list)

    @property
    def model(self) -> str:
        return self.gpt_model or gpt_settings.model_default

    @property
    def has_reached_image_limits(self) -> bool:
        if not gpt_settings.image_generations_monthly_limit:
            return False
        if self.id in gpt_settings.image_generations_whitelist:
            return False
        return len(self.images) >= gpt_settings.image_generations_monthly_limit
