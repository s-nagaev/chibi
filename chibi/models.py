from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str
    expire_at: Optional[float]


class User(BaseModel):
    id: int
    api_token: Optional[str]
    messages: list[Message] = Field(default_factory=list)
