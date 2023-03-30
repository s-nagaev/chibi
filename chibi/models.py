import time
from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    id: int = Field(default_factory=time.time_ns)
    role: str
    content: str
    expire_at: Optional[float] = None


class User(BaseModel):
    id: int
    api_token: Optional[str]
    messages: list[Message] = Field(default_factory=list)
