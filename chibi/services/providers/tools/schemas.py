from typing import Any

from pydantic import BaseModel


class ToolResponseSchema(BaseModel):
    tool_name: str
    status: str
    result: dict[str, Any] | list[dict[str, Any]] | str
    additional_details: str | None = None


class ToolCallSchema(BaseModel):
    tool_name: str
    args: dict[str, Any]
