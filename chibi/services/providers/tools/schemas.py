from typing import Any

from pydantic import BaseModel


class ToolResponse(BaseModel):
    tool_name: str
    status: str
    result: dict[str, Any] | list[dict[str, Any]] | str
    additional_details: str | None = None
