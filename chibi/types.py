from typing import Literal, Required, TypedDict, Union

from openai.types.chat import ChatCompletionMessageParam


class UserMessageSchema(TypedDict, total=False):
    content: Required[str]
    role: Required[Literal["user"]]


class AssistantMessageSchema(TypedDict, total=False):
    content: Required[str]
    role: Required[Literal["assistant"]]


ChatCompletionMessageSchema = Union[AssistantMessageSchema, UserMessageSchema, ChatCompletionMessageParam]
