import time
from abc import ABC, abstractmethod
from typing import Optional

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from chibi.models import ImageMeta, Message, User

CHAT_COMPLETION_CLASSES = {
    "assistant": ChatCompletionAssistantMessageParam,
    "function": ChatCompletionFunctionMessageParam,
    "tool": ChatCompletionToolMessageParam,
    "user": ChatCompletionUserMessageParam,
}


class Database(ABC):
    async def get_or_create_user(self, user_id: int) -> User:
        if user := await self.get_user(user_id=user_id):
            return user
        return await self.create_user(user_id=user_id)

    @abstractmethod
    async def save_user(self, user: User) -> None: ...

    @abstractmethod
    async def create_user(self, user_id: int) -> User: ...

    @abstractmethod
    async def get_user(self, user_id: int) -> User | None: ...

    @abstractmethod
    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None: ...

    @abstractmethod
    async def get_messages(self, user: User) -> list[dict[str, str]]: ...

    @abstractmethod
    async def drop_messages(self, user: User) -> None: ...

    async def get_conversation_messages(self, user: User) -> list[Message]:
        messages = await self.get_messages(user=user)
        return [Message(**msg) for msg in messages]

    async def count_image(self, user_id: int) -> None:
        user = await self.get_or_create_user(user_id=user_id)
        expire_at = time.time() + 60 * 750  # ~ 1 month
        user.images.append(ImageMeta(expire_at=expire_at))
        await self.save_user(user)
