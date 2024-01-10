import time
from abc import ABC, abstractmethod
from typing import Optional

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from chibi.models import ImageMeta, Message, User

CHAT_COMPLETION_CLASSES = {
    "system": ChatCompletionSystemMessageParam,
    "user": ChatCompletionUserMessageParam,
    "assistant": ChatCompletionAssistantMessageParam,
}


class Database(ABC):
    @abstractmethod
    async def get_or_create_user(self, user_id: int) -> User:
        ...

    @abstractmethod
    async def save_user(self, user: User) -> None:
        ...

    @abstractmethod
    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None:
        ...

    @abstractmethod
    async def get_messages(self, user: User) -> list[dict[str, str]]:
        ...

    @abstractmethod
    async def drop_messages(self, user: User) -> None:
        ...

    async def get_conversation_messages(self, user: User) -> list[ChatCompletionMessageParam]:
        messages = await self.get_messages(user=user)
        conversation_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            wrapper_class = CHAT_COMPLETION_CLASSES.get(message["role"])
            if not wrapper_class:
                conversation_messages.append(message)  # type: ignore
                continue
            conversation_messages.append(wrapper_class(**message))
        return conversation_messages

    async def count_image(self, user_id: int) -> None:
        user = await self.get_or_create_user(user_id=user_id)
        expire_at = time.time() + 60 * 750  # ~ 1 month
        user.images.append(ImageMeta(expire_at=expire_at))
        await self.save_user(user)
