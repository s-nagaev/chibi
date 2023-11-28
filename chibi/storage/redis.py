from typing import Optional
from urllib.parse import urlparse

from loguru import logger
from redis.asyncio import Redis, from_url

from chibi.config import gpt_settings
from chibi.models import Message, User
from chibi.storage.abstract import Database


class RedisStorage(Database):
    def __init__(self, url: str, password: str | None = None, db: int = 0) -> None:
        self.redis: Redis
        self.url = url
        self.password = password
        self.db = db
        logger.info("Redis storage initialized.")

    @classmethod
    async def create(cls, url: str, password: str | None = None, db: int = 0) -> "RedisStorage":
        instance = cls(url, password, db)
        await instance.connect()
        return instance

    async def connect(self) -> None:
        redis_dsn = self._combine_redis_dsn(base_dsn=self.url, password=self.password)
        self.redis = await from_url(redis_dsn)

    def _combine_redis_dsn(self, base_dsn: str, password: str | None) -> str:
        if not password:
            return base_dsn

        parsed_dsn = urlparse(base_dsn)
        password_in_dsn = parsed_dsn.password or None

        if password_in_dsn:
            logger.warning(
                "Redis password specified twice: in the REDIS_PASSWORD and REDIS environment variables. "
                "Trying to use the password from the Redis DSN..."
            )
            return base_dsn

        if host := parsed_dsn.hostname:
            return base_dsn.replace(host, f":{password}@{host}")

        raise ValueError("Incorrect Redis DSN string provided.")

    async def save_user(self, user: User) -> None:
        user_key = f"user:{user.id}"
        user_data = user.json()
        await self.redis.set(user_key, user_data)

        for message in user.messages:
            message_key = f"user:{user.id}:message:{message.id}"
            message_data = message.json()
            await self.redis.set(message_key, message_data)
            await self.redis.expire(message_key, gpt_settings.messages_ttl)

    async def create_user(self, user_id: int) -> User:
        user = User(id=user_id)
        initial_message = Message(role="system", content=gpt_settings.assistant_prompt)
        user.messages.append(initial_message)
        user_key = f"user:{user_id}"

        await self.redis.set(user_key, user.json())
        await self.add_message(user=user, message=initial_message)
        return user

    async def get_user(self, user_id: int) -> Optional[User]:
        user_key = f"user:{user_id}"
        user_data = await self.redis.get(user_key)
        if not user_data:
            return None

        user = User.parse_raw(user_data)
        message_keys_pattern = f"user:{user.id}:message:*"
        message_keys = await self.redis.keys(message_keys_pattern)
        user_messages = [Message.parse_raw(await self.redis.get(message_key)) for message_key in message_keys]

        user.messages = sorted(user_messages, key=lambda msg: msg.id)

        return user

    async def get_or_create_user(self, user_id: int) -> User:
        if user := await self.get_user(user_id=user_id):
            return user
        return await self.create_user(user_id=user_id)

    async def add_message(self, user: User, message: Message, ttl: Optional[int] = None) -> None:
        message_to_save = Message(role=message.role, content=message.content)
        message_key = f"user:{user.id}:message:{message.id}"

        await self.redis.set(name=message_key, value=message_to_save.json(exclude={"expire_at"}))
        if ttl:
            await self.redis.expire(name=message_key, time=ttl)

    async def get_messages(self, user: User) -> list[dict[str, str]]:
        user_refreshed = await self.get_or_create_user(user_id=user.id)
        return [msg.dict(exclude={"expire_at", "id"}) for msg in user_refreshed.messages]

    async def drop_messages(self, user: User) -> None:
        message_keys_pattern = f"user:{user.id}:message:*"
        message_keys = await self.redis.keys(message_keys_pattern)

        for message_key in message_keys:
            await self.redis.delete(message_key)

        initial_message = Message(role="system", content=gpt_settings.assistant_prompt)
        user.messages.append(initial_message)
        await self.add_message(user=user, message=initial_message, ttl=gpt_settings.messages_ttl)

    async def close(self) -> None:
        await self.redis.close()
