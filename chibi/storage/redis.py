import json
from urllib.parse import urlparse

from loguru import logger
from redis.asyncio import Redis, from_url
from redis.exceptions import ConnectionError, TimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from chibi.models import Message, User
from chibi.storage.abstract import Database

retry_connection = retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)


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

    @retry_connection
    async def save_user(self, user: User) -> None:
        user_key = f"user:{user.id}"
        user_data = user.model_dump_json()
        await self.redis.set(user_key, user_data)

    @retry_connection
    async def create_user(self, user_id: int) -> User:
        user = User(id=user_id)
        user_key = f"user:{user_id}"
        await self.redis.set(user_key, user.model_dump_json())
        return user

    @retry_connection
    async def get_user(self, user_id: int) -> User | None:
        user_key = f"user:{user_id}"
        user_data = await self.redis.get(user_key)
        if not user_data:
            return None

        user = User.model_validate_json(user_data)
        return user

    @retry_connection
    async def add_message(self, user: User, message: Message, ttl: int | None = None, thread_id: int = 0) -> None:
        if thread_id:
            message_key = f"user:{user.id}:thread:{thread_id}:message:{message.id}"
        else:
            message_key = f"user:{user.id}:message:{message.id}"

        await self.redis.set(name=message_key, value=message.model_dump_json(exclude={"expire_at"}))
        if ttl:
            await self.redis.expire(name=message_key, time=ttl)

    @retry_connection
    async def get_messages(self, user: User, thread_id: int = 0) -> list[dict[str, str]]:
        if thread_id:
            message_keys_pattern = f"user:{user.id}:thread:{thread_id}:message:*"
        else:
            message_keys_pattern = f"user:{user.id}:message:*"

        message_keys = set(await self.redis.keys(message_keys_pattern))
        if not message_keys:
            return []

        messages = [json.loads(await self.redis.get(message_key)) for message_key in message_keys]
        messages.sort(key=lambda msg: msg.get("id", 0))
        return messages

    @retry_connection
    async def drop_messages(self, user: User, thread_id: int = 0) -> None:
        if thread_id:
            message_keys_pattern = f"user:{user.id}:thread:{thread_id}:message:*"
        else:
            message_keys_pattern = f"user:{user.id}:message:*"

        message_keys = await self.redis.keys(message_keys_pattern)

        for message_key in message_keys:
            await self.redis.delete(message_key)

    @retry_connection
    async def close(self) -> None:
        await self.redis.aclose()
