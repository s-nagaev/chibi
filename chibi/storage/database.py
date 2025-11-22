import asyncio
from functools import wraps
from typing import Awaitable, Callable, Concatenate, Optional, ParamSpec, TypeVar, cast

from chibi.config.app import application_settings
from chibi.storage.abstract import Database
from chibi.storage.dynamodb import DynamoDBStorage
from chibi.storage.local import LocalStorage
from chibi.storage.redis import RedisStorage

R = TypeVar("R")
P = ParamSpec("P")


class DatabaseCache:
    """
    Caches a Database instance according to application settings.
    Supports 'local', 'redis', and 'dynamodb' backends.
    """

    def __init__(self) -> None:
        self._cache: Optional[Database] = None
        self._lock = asyncio.Lock()

    async def get_database(self) -> Database:
        """Get or create the Database instance based on storage_backend setting.

        Returns:
            Initialized Database instance.
        """
        async with self._lock:
            if self._cache is not None:
                return self._cache

            backend = application_settings.storage_backend.lower()
            if backend == "redis":
                # RedisStorage.create expects URL and password
                self._cache = await RedisStorage.create(
                    url=cast(str, application_settings.redis),
                    password=application_settings.redis_password,
                )
            elif backend == "dynamodb":
                # DynamoDBStorage.create expects region, access_key, secret_key, tables
                self._cache = await DynamoDBStorage.create(
                    region=application_settings.aws_region or "",
                    access_key=application_settings.aws_access_key_id,
                    secret_access_key=application_settings.aws_secret_access_key,
                    users_table=application_settings.ddb_users_table or "",
                    messages_table=application_settings.ddb_messages_table or "",
                )
            else:
                # default to local storage
                self._cache = LocalStorage(application_settings.local_data_path)

            return self._cache

    def clear_cache(self) -> None:
        """
        Clear the cached Database instance, forcing reinitialization on next use.
        """
        self._cache = None


_db_provider = DatabaseCache()


def inject_database(
    func: Callable[Concatenate[Database, P], Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Decorator to inject the Database instance into async functions.

    Wraps a function with signature func(db, *args, **kwargs) -> Awaitable.

    Args:
        func: The function to decorate.

    Returns:
        Function execution wrapper.
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        db = await _db_provider.get_database()
        return await func(db, *args, **kwargs)

    return wrapper
