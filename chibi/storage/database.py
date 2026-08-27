import asyncio
from functools import wraps
from typing import Awaitable, Callable, Concatenate, Generic, Optional, ParamSpec, TypeVar, cast

from chibi.config.app import application_settings
from chibi.memory.chroma import memory, with_chroma_archival
from chibi.storage.abstract import Database
from chibi.storage.dynamodb import DynamoDBStorage
from chibi.storage.local import LocalStorage
from chibi.storage.redis import RedisStorage

R = TypeVar("R")
P = ParamSpec("P")


class InjectedCallable(Generic[P, R]):
    """Callable produced by ``inject_database`` that preserves the original function.

    Exposes the undecorated function via ``__wrapped__`` so type checkers can
    see the original signature (including the injected ``Database`` parameter).
    """

    __wrapped__: Callable[Concatenate[Database, P], Awaitable[R]]

    def __init__(
        self,
        func: Callable[Concatenate[Database, P], Awaitable[R]],
        wrapper: Callable[P, Awaitable[R]],
    ) -> None:
        self.__wrapped__ = func
        self._wrapper = wrapper

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]:
        return self._wrapper(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<InjectedCallable {self.__wrapped__.__module__}.{self.__wrapped__.__name__}>"


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
                inner: RedisStorage | DynamoDBStorage | LocalStorage = await RedisStorage.create(
                    url=cast(str, application_settings.redis),
                    password=application_settings.redis_password,
                )
            elif backend == "dynamodb":
                # DynamoDBStorage.create expects region, access_key, secret_key, tables
                inner = await DynamoDBStorage.create(
                    region=application_settings.aws_region or "",
                    access_key=application_settings.aws_access_key_id,
                    secret_access_key=application_settings.aws_secret_access_key,
                    users_table=application_settings.ddb_users_table or "",
                    messages_table=application_settings.ddb_messages_table or "",
                )
            else:
                # default to local storage
                inner = LocalStorage(application_settings.local_data_path)

            # Attach ChromaDB archival (fire-and-forget) if memory is configured
            self._cache = with_chroma_archival(memory)(inner)

            return self._cache

    def clear_cache(self) -> None:
        """
        Clear the cached Database instance, forcing reinitialization on next use.
        """
        self._cache = None


_db_provider = DatabaseCache()


def inject_database(
    func: Callable[Concatenate[Database, P], Awaitable[R]],
) -> InjectedCallable[P, R]:
    """Decorator to inject the Database instance into async functions.

    Wraps a function with signature func(db, *args, **kwargs) -> Awaitable.
    The returned callable preserves the original function via ``__wrapped__``.

    Args:
        func: The function to decorate.

    Returns:
        Function execution wrapper.
    """

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        db = await _db_provider.get_database()
        return await func(db, *args, **kwargs)

    return InjectedCallable(func, wrapper)
