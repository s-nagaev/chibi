import asyncio
from functools import wraps
from typing import Any, Callable, Optional

from chibi.config import application_settings
from chibi.storage.abstract import Database

if application_settings.redis:
    from chibi.storage.redis import RedisStorage
else:
    from chibi.storage.local import LocalStorage


class DatabaseCache:
    def __init__(self) -> None:
        self._cache: Optional[Database] = None
        self._lock = asyncio.Lock()

    async def get_database(self) -> Database:
        async with self._lock:
            if self._cache is not None:
                return self._cache

            if application_settings.redis:
                self._cache = await RedisStorage.create(
                    url=application_settings.redis, password=application_settings.redis_password
                )
            else:
                self._cache = LocalStorage(application_settings.local_data_path)

            return self._cache

    def clear_cache(self) -> None:
        self._cache = None


_db_provider = DatabaseCache()


def inject_database(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        db = await _db_provider.get_database()
        return await func(db, *args, **kwargs)

    return wrapper
