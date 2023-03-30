import asyncio
from functools import wraps
from typing import Optional

from chibi.config import application_settings
from chibi.storage.abc import Database
from chibi.storage.local import LocalStorage
from chibi.storage.redis import RedisStorage


class DatabaseCache:
    def __init__(self):
        self._cache: Optional[Database] = None
        self._lock = asyncio.Lock()

    async def get_database(self) -> Database:
        async with self._lock:
            if self._cache is not None:
                return self._cache

            if application_settings.redis:
                self._cache = await RedisStorage.create(application_settings.redis)
            else:
                self._cache = LocalStorage(application_settings.local_data_path)

            return self._cache

    def clear_cache(self):
        self._cache = None


_db_provider = DatabaseCache()


def inject_database(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        db = await _db_provider.get_database()
        return await func(db, *args, **kwargs)

    return wrapper
