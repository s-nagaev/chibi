import asyncio
from typing import Hashable
from weakref import WeakValueDictionary

from chibi.utils.app import SingletonMeta


class LockManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        """Initialize the lock manager."""
        self._locks: WeakValueDictionary[Hashable, asyncio.Lock] = WeakValueDictionary()
        self._dict_lock: asyncio.Lock = asyncio.Lock()

    async def get_lock(self, key: Hashable) -> asyncio.Lock:
        async with self._dict_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock
