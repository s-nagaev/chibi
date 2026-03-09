from typing import Any

from chibi.storage.files.file_storage import FileStorage


class LocalFileStorage(FileStorage):
    async def save(self, file_metadata: dict[str, Any]) -> str:
        raise NotImplementedError

    async def get_bytes(self, file_id: str) -> bytes:
        raise NotImplementedError

    async def get_base64(self, file_id: str) -> str:
        raise NotImplementedError

    async def get_text(self, file_id: str) -> str:
        raise NotImplementedError

    async def delete(self, file_id: str) -> None:
        raise NotImplementedError

    async def get_file_info(self, file_id: str) -> dict[str, Any]:
        raise NotImplementedError
