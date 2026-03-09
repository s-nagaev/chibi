import base64
from io import BytesIO
from typing import Any

from telegram import File

from chibi.services.interface import TelegramInterface
from chibi.services.user import get_telegram_document, get_telegram_documents, save_telegram_document_metadata
from chibi.storage.files.file_storage import FileStorage


class TelegramFileStorage(FileStorage):
    def __init__(self, interface: TelegramInterface) -> None:
        self.interface = interface

    async def save(self, file_metadata: dict[str, Any]) -> str:
        return await save_telegram_document_metadata(
            user_id=self.interface.user_id,
            file_metadata=file_metadata,
        )

    async def get_bytes(self, file_id: str) -> bytes:
        file_meta = await get_telegram_document(user_id=self.interface.user_id, file_unique_id=file_id)
        if not file_meta:
            raise FileNotFoundError(f"No file with ID '{file_id}' found")

        telegram_file_id = file_meta.file_id

        file: File = await self.interface.context.bot.get_file(file_id=telegram_file_id)
        data = BytesIO()
        await file.download_to_memory(out=data)
        data.seek(0)
        return data.getvalue()

    async def get_base64(self, file_id: str) -> str:
        file_bytes = await self.get_bytes(file_id)
        return base64.b64encode(file_bytes).decode("ascii")

    async def get_available_files(self, limit: int = 0) -> dict[str, str | int]:
        files = await get_telegram_documents(user_id=self.interface.user_id, limit=limit)
        return {
            file.file_unique_id: f"{file.file_name} ({file.short_description or 'description n/a'})"
            for file in files.values()
        }

    async def get_text(self, file_id: str) -> str:
        raise NotImplementedError

    async def delete(self, file_id: str) -> None:
        raise NotImplementedError

    async def get_file_info(self, file_id: str) -> dict[str, Any]:
        file_meta = await get_telegram_document(user_id=self.interface.user_id, file_unique_id=file_id)
        if not file_meta:
            raise FileNotFoundError(f"No file with ID '{file_id}' found")
        return file_meta.model_dump()
