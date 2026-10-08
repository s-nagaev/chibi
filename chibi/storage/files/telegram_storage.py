import base64
from io import BytesIO
from typing import Any

from telegram import File

from chibi.exceptions import StorageError
from chibi.services.interface import TelegramInterface
from chibi.services.user import get_telegram_document, get_telegram_documents, save_telegram_document_metadata
from chibi.storage.files.file_storage import FileStorage
from chibi.utils.text_files import is_text_file

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_DECODED_TEXT_CHARS = 300_000
TRUNCATION_MARKER_TEMPLATE = "...[truncated {truncated} of {total} characters]"


class TelegramFileStorage(FileStorage):
    def __init__(self, interface: TelegramInterface) -> None:
        """Initialize the storage with the Telegram interface of the current user.

        Args:
            interface: Telegram interface carrying the update context.
        """
        self.interface = interface

    async def save(self, file_metadata: dict[str, Any]) -> str:
        """Persist Telegram document metadata for the current user.

        Args:
            file_metadata: Metadata of the uploaded document.

        Returns:
            The unique ID the metadata is stored under.
        """
        return await save_telegram_document_metadata(
            storage_id=self.interface.user_id,
            file_metadata=file_metadata,
        )

    async def get_bytes(self, file_id: str) -> bytes:
        """Download the raw content of a stored Telegram document.

        Args:
            file_id: Unique ID of the stored document.

        Returns:
            The raw file content.

        Raises:
            FileNotFoundError: If no document with the given ID is stored.
        """
        file_meta = await get_telegram_document(storage_id=self.interface.user_id, file_unique_id=file_id)
        if not file_meta:
            raise FileNotFoundError(f"No file with ID '{file_id}' found")

        telegram_file_id = file_meta.file_id

        file: File = await self.interface.context.bot.get_file(file_id=telegram_file_id)
        data = BytesIO()
        await file.download_to_memory(out=data)
        data.seek(0)
        return data.getvalue()

    async def get_base64(self, file_id: str) -> str:
        """Download a stored Telegram document and return it base64-encoded.

        Args:
            file_id: Unique ID of the stored document.

        Returns:
            The base64-encoded file content.
        """
        file_bytes = await self.get_bytes(file_id)
        return base64.b64encode(file_bytes).decode("ascii")

    async def get_available_files(self, limit: int = 0) -> dict[str, str | int]:
        """List stored Telegram documents of the current user.

        Args:
            limit: Maximum number of the most recent files to return; 0 means all.

        Returns:
            Mapping of file unique IDs to human-readable descriptions.
        """
        files = await get_telegram_documents(storage_id=self.interface.user_id, limit=limit)
        return {
            file.file_unique_id: f"{file.file_name} ({file.short_description or 'description n/a'})"
            for file in files.values()
        }

    async def get_text(self, file_id: str) -> str:
        """Fetch and decode the content of a stored text file.

        Args:
            file_id: Unique ID of the stored document.

        Returns:
            The decoded UTF-8 content, truncated with an explicit marker if it
            exceeds MAX_DECODED_TEXT_CHARS characters.

        Raises:
            FileNotFoundError: If no document with the given ID is stored.
            StorageError: If the file is not a text file or exceeds
                MAX_TEXT_FILE_BYTES bytes.
        """
        file_meta = await get_telegram_document(storage_id=self.interface.user_id, file_unique_id=file_id)
        if not file_meta:
            raise FileNotFoundError(f"No file with ID '{file_id}' found")

        if not is_text_file(mime_type=file_meta.mime_type, file_name=file_meta.file_name):
            raise StorageError(f"File '{file_meta.file_name}' is not a text file.")

        if file_meta.file_size > MAX_TEXT_FILE_BYTES:
            raise StorageError(
                f"File '{file_meta.file_name}' is too large to read as text "
                f"({file_meta.file_size} bytes, limit is {MAX_TEXT_FILE_BYTES} bytes)."
            )

        raw = await self.get_bytes(file_id)
        text = raw.decode("utf-8", errors="replace")
        if len(text) > MAX_DECODED_TEXT_CHARS:
            truncated_count = len(text) - MAX_DECODED_TEXT_CHARS
            marker = TRUNCATION_MARKER_TEMPLATE.format(truncated=truncated_count, total=len(text))
            text = text[:MAX_DECODED_TEXT_CHARS] + marker
        return text

    async def delete(self, file_id: str) -> None:
        """Delete a stored Telegram document.

        Args:
            file_id: Unique ID of the stored document.
        """
        raise NotImplementedError

    async def get_file_info(self, file_id: str) -> dict[str, Any]:
        """Return stored metadata of a Telegram document.

        Args:
            file_id: Unique ID of the stored document.

        Returns:
            The stored document metadata.

        Raises:
            FileNotFoundError: If no document with the given ID is stored.
        """
        file_meta = await get_telegram_document(storage_id=self.interface.user_id, file_unique_id=file_id)
        if not file_meta:
            raise FileNotFoundError(f"No file with ID '{file_id}' found")
        return file_meta.model_dump()
