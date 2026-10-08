"""
Tool for reading the content of text-based files sent via Telegram.
"""

from typing import Any, Unpack

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.storage.files import get_file_storage


class ReadTelegramFileTool(ChibiTool):
    """Read the content of a text-based file sent via Telegram."""

    register = True
    run_in_background_by_default = False
    name = "read_telegram_file"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="read_telegram_file",
            description=(
                "Read the content of a text-based file (e.g. .md, .txt, .py, .json) that was "
                "previously sent as a document to this Telegram chat. "
                "The file_id comes from upload captions or the get_file_info tool. "
                "Non-text files (images, binaries) and oversized files are rejected. "
                "Very long files may be truncated with an explicit marker at the end of the content. "
                "Only works when file storage is set to Telegram; otherwise an error is returned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique identifier of a previously uploaded text-based file.",
                    },
                },
                "required": ["file_id"],
            },
        ),
    )

    @classmethod
    async def function(cls, file_id: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """Read the content of a stored Telegram text file.

        Args:
            file_id: The unique identifier of a previously uploaded text-based file.

        Returns:
            Dict containing the decoded file content. The content may end with an
            explicit truncation marker added by the storage layer.

        Raises:
            NoUserInterfaceProvidedException: If no user interface is available.
            ValueError: If the current file storage or interface type is not supported.
            FileNotFoundError: If no document with the given ID is stored.
            StorageError: If the file is not a text file or exceeds the size limit.
        """
        interface = cls.get_interface(kwargs=kwargs)
        storage = get_file_storage(interface=interface)
        content = await storage.get_text(file_id=file_id)
        return {"content": content}
