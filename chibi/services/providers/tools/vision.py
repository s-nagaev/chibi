"""
Vision tools for file info and image analysis.
"""

import mimetypes
from pathlib import Path
from typing import Any, Unpack

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import describe_image
from chibi.storage.files import get_file_storage


class GetFileInfoTool(ChibiTool):
    """Get metadata information about a file."""

    register = True
    run_in_background_by_default = False
    name = "get_file_info"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_file_info",
            description="Get metadata information about a file that was previously uploaded.",
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique identifier of the file.",
                    },
                },
                "required": ["file_id"],
            },
        ),
    )

    @classmethod
    async def function(cls, file_id: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        """Get file metadata by file ID.

        Args:
            file_id: The unique identifier of the file.

        Returns:
            Dict containing file metadata.
        """
        interface = cls.get_interface(kwargs=kwargs)
        storage = get_file_storage(interface=interface)
        return await storage.get_file_info(file_id=file_id)


class AnalyzeImageTool(ChibiTool):
    """Analyze images using vision model with custom prompt."""

    register = True
    run_in_background_by_default = True
    name = "analyze_image"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="analyze_image",
            description=(
                "Analyze an image using the vision model. "
                "Provide either a file_id (for previously uploaded files) or absolute_path (for local files). "
                "You must provide exactly one of these, not both."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique identifier of a previously uploaded file.",
                    },
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute path to a local image file.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to guide the vision analysis.",
                    },
                },
                "required": ["prompt"],
            },
        ),
    )

    @classmethod
    async def function(
        cls,
        file_id: str | None = None,
        absolute_path: str | None = None,
        prompt: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        """Analyze an image using the vision model.

        Args:
            file_id: The unique identifier of a previously uploaded file.
            absolute_path: Absolute path to a local image file.
            prompt: Prompt to guide the vision analysis.

        Returns:
            Vision analysis result.

        Raises:
            ValueError: If both or neither of file_id and absolute_path are provided.
            PermissionError: If filesystem_access is required but not enabled.
            FileNotFoundError: If the file does not exist.
        """
        # XOR validation: exactly one of file_id or absolute_path must be provided
        if (file_id is None) == (absolute_path is None):
            raise ValueError("Exactly one of 'file_id' or 'absolute_path' must be provided, not both or neither.")

        interface = cls.get_interface(kwargs=kwargs)
        user_id = interface.user_id

        if absolute_path:
            if not gpt_settings.filesystem_access:
                raise PermissionError(
                    "Filesystem access is not enabled. Cannot analyze local files without filesystem_access permission."
                )

            # Read local file
            path = Path(absolute_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, "rb") as f:
                image_bytes = f.read()

            # Determine mime type
            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type is None:
                mime_type = "application/octet-stream"
        else:
            # Get file from storage
            assert file_id is not None  # Type narrowing for mypy
            storage = get_file_storage(interface=interface)
            file_info = await storage.get_file_info(file_id=file_id)
            image_bytes = await storage.get_bytes(file_id=file_id)
            mime_type = file_info.get("mime_type", "application/octet-stream")

        # Call describe_image with the custom prompt
        result = await describe_image(
            user_id=user_id,
            image=image_bytes,
            mime_type=mime_type,
            prompt=prompt,
        )

        return result.model_dump()
