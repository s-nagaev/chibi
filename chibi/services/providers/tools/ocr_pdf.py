"""
OCR PDF tool for extracting text from PDF documents.
"""

from pathlib import Path
from typing import Any, Unpack

from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import ocr_pdf
from chibi.storage.files import get_file_storage


class OcrPdfTool(ChibiTool):
    """Extract text from PDF documents using OCR capabilities."""

    register = True
    run_in_background_by_default = True
    name = "ocr_pdf"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="ocr_pdf",
            description=(
                "Extract text from a PDF document using OCR (Optical Character Recognition). "
                "Provide either a file_id (for previously uploaded files) or absolute_path (for local files). "
                "You must provide exactly one of these, not both."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The unique identifier of a previously uploaded PDF file.",
                    },
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute path to a local PDF file.",
                    },
                },
                "required": [],
            },
        ),
    )

    @classmethod
    async def function(
        cls,
        file_id: str | None = None,
        absolute_path: str | None = None,
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, Any]:
        """Extract text from a PDF document using OCR.

        Args:
            file_id: The unique identifier of a previously uploaded PDF file.
            absolute_path: Absolute path to a local PDF file.

        Returns:
            Dict containing the extracted text and metadata.

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
                    "Filesystem access is not enabled. Cannot read local files without filesystem_access permission."
                )

            # Read local file
            path = Path(absolute_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if path.suffix.lower() != ".pdf":
                raise ValueError(f"File is not a PDF: {path}")

            with open(path, "rb") as f:
                pdf_bytes = f.read()
        else:
            # Get file from storage
            assert file_id is not None  # Type narrowing for mypy
            storage = get_file_storage(interface=interface)
            file_info = await storage.get_file_info(file_id=file_id)

            # Verify it's a PDF
            mime_type = file_info.get("mime_type", "")
            if "pdf" not in mime_type.lower():
                raise ValueError(f"File is not a PDF: mime_type={mime_type}")

            pdf_bytes = await storage.get_bytes(file_id=file_id)

        # Call ocr_pdf service function
        result = await ocr_pdf(user_id=user_id, pdf=pdf_bytes)

        return result.model_dump()
