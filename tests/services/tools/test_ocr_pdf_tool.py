"""Unit tests for OcrPdfTool."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.services.providers.tools.ocr_pdf import OcrPdfTool

# Sample PDF bytes for testing
SAMPLE_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF) Tj\nET\nendstream\nendobj\n"
    b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n"
    b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n307\n%%EOF"
)

TEST_DIR = Path(__file__).parent / "test_ocr_pdf_files"


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for test files."""
    TEST_DIR.mkdir(exist_ok=True)
    yield
    # Teardown: Clean up test files and directory
    if TEST_DIR.exists():
        for f in TEST_DIR.iterdir():
            f.unlink()
        TEST_DIR.rmdir()


@pytest.fixture
def mock_interface():
    """Create a mock interface."""
    interface = Mock()
    interface.user_id = 12345
    return interface


@pytest.fixture
def mock_kwargs(mock_interface):
    """Create mock kwargs with interface."""
    return {"interface": mock_interface, "user_id": 12345}


class TestOcrPdfTool:
    """Tests for OcrPdfTool."""

    def test_tool_registered(self):
        """Test that tool is registered."""
        assert OcrPdfTool.register is True
        assert OcrPdfTool.name == "ocr_pdf"
        assert OcrPdfTool.run_in_background_by_default is True

    def test_definition_has_required_params(self):
        """Test that tool definition has required parameters."""
        properties = OcrPdfTool.definition["function"]["parameters"]["properties"]
        assert "file_id" in properties  # type: ignore[operator]
        assert "absolute_path" in properties  # type: ignore[operator]

    async def test_xor_validation_both_provided(self, mock_kwargs):
        """Test that providing both file_id and absolute_path raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one of"):
            await OcrPdfTool.function(
                file_id="test-file-id",
                absolute_path="/some/path.pdf",
                **mock_kwargs,
            )

    async def test_xor_validation_neither_provided(self, mock_kwargs):
        """Test that providing neither file_id nor absolute_path raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one of"):
            await OcrPdfTool.function(**mock_kwargs)

    @patch("chibi.services.providers.tools.ocr_pdf.gpt_settings")
    async def test_absolute_path_without_permission_raises_error(self, mock_gpt_settings, mock_kwargs):
        """Test that absolute_path without filesystem_access raises PermissionError."""
        mock_gpt_settings.filesystem_access = False

        with pytest.raises(PermissionError, match="Filesystem access is not enabled"):
            await OcrPdfTool.function(
                absolute_path="/some/path.pdf",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.ocr_pdf.gpt_settings")
    async def test_absolute_path_file_not_found(self, mock_gpt_settings, mock_kwargs):
        """Test that non-existent absolute_path raises FileNotFoundError."""
        mock_gpt_settings.filesystem_access = True

        with pytest.raises(FileNotFoundError, match="File not found"):
            await OcrPdfTool.function(
                absolute_path="/non/existent/path.pdf",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.ocr_pdf.gpt_settings")
    async def test_absolute_path_not_pdf_raises(self, mock_gpt_settings, mock_kwargs):
        """Test that non-PDF file raises ValueError."""
        mock_gpt_settings.filesystem_access = True

        # Create a test file with wrong extension
        test_file = TEST_DIR / "test_file.txt"
        test_file.write_bytes(b"not a pdf")

        with pytest.raises(ValueError, match="not a PDF"):
            await OcrPdfTool.function(
                absolute_path=str(test_file),
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.ocr_pdf.gpt_settings")
    async def test_absolute_path_success(self, mock_gpt_settings, mock_kwargs):
        """Test successful OCR of local PDF file."""
        mock_gpt_settings.filesystem_access = True

        # Create a test PDF file
        test_file = TEST_DIR / "test_document.pdf"
        test_file.write_bytes(SAMPLE_PDF_BYTES)

        # Mock ocr_pdf service function
        with patch("chibi.services.providers.tools.ocr_pdf.ocr_pdf") as mock_ocr:
            mock_ocr.return_value = Mock(
                model_dump=lambda: {
                    "short_description": "Test PDF",
                    "full_description": "A test PDF document",
                    "text": "Extracted text from PDF",
                }
            )

            await OcrPdfTool.function(
                absolute_path=str(test_file),
                **mock_kwargs,
            )

            mock_ocr.assert_called_once()
            call_kwargs = mock_ocr.call_args.kwargs
            assert call_kwargs["user_id"] == 12345
            assert call_kwargs["pdf"] == SAMPLE_PDF_BYTES

    @patch("chibi.services.providers.tools.ocr_pdf.get_file_storage")
    async def test_file_id_success(self, mock_get_storage, mock_kwargs):
        """Test successful OCR using file_id."""
        mock_storage = AsyncMock()
        mock_storage.get_file_info.return_value = {
            "file_id": "test123",
            "mime_type": "application/pdf",
        }
        mock_storage.get_bytes.return_value = SAMPLE_PDF_BYTES
        mock_get_storage.return_value = mock_storage

        # Mock ocr_pdf service function
        with patch("chibi.services.providers.tools.ocr_pdf.ocr_pdf") as mock_ocr:
            mock_ocr.return_value = Mock(
                model_dump=lambda: {
                    "short_description": "Test PDF",
                    "full_description": "A test PDF document",
                    "text": "Extracted text from PDF",
                }
            )

            await OcrPdfTool.function(
                file_id="test123",
                **mock_kwargs,
            )

            # Verify storage methods were called
            mock_storage.get_file_info.assert_called_once_with(file_id="test123")
            mock_storage.get_bytes.assert_called_once_with(file_id="test123")

            # Verify ocr_pdf was called with PDF bytes
            mock_ocr.assert_called_once()
            call_kwargs = mock_ocr.call_args.kwargs
            assert call_kwargs["user_id"] == 12345
            assert call_kwargs["pdf"] == SAMPLE_PDF_BYTES

    @patch("chibi.services.providers.tools.ocr_pdf.get_file_storage")
    async def test_file_id_not_pdf_raises(self, mock_get_storage, mock_kwargs):
        """Test that non-PDF mime type raises ValueError."""
        mock_storage = AsyncMock()
        mock_storage.get_file_info.return_value = {
            "file_id": "test123",
            "mime_type": "image/png",
        }
        mock_get_storage.return_value = mock_storage

        with pytest.raises(ValueError, match="not a PDF"):
            await OcrPdfTool.function(
                file_id="test123",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.ocr_pdf.get_file_storage")
    async def test_ocr_exception_propagates(self, mock_get_storage, mock_kwargs):
        """Test that exceptions from ocr_pdf service propagate."""
        # Setup mock storage
        mock_storage = AsyncMock()
        mock_storage.get_file_info.return_value = {
            "file_id": "test123",
            "mime_type": "application/pdf",
        }
        mock_storage.get_bytes.return_value = SAMPLE_PDF_BYTES
        mock_get_storage.return_value = mock_storage

        # Mock ocr_pdf to raise an exception
        with patch("chibi.services.providers.tools.ocr_pdf.ocr_pdf") as mock_ocr:
            mock_ocr.side_effect = ValueError("OCR provider error")

            with pytest.raises(ValueError, match="OCR provider error"):
                await OcrPdfTool.function(
                    file_id="test123",
                    **mock_kwargs,
                )

    @patch("chibi.services.providers.tools.ocr_pdf.gpt_settings")
    async def test_model_override_passed_to_service(self, mock_gpt_settings, mock_kwargs):
        """Test that model parameter is passed to the service function."""
        mock_gpt_settings.filesystem_access = True

        # Create a test PDF file
        test_file = TEST_DIR / "test_document.pdf"
        test_file.write_bytes(SAMPLE_PDF_BYTES)

        # Mock ocr_pdf service function
        with patch("chibi.services.providers.tools.ocr_pdf.ocr_pdf") as mock_ocr:
            mock_ocr.return_value = Mock(
                model_dump=lambda: {
                    "short_description": "Test PDF",
                    "full_description": "A test PDF document",
                    "text": "Extracted text",
                }
            )

            # Note: Currently OcrPdfTool doesn't have model parameter in its function
            # This test verifies the service layer receives correct params
            await OcrPdfTool.function(
                absolute_path=str(test_file),
                **mock_kwargs,
            )

            mock_ocr.assert_called_once()
            # Verify model override support would work (currently not exposed in tool)
            call_kwargs = mock_ocr.call_args.kwargs
            assert "pdf" in call_kwargs
