"""
Tests for vision tools: GetFileInfoTool and AnalyzeImageTool.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.services.providers.tools import AnalyzeImageTool, GetFileInfoTool

TEST_DIR = Path(__file__).parent / "test_vision_files"


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


class TestGetFileInfoTool:
    """Tests for GetFileInfoTool."""

    def test_tool_registered(self):
        """Test that tool is registered."""
        assert GetFileInfoTool.register is True
        assert GetFileInfoTool.name == "get_file_info"

    def test_definition_has_file_id_param(self):
        """Test that tool definition has required file_id parameter."""
        properties = GetFileInfoTool.definition["function"]["parameters"]["properties"]
        assert "file_id" in properties  # type: ignore[operator]

    @patch("chibi.services.providers.tools.vision.get_file_storage")
    async def test_function_calls_storage_get_file_info(self, mock_get_storage, mock_kwargs):
        """Test that function calls storage.get_file_info."""
        mock_storage = AsyncMock()
        mock_storage.get_file_info.return_value = {
            "file_id": "test123",
            "file_name": "test.txt",
            "file_size": 1024,
            "mime_type": "text/plain",
        }
        mock_get_storage.return_value = mock_storage

        result = await GetFileInfoTool.function(file_id="test123", **mock_kwargs)

        mock_storage.get_file_info.assert_called_once_with(file_id="test123")
        assert result["file_id"] == "test123"
        assert result["file_name"] == "test.txt"


class TestAnalyzeImageTool:
    """Tests for AnalyzeImageTool."""

    def test_tool_registered(self):
        """Test that tool is registered."""
        assert AnalyzeImageTool.register is True
        assert AnalyzeImageTool.name == "analyze_image"
        assert AnalyzeImageTool.run_in_background_by_default is True

    def test_definition_has_required_params(self):
        """Test that tool definition has required parameters."""
        properties = AnalyzeImageTool.definition["function"]["parameters"]["properties"]
        assert "file_id" in properties  # type: ignore[operator]
        assert "absolute_path" in properties  # type: ignore[operator]
        assert "prompt" in properties  # type: ignore[operator]

    async def test_xor_validation_both_provided(self, mock_kwargs):
        """Test that providing both file_id and absolute_path raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one of"):
            await AnalyzeImageTool.function(
                file_id="test123",
                absolute_path="/some/path",
                prompt="Describe this",
                **mock_kwargs,
            )

    async def test_xor_validation_neither_provided(self, mock_kwargs):
        """Test that providing neither file_id nor absolute_path raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one of"):
            await AnalyzeImageTool.function(
                prompt="Describe this",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.vision.gpt_settings")
    async def test_absolute_path_without_permission_raises_error(self, mock_gpt_settings, mock_kwargs):
        """Test that absolute_path without filesystem_access raises PermissionError."""
        mock_gpt_settings.filesystem_access = False

        with pytest.raises(PermissionError, match="Filesystem access is not enabled"):
            await AnalyzeImageTool.function(
                absolute_path="/some/path",
                prompt="Describe this",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.vision.gpt_settings")
    async def test_absolute_path_file_not_found(self, mock_gpt_settings, mock_kwargs):
        """Test that non-existent absolute_path raises FileNotFoundError."""
        mock_gpt_settings.filesystem_access = True

        with pytest.raises(FileNotFoundError, match="File not found"):
            await AnalyzeImageTool.function(
                absolute_path="/non/existent/path.png",
                prompt="Describe this",
                **mock_kwargs,
            )

    @patch("chibi.services.providers.tools.vision.gpt_settings")
    async def test_absolute_path_success(self, mock_gpt_settings, mock_kwargs):
        """Test successful analysis of local file."""
        mock_gpt_settings.filesystem_access = True

        # Create a test image file
        test_file = TEST_DIR / "test_image.png"
        test_file.write_bytes(b"fake image data")

        # Mock describe_image
        with patch("chibi.services.providers.tools.vision.describe_image") as mock_describe:
            mock_describe.return_value = Mock(
                model_dump=lambda: {
                    "short_description": "A test image",
                    "full_description": "This is a test image description",
                    "text": None,
                }
            )

            await AnalyzeImageTool.function(
                absolute_path=str(test_file),
                prompt="Describe this image",
                **mock_kwargs,
            )

            mock_describe.assert_called_once()
            call_kwargs = mock_describe.call_args.kwargs
            assert call_kwargs["prompt"] == "Describe this image"
            assert call_kwargs["mime_type"] == "image/png"

    @patch("chibi.services.providers.tools.vision.get_file_storage")
    async def test_file_id_success(self, mock_get_storage, mock_kwargs):
        """Test successful analysis using file_id."""
        mock_storage = AsyncMock()
        mock_storage.get_file_info.return_value = {
            "file_id": "test123",
            "mime_type": "image/jpeg",
        }
        mock_storage.get_bytes.return_value = b"fake image bytes"
        mock_get_storage.return_value = mock_storage

        # Mock describe_image
        with patch("chibi.services.providers.tools.vision.describe_image") as mock_describe:
            mock_describe.return_value = Mock(
                model_dump=lambda: {
                    "short_description": "A test image",
                    "full_description": "This is a test image description",
                    "text": "extracted text",
                }
            )

            await AnalyzeImageTool.function(
                file_id="test123",
                prompt="Extract text from this",
                **mock_kwargs,
            )

            mock_describe.assert_called_once()
            call_kwargs = mock_describe.call_args.kwargs
            assert call_kwargs["prompt"] == "Extract text from this"
            assert call_kwargs["mime_type"] == "image/jpeg"
            assert call_kwargs["image"] == b"fake image bytes"

    @patch("chibi.services.providers.tools.vision.gpt_settings")
    async def test_mime_type_fallback(self, mock_gpt_settings, mock_kwargs):
        """Test that unknown file types fall back to application/octet-stream."""
        mock_gpt_settings.filesystem_access = True

        # Create a test file with no extension (will have no mimetype)
        test_file = TEST_DIR / "test_image_noext"
        test_file.write_bytes(b"fake image data")

        with patch("chibi.services.providers.tools.vision.describe_image") as mock_describe:
            mock_describe.return_value = Mock(model_dump=lambda: {})

            await AnalyzeImageTool.function(
                absolute_path=str(test_file),
                prompt="Describe this",
                **mock_kwargs,
            )

            call_kwargs = mock_describe.call_args.kwargs
            assert call_kwargs["mime_type"] == "application/octet-stream"
