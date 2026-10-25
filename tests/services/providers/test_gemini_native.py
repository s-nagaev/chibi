"""Unit tests for Gemini provider OCR functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.gemini_native import Gemini

# Sample PDF bytes for testing
SAMPLE_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF"


# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-ocr"


@pytest.mark.asyncio
async def test_ocr_basic_pdf_extraction():
    """Test basic PDF text extraction with ocr method."""
    provider = Gemini(token=TEST_TOKEN)

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "A test PDF", '
        '"full_description": "This is a test PDF document", '
        '"text": "Extracted text from PDF"}'
    )

    # Patch _generate_content to return our mock response
    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test PDF"
    assert result.full_description == "This is a test PDF document"
    assert result.text == "Extracted text from PDF"


@pytest.mark.asyncio
async def test_ocr_with_custom_model():
    """Test ocr method with custom model parameter."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Custom model test", '
        '"full_description": "Testing with custom model", '
        '"text": "Custom model extracted text"}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        result = await provider.ocr(
            pdf=SAMPLE_PDF_BYTES,
            model="custom-ocr-model",
        )

    assert captured_model == "custom-ocr-model"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_ocr_default_model():
    """Test that default vision model is used when none specified."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Test PDF", "full_description": "Test PDF description", "text": "Test text"}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_ocr_pdf_bytes_passed_correctly():
    """Test that PDF bytes are correctly passed via Part.from_bytes."""
    provider = Gemini(token=TEST_TOKEN)

    captured_contents = None
    captured_mime_type = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_contents, captured_mime_type
        captured_contents = contents
        # Check the Part.from_bytes call in contents
        if isinstance(contents, list) and len(contents) > 0:
            part = contents[0]
            if hasattr(part, "mime_type"):
                captured_mime_type = part.mime_type
            # Also capture the data if available
            if hasattr(part, "data"):
                captured_mime_type = part.mime_type
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        test_pdf = b"%PDF-1.4 test pdf content"
        await provider.ocr(pdf=test_pdf)

    # Verify that contents is a list with the PDF part
    assert isinstance(captured_contents, list)
    assert len(captured_contents) >= 1


@pytest.mark.asyncio
async def test_ocr_part_from_bytes_mime_type():
    """Test that Part.from_bytes is called with correct mime_type='application/pdf'."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    # We need to verify Part.from_bytes is called with the right mime_type
    # The ocr method builds contents as: [Part.from_bytes(data=pdf, mime_type="application/pdf"), prompt]
    with patch("chibi.services.providers.gemini_native.Part.from_bytes") as mock_from_bytes:
        mock_part = MagicMock()
        mock_part.mime_type = "application/pdf"
        mock_from_bytes.return_value = mock_part

        with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

        # Verify Part.from_bytes was called with correct mime_type
        mock_from_bytes.assert_called_once()
        call_kwargs = mock_from_bytes.call_args.kwargs
        assert call_kwargs.get("mime_type") == "application/pdf"
        assert call_kwargs.get("data") == SAMPLE_PDF_BYTES


@pytest.mark.asyncio
async def test_ocr_empty_response_raises_error():
    """Test that empty response raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = None

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_parse_error_raises_error():
    """Test that parse error raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = "invalid json that cannot be parsed"

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "parse" in str(exc_info.value).lower() or "extract" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_returns_vision_result_schema():
    """Test that ocr method returns VisionResultSchema instance."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Invoice", '
        '"full_description": "Invoice for services", '
        '"text": "Invoice #12345\\nDate: 2024-01-01\\nAmount: $100.00"}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "Invoice"
    assert result.full_description == "Invoice for services"
    assert result.text is not None and "Invoice #12345" in result.text


@pytest.mark.asyncio
async def test_ocr_with_text_extraction():
    """Test ocr method when PDF contains text."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Document", '
        '"full_description": "A legal document", '
        '"text": "Important legal text extracted from the PDF document"}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.text == "Important legal text extracted from the PDF document"


@pytest.mark.asyncio
async def test_ocr_optional_text_field():
    """Test that ocr works when text field is null/optional."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Image scan", "full_description": "Scanned image without text", "text": null}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert isinstance(result, VisionResultSchema)
    assert result.text is None


@pytest.mark.asyncio
async def test_ocr_mock_generate_content_called():
    """Test that _generate_content is called correctly with expected parameters."""
    provider = Gemini(token=TEST_TOKEN)

    call_args = {}

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        call_args["model"] = model
        call_args["contents"] = contents
        call_args["config"] = config
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify _generate_content was called
    assert "model" in call_args
    assert "contents" in call_args
    assert "config" in call_args

    # Verify model is set
    assert call_args["model"] == provider.default_vision_model

    # Verify contents structure - should be a list with Part and prompt
    assert isinstance(call_args["contents"], list)
    assert len(call_args["contents"]) == 2  # Part.from_bytes + prompt string


@pytest.mark.asyncio
async def test_ocr_config_has_response_schema():
    """Test that generation config includes VisionResultSchema as response_schema."""
    provider = Gemini(token=TEST_TOKEN)

    captured_config = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": "Test text"}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_config
        captured_config = config
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify config has response_schema set to VisionResultSchema
    assert captured_config is not None
    assert hasattr(captured_config, "response_schema")
    assert captured_config.response_schema == VisionResultSchema
