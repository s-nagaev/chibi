"""Unit tests for MistralAI provider OCR method."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.provider import ServiceResponseError

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

# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-ocr"


@pytest.mark.asyncio
async def test_ocr_basic_pdf_extraction():
    """Test basic PDF text extraction with ocr method."""
    provider = MistralAI(token=TEST_TOKEN)

    # Create mock client
    mock_client = MagicMock()

    # Create mock OCR response with pages
    mock_page1 = MagicMock()
    mock_page1.markdown = "Page 1 text content"
    mock_page2 = MagicMock()
    mock_page2.markdown = "Page 2 text content"

    mock_response = MagicMock()
    mock_response.pages = [mock_page1, mock_page2]

    async def mock_process(*args, **kwargs):
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    # short_description is the first 100 chars of combined text (with newlines replaced by spaces)
    assert result.short_description.startswith("Page 1 text content")
    assert result.full_description is not None and "Page 1 text content" in result.full_description
    assert result.text is not None and "Page 2 text content" in result.text


@pytest.mark.asyncio
async def test_ocr_with_custom_model():
    """Test ocr method with custom model parameter."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_model = None

    mock_page = MagicMock()
    mock_page.markdown = "Custom model test"

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    result = await provider.ocr(
        pdf=SAMPLE_PDF_BYTES,
        model="mistral-ocr-latest-v2",
    )

    assert captured_model == "mistral-ocr-latest-v2"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_ocr_payload_uses_document_url_chunk():
    """Test that ocr method correctly uses DocumentURLChunk with base64 encoded PDF.

    This test strictly verifies the payload structure for the OCR API call.
    """
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_kwargs = {}

    mock_page = MagicMock()
    mock_page.markdown = "Test PDF content"

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    test_pdf = b"Hello World PDF"
    expected_base64 = "SGVsbG8gV29ybGQgUERG"

    await provider.ocr(pdf=test_pdf)

    # Verify the document parameter was passed
    document = captured_kwargs.get("document")
    assert document is not None, "Document parameter should be present"

    # Verify DocumentURLChunk was used with correct base64 data URL
    from mistralai.models import DocumentURLChunk

    assert isinstance(document, DocumentURLChunk), f"Expected DocumentURLChunk, got {type(document)}"

    expected_url = f"data:application/pdf;base64,{expected_base64}"
    assert document.document_url == expected_url, (
        f"Expected document_url to be '{expected_url}', got '{document.document_url}'"
    )


@pytest.mark.asyncio
async def test_ocr_default_model():
    """Test that default OCR model is used when none specified."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_model = None

    mock_page = MagicMock()
    mock_page.markdown = "Default model test"

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_model == "mistral-ocr-latest"


@pytest.mark.asyncio
async def test_ocr_empty_response_raises_error():
    """Test that empty response raises ServiceResponseError."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.pages = []

    async def mock_process(*args, **kwargs):
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "empty response" in str(exc_info.value).lower()
    assert "MistralAI" in str(exc_info.value)
    assert "mistral-ocr-latest" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ocr_mock_called_once():
    """Test that ocr.process_async is called exactly once with correct parameters."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    call_count = 0
    captured_args: dict[str, Any] = {}

    mock_page = MagicMock()
    mock_page.markdown = "Test content"

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES, model="custom-ocr-model")

    # Verify mock was called exactly once
    assert call_count == 1, f"Expected process_async to be called once, but was called {call_count} times"

    # Verify correct parameters were passed
    assert captured_args["kwargs"]["model"] == "custom-ocr-model"
    assert captured_args["kwargs"]["document"] is not None


@pytest.mark.asyncio
async def test_ocr_multiple_pages_combined():
    """Test that text from multiple pages is properly combined."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    mock_page1 = MagicMock()
    mock_page1.markdown = "First page content"
    mock_page2 = MagicMock()
    mock_page2.markdown = "Second page content"
    mock_page3 = MagicMock()
    mock_page3.markdown = "Third page content"

    mock_response = MagicMock()
    mock_response.pages = [mock_page1, mock_page2, mock_page3]

    async def mock_process(*args, **kwargs):
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify all pages are combined
    text_content = result.text or ""
    assert "First page content" in text_content
    assert "Second page content" in text_content
    assert "Third page content" in text_content

    # Verify pages are separated by newlines
    assert "\n\n" in text_content


@pytest.mark.asyncio
async def test_ocr_vision_result_schema_output():
    """Test that ocr method returns properly formatted VisionResultSchema."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    long_text = "A" * 1500  # Text longer than 1000 chars to test truncation

    mock_page = MagicMock()
    mock_page.markdown = long_text

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify result is VisionResultSchema
    assert isinstance(result, VisionResultSchema)

    # Verify short_description is truncated (first 100 chars + "...")
    assert len(result.short_description) <= 103  # 100 chars + "..."
    assert result.short_description.endswith("...")

    # Verify full_description is limited to 1000 chars
    assert len(result.full_description) <= 1000

    # Verify text contains full content
    assert result.text == long_text


@pytest.mark.asyncio
async def test_ocr_exception_handling():
    """Test that exceptions during OCR are properly wrapped in ServiceResponseError."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_process(*args, **kwargs):
        raise Exception("Connection timeout")

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "Connection timeout" in str(exc_info.value)
    assert "MistralAI" in str(exc_info.value)
    assert "mistral-ocr-latest" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ocr_short_description_formatting():
    """Test that short_description is properly formatted from first 100 chars."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    # Multi-line text to test newlines are replaced with spaces
    mock_page = MagicMock()
    mock_page.markdown = "Line one\nLine two\nLine three with more content here"

    mock_response = MagicMock()
    mock_response.pages = [mock_page]

    async def mock_process(*args, **kwargs):
        return mock_response

    mock_client.ocr.process_async = mock_process

    provider.__dict__["_client"] = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify newlines are replaced with spaces in short_description
    assert "\n" not in result.short_description
    assert "Line one Line two" in result.short_description
