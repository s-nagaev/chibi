"""Unit tests for Anthropic provider OCR method."""

from unittest.mock import MagicMock

import pytest
from anthropic.types import ToolUseBlock

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.anthropic import Anthropic

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
    provider = Anthropic(token=TEST_TOKEN)

    # Create mock client with proper message response
    mock_client = MagicMock()

    # Create mock response with proper ToolUseBlock type
    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={
            "short_description": "A test PDF document",
            "full_description": "This is a test PDF with some content",
            "text": "Test PDF content extracted from document",
        },
    )
    mock_response.content = [mock_content]

    # Create async mock for messages.create
    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.messages.create = mock_create

    provider.client = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test PDF document"
    assert result.full_description == "This is a test PDF with some content"
    assert result.text == "Test PDF content extracted from document"


@pytest.mark.asyncio
async def test_ocr_with_custom_model():
    """Test ocr method with custom model parameter."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={
            "short_description": "Custom model test",
            "full_description": "Testing OCR with custom model",
            "text": "Custom model OCR result",
        },
    )
    mock_response.content = [mock_content]

    captured_model = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    result = await provider.ocr(
        pdf=SAMPLE_PDF_BYTES,
        model="claude-opus-4-6",
    )

    assert captured_model == "claude-opus-4-6"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_ocr_payload_uses_document_with_base64():
    """Test that ocr method correctly uses 'document' type with base64 encoding."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_content = {}

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={"short_description": "Test", "full_description": "Test", "text": "Test"},
    )
    mock_response.content = [mock_content]

    async def mock_create(*args, **kwargs):
        captured_content.update(kwargs)
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    test_pdf = b"Hello World PDF"
    expected_base64 = "SGVsbG8gV29ybGQgUERG"

    await provider.ocr(pdf=test_pdf)

    messages = captured_content.get("messages", [])
    assert len(messages) > 0
    # Anthropic SDK messages content is a list of blocks, not a dict
    content = messages[0].get("content", [])

    # Find the document block
    document_block = None
    for block in content:
        # Anthropic SDK blocks are objects, but in mock we might have dicts
        # If it's a dict, check type
        if isinstance(block, dict) and block.get("type") == "document":
            document_block = block
            break
        # If it's an object (as in real SDK), check attribute
        elif hasattr(block, "type") and block.type == "document":
            document_block = block
            break

    assert document_block is not None, "Document block should be present in content"
    # Check source
    if isinstance(document_block, dict):
        source = document_block["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == "application/pdf"
        assert source["data"] == expected_base64
    else:
        assert document_block.source.type == "base64"
        assert document_block.source.media_type == "application/pdf"
        assert document_block.source.data == expected_base64


@pytest.mark.asyncio
async def test_ocr_default_model():
    """Test that default vision model is used when none specified."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={"short_description": "Test", "full_description": "Test", "text": "Test"},
    )
    mock_response.content = [mock_content]

    captured_model = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_model
        captured_model = args[1] if len(args) > 1 else kwargs.get("model")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_ocr_empty_response_raises_error():
    """Test that empty response raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_create(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.content = []
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "no tool use block found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_empty_content_raises_error():
    """Test that empty content raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_create(*args, **kwargs):
        mock_response = MagicMock()
        # Return a TextBlock instead of ToolUseBlock to trigger error
        from anthropic.types import TextBlock

        mock_content = TextBlock(type="text", text="some text")
        mock_response.content = [mock_content]
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "no tool use block found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_includes_extraction_prompt():
    """Test that ocr method includes text extraction prompt."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_prompt = {}

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={"short_description": "Test", "full_description": "Test", "text": "Test"},
    )
    mock_response.content = [mock_content]

    async def mock_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            for block in content:
                # Check for dict or object
                if isinstance(block, dict) and block.get("type") == "text":
                    captured_prompt["prompt"] = block.get("text", "")
                elif hasattr(block, "type") and block.type == "text":
                    captured_prompt["prompt"] = block.text
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "prompt" in captured_prompt
    assert "Extract all text from this PDF" in captured_prompt["prompt"]


@pytest.mark.asyncio
async def test_ocr_returns_vision_result_schema():
    """Test that ocr method returns properly validated VisionResultSchema."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={
            "short_description": "Invoice document",
            "full_description": "A standard invoice for services rendered",
            "text": "INVOICE\n#12345\nAmount: $500.00",
        },
    )
    mock_response.content = [mock_content]

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify all fields are correctly parsed
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "Invoice document"
    assert result.full_description == "A standard invoice for services rendered"
    assert result.text == "INVOICE\n#12345\nAmount: $500.00"


@pytest.mark.asyncio
async def test_ocr_mock_called_correctly():
    """Test that the mock client was called with correct parameters."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    call_count = 0

    mock_response = MagicMock()
    mock_content = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="print_pdf_ocr_result",
        input={"short_description": "Test", "full_description": "Test", "text": "Test"},
    )
    mock_response.content = [mock_content]

    async def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify the mock was called
    assert call_count == 1, "messages.create should be called exactly once"
