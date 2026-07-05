"""Unit tests for OpenAI provider OCR method."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chibi.exceptions import NoModelSelectedError, ServiceResponseError
from chibi.schemas.app import ModeratorsAnswer, VisionResultSchema
from chibi.services.providers.openai import OpenAI

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


def create_mock_response(vision_result: VisionResultSchema) -> ChatCompletion:
    """Create a properly typed ChatCompletion mock response."""
    mock_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        parsed=vision_result,
    )
    mock_choice = Choice(
        index=0,
        message=mock_message,
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )


def create_mock_response_with_empty_choices() -> ChatCompletion:
    """Create a ChatCompletion with empty choices."""
    return ChatCompletion(
        id="test-id",
        choices=[],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )


def create_mock_response_with_none_parsed() -> ChatCompletion:
    """Create a ChatCompletion with message.parsed set to None."""
    mock_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        parsed=None,
    )
    mock_choice = Choice(
        index=0,
        message=mock_message,
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-id",
        choices=[mock_choice],
        created=1234567890,
        model="gpt-4.1-mini",
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_ocr_basic_pdf_extraction():
    """Test basic PDF text extraction with ocr method."""
    provider = OpenAI(token=TEST_TOKEN)

    vision_result = VisionResultSchema(
        short_description="A test PDF document",
        full_description="This is a test PDF with some content",
        text="Test PDF content extracted from document",
    )
    mock_response = create_mock_response(vision_result)

    # Create async mock for chat.completions.parse
    async def mock_parse(*args, **kwargs):
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse

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
    provider = OpenAI(token=TEST_TOKEN)

    captured_model = None

    vision_result = VisionResultSchema(
        short_description="Custom model test",
        full_description="Testing OCR with custom model",
        text="Custom model OCR result",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    result = await provider.ocr(
        pdf=SAMPLE_PDF_BYTES,
        model="gpt-4.1",
    )

    assert captured_model == "gpt-4.1"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_ocr_payload_uses_file_with_base64():
    """Test that ocr method correctly uses 'file' type with base64 encoding.

    This test strictly verifies the payload structure:
    {"type": "file", "file": {"file_data": "data:application/pdf;base64,..."}}
    """
    provider = OpenAI(token=TEST_TOKEN)

    captured_messages = {}

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        captured_messages.update(kwargs)
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    test_pdf = b"Hello World PDF"
    expected_base64 = "SGVsbG8gV29ybGQgUERG"

    await provider.ocr(pdf=test_pdf)

    messages = captured_messages.get("messages", [])
    assert len(messages) > 0

    # Get the content from the user message
    user_message = messages[0]
    content = user_message.get("content", [])

    # Find the file block - should be {"type": "file", "file": {"file_data": "..."}}
    file_block = None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "file":
            file_block = block
            break

    assert file_block is not None, "File block should be present in content"
    assert file_block["type"] == "file", "Block type should be 'file'"

    # Verify the file_data structure matches the required format
    assert "file" in file_block, "File block should have 'file' key"
    assert "file_data" in file_block["file"], "File block should have 'file_data' key"

    # Verify the file_data contains the expected base64 PDF data
    file_data = file_block["file"]["file_data"]
    assert file_data.startswith("data:application/pdf;base64,"), (
        f"file_data should start with 'data:application/pdf;base64,' but got: {file_data[:50]}"
    )
    assert file_data.endswith(expected_base64), "file_data should end with the base64-encoded PDF"

    # Verify the full base64 content
    actual_base64 = file_data.replace("data:application/pdf;base64,", "")
    assert actual_base64 == expected_base64, f"Expected base64 {expected_base64}, got {actual_base64}"


@pytest.mark.asyncio
async def test_ocr_default_model():
    """Test that default vision model is used when none specified."""
    provider = OpenAI(token=TEST_TOKEN)

    captured_model = None

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        nonlocal captured_model
        captured_model = args[1] if len(args) > 1 else kwargs.get("model")
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_ocr_empty_response_raises_error():
    """Test that empty response raises ServiceResponseError."""
    provider = OpenAI(token=TEST_TOKEN)

    mock_response = create_mock_response_with_empty_choices()

    async def mock_parse(*args, **kwargs):
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_empty_parsed_raises_error():
    """Test that empty parsed result raises ServiceResponseError."""
    provider = OpenAI(token=TEST_TOKEN)

    mock_response = create_mock_response_with_none_parsed()

    async def mock_parse(*args, **kwargs):
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "empty parsed result" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_includes_extraction_prompt():
    """Test that ocr method includes text extraction prompt."""
    provider = OpenAI(token=TEST_TOKEN)

    captured_prompt = {}

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    captured_prompt["prompt"] = block.get("text", "")
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "prompt" in captured_prompt
    assert "Extract all text from this PDF" in captured_prompt["prompt"]


@pytest.mark.asyncio
async def test_ocr_returns_vision_result_schema():
    """Test that ocr method returns properly validated VisionResultSchema."""
    provider = OpenAI(token=TEST_TOKEN)

    vision_result = VisionResultSchema(
        short_description="Invoice document",
        full_description="A standard invoice for services rendered",
        text="INVOICE\n#12345\nAmount: $500.00",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    result = await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify all fields are correctly parsed
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "Invoice document"
    assert result.full_description == "A standard invoice for services rendered"
    assert result.text == "INVOICE\n#12345\nAmount: $500.00"


@pytest.mark.asyncio
async def test_ocr_uses_parse_method():
    """Test that the ocr method uses chat.completions.parse for structured output."""
    provider = OpenAI(token=TEST_TOKEN)

    call_count = 0

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    # Verify chat.completions.parse was called
    assert call_count == 1, "chat.completions.parse should be called exactly once"


@pytest.mark.asyncio
async def test_ocr_response_format_is_vision_result_schema():
    """Test that the response_format parameter is set to VisionResultSchema."""
    provider = OpenAI(token=TEST_TOKEN)

    captured_response_format = {}

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        captured_response_format["response_format"] = kwargs.get("response_format")
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert captured_response_format.get("response_format") == VisionResultSchema


@pytest.mark.asyncio
async def test_ocr_no_model_raises_error():
    """Test that missing model raises NoModelSelectedError."""

    provider = OpenAI(token=TEST_TOKEN)
    provider.default_ocr_model = None  # type: ignore[assignment]

    mock_client = MagicMock()
    # Ensure parse is awaitable to avoid TypeError if provider attempts to await it.
    mock_client.chat.completions.parse = AsyncMock(side_effect=NoModelSelectedError("No OCR model selected"))
    provider.client = mock_client

    with pytest.raises(NoModelSelectedError) as exc_info:
        await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    assert "no ocr model selected" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ocr_includes_filename_in_file_block():
    """Test that the file block includes the filename."""
    provider = OpenAI(token=TEST_TOKEN)

    captured_messages = {}

    vision_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text="Test",
    )
    mock_response = create_mock_response(vision_result)

    async def mock_parse(*args, **kwargs):
        captured_messages.update(kwargs)
        return mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.parse = mock_parse
    provider.client = mock_client

    await provider.ocr(pdf=SAMPLE_PDF_BYTES)

    messages = captured_messages.get("messages", [])
    content = messages[0].get("content", [])

    # Find the file block
    file_block = None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "file":
            file_block = block
            break

    assert file_block is not None
    assert "file" in file_block
    assert "filename" in file_block["file"]
    assert file_block["file"]["filename"] == "document.pdf"


# ==============================================================================

# ==============================================================================
# moderate_command Characterization Tests
# ==============================================================================


def create_moderation_completion(content: str, model: str = "gpt-5-mini") -> ChatCompletion:
    """Create a ChatCompletion response for the moderator."""
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
    )
    choice = Choice(
        index=0,
        message=message,
        finish_reason="stop",
    )
    return ChatCompletion(
        id="test-moderation-id",
        choices=[choice],
        created=1234567890,
        model=model,
        object="chat.completion",
    )


def create_mock_openai_client(response: ChatCompletion) -> SimpleNamespace:
    """Create a non-callable mock client for OpenAI provider tests.

    OpenAIFriendlyProvider.__getattribute__ wraps any callable attribute,
    so MagicMock cannot be returned directly from the ``client`` property.
    SimpleNamespace is not callable and lets nested attributes pass through.
    """

    async def mock_create(*args: object, **kwargs: object) -> ChatCompletion:
        return response

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock_create),
        ),
    )


@pytest.mark.asyncio
async def test_moderate_command_accepted():
    """Test happy path: moderator returns accepted verdict."""
    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        response = create_moderation_completion('{"verdict": "accepted"}')
        mock_async_openai_class.return_value = create_mock_openai_client(response)
        provider = OpenAI(token=TEST_TOKEN)

        result = await provider.moderate_command(cmd="ls -la")

    assert isinstance(result, ModeratorsAnswer)
    assert result.verdict == "accepted"
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_moderate_command_declined_with_reason():
    """Test happy path: moderator returns declined verdict with reason."""
    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        response = create_moderation_completion('{"verdict": "declined", "reason": "potentially unsafe"}')
        mock_async_openai_class.return_value = create_mock_openai_client(response)
        provider = OpenAI(token=TEST_TOKEN)

        result = await provider.moderate_command(cmd="rm -rf /")

    assert isinstance(result, ModeratorsAnswer)
    assert result.verdict == "declined"
    assert result.reason == "potentially unsafe"
    assert result.status == "operation aborted"


@pytest.mark.asyncio
async def test_moderate_command_strips_json_markdown_wrapper():
    """Test that markdown JSON wrapper is stripped before parsing."""
    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        response = create_moderation_completion('```json\n{"verdict": "accepted"}\n```')
        mock_async_openai_class.return_value = create_mock_openai_client(response)
        provider = OpenAI(token=TEST_TOKEN)

        result = await provider.moderate_command(cmd="ls")

    assert result.verdict == "accepted"


@pytest.mark.asyncio
async def test_moderate_command_temperature_for_mini_model():
    """Test temperature=1 heuristic for models with 'mini' in name."""
    captured_kwargs: dict[str, object] = {}

    async def tracking_create(*args: object, **kwargs: object) -> ChatCompletion:
        captured_kwargs.update(kwargs)
        model_name = cast(str, kwargs.get("model", "unknown"))
        return create_moderation_completion('{"verdict": "accepted"}', model=model_name)

    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        mock_async_openai_class.return_value = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=tracking_create),
            ),
        )
        provider = OpenAI(token=TEST_TOKEN)

        await provider.moderate_command(cmd="ls", model="gpt-5-mini")

    assert captured_kwargs.get("temperature") == 1


@pytest.mark.asyncio
async def test_moderate_command_temperature_for_reasoning_model():
    """Test temperature=1 heuristic for models starting with 'o'."""
    captured_kwargs: dict[str, object] = {}

    async def tracking_create(*args: object, **kwargs: object) -> ChatCompletion:
        captured_kwargs.update(kwargs)
        model_name = cast(str, kwargs.get("model", "unknown"))
        return create_moderation_completion('{"verdict": "accepted"}', model=model_name)

    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        mock_async_openai_class.return_value = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=tracking_create),
            ),
        )
        provider = OpenAI(token=TEST_TOKEN)

        await provider.moderate_command(cmd="ls", model="o3-mini")

    assert captured_kwargs.get("temperature") == 1


@pytest.mark.asyncio
async def test_moderate_command_temperature_zero_for_regular_model():
    """Test temperature=0 for regular models without mini/o/nano prefix."""
    captured_kwargs: dict[str, object] = {}

    async def tracking_create(*args: object, **kwargs: object) -> ChatCompletion:
        captured_kwargs.update(kwargs)
        model_name = cast(str, kwargs.get("model", "unknown"))
        return create_moderation_completion('{"verdict": "accepted"}', model=model_name)

    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        mock_async_openai_class.return_value = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=tracking_create),
            ),
        )
        provider = OpenAI(token=TEST_TOKEN)

        await provider.moderate_command(cmd="ls", model="gpt-4")

    assert captured_kwargs.get("temperature") == 0.0


@pytest.mark.asyncio
async def test_moderate_command_invalid_json_returns_declined():
    """Test invalid JSON response returns declined with error status."""
    with patch("chibi.services.providers.provider.AsyncOpenAI") as mock_async_openai_class:
        response = create_moderation_completion("this is not valid json")
        mock_async_openai_class.return_value = create_mock_openai_client(response)
        provider = OpenAI(token=TEST_TOKEN)

        result = await provider.moderate_command(cmd="ls")

    assert result.verdict == "declined"
    assert result.status == "error"
