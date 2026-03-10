from unittest.mock import MagicMock

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.anthropic import Anthropic

# Sample image bytes for testing
SAMPLE_IMAGE_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-vision"


@pytest.mark.asyncio
async def test_vision_basic_image_analysis():
    """Test basic image analysis with vision method."""
    provider = Anthropic(token=TEST_TOKEN)

    # Create mock client with proper message response
    mock_client = MagicMock()

    # Create mock response with proper ToolUseBlock type
    mock_response = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "A test image",
            "full_description": "This is a test image description",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    # Create async mock for messages.create
    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.messages.create = mock_create

    provider.client = mock_client

    result = await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test image"
    assert result.full_description == "This is a test image description"


@pytest.mark.asyncio
async def test_vision_with_custom_model():
    """Test vision method with custom model parameter."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Custom model test",
            "full_description": "Testing with custom model",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    captured_model = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    result = await provider.vision(
        image=SAMPLE_IMAGE_BYTES,
        mime_type="image/png",
        model="claude-opus-4-6",
    )

    assert captured_model == "claude-opus-4-6"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_vision_with_text_extraction():
    """Test vision method when image contains text."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Document with text",
            "full_description": "A document containing important information",
            "text": "Important information extracted from document",
        },
    )
    mock_response.content = [mock_tool_call]

    async def mock_create(*args, **kwargs):
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    result = await provider.vision(
        image=SAMPLE_IMAGE_BYTES,
        mime_type="image/jpeg",
    )

    assert isinstance(result, VisionResultSchema)
    assert result.text == "Important information extracted from document"


@pytest.mark.asyncio
async def test_vision_empty_response_raises_error():
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
        await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_empty_content_raises_error():
    """Test that content without ToolUseBlock raises ServiceResponseError."""
    from anthropic.types import TextBlock

    from chibi.services.providers.provider import ServiceResponseError

    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_create(*args, **kwargs):
        mock_response = MagicMock()
        # Use TextBlock instead of ToolUseBlock to simulate missing tool call
        mock_content = TextBlock(type="text", text="Some text response without tool")
        mock_response.content = [mock_content]
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_base64_encoding():
    """Test that image is correctly encoded to base64."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_content = {}

    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Test",
            "full_description": "Test",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    async def mock_create(*args, **kwargs):
        captured_content.update(kwargs)
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    test_image = b"Hello World"
    expected_base64 = "SGVsbG8gV29ybGQ="

    await provider.vision(image=test_image, mime_type="image/png")

    messages = captured_content.get("messages", [])
    assert len(messages) > 0
    content = messages[0].get("content", [])

    # Find the image block
    image_block = None
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            image_block = block
            break

    assert image_block is not None
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == expected_base64


@pytest.mark.asyncio
async def test_vision_default_model():
    """Test that default vision model is used when none specified."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Test",
            "full_description": "Test",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    captured_model = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_model
        captured_model = args[1] if len(args) > 1 else kwargs.get("model")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_vision_with_custom_prompt():
    """Test vision method with custom prompt parameter."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Test",
            "full_description": "Test with custom prompt",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    captured_prompt = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_prompt
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    captured_prompt = block.get("text", "")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    custom_prompt = "Analyze this image for objects and colors"
    result = await provider.vision(
        image=SAMPLE_IMAGE_BYTES,
        mime_type="image/png",
        prompt=custom_prompt,
    )

    assert captured_prompt is not None
    assert custom_prompt in captured_prompt
    assert result.full_description == "Test with custom prompt"


@pytest.mark.asyncio
async def test_vision_default_prompt():
    """Test that default prompt is used when none specified."""
    provider = Anthropic(token=TEST_TOKEN)

    mock_client = MagicMock()
    from anthropic.types import ToolUseBlock

    mock_response = MagicMock()
    mock_tool_call = ToolUseBlock(
        type="tool_use",
        id="test-id-123",
        name="analyze_and_describe_image",
        input={
            "short_description": "Test",
            "full_description": "Test with default prompt",
            "text": None,
        },
    )
    mock_response.content = [mock_tool_call]

    captured_prompt = None

    async def mock_create(*args, **kwargs):
        nonlocal captured_prompt
        messages = kwargs.get("messages", [])
        if messages:
            content = messages[0].get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    captured_prompt = block.get("text", "")
        return mock_response

    mock_client.messages.create = mock_create
    provider.client = mock_client

    result = await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert captured_prompt is not None
    assert "Describe the image in detail" in captured_prompt
    assert result.full_description == "Test with default prompt"
