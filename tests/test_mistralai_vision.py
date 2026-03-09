from unittest.mock import MagicMock

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.mistralai_native import MistralAI

# Sample image bytes for testing
SAMPLE_IMAGE_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-vision"


@pytest.mark.asyncio
async def test_vision_basic_image_analysis():
    """Test basic image analysis with vision method."""
    provider = MistralAI(token=TEST_TOKEN)

    # Create mock client
    mock_client = MagicMock()

    # Create a real VisionResultSchema for the mock
    mock_result = VisionResultSchema(
        short_description="A test image",
        full_description="This is a test image description",
        text=None,
    )

    # Create mock for parse
    async def mock_parse(*args, **kwargs):
        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    result = await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test image"
    assert result.full_description == "This is a test image description"


@pytest.mark.asyncio
async def test_vision_with_custom_model():
    """Test vision method with custom model parameter."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    mock_result = VisionResultSchema(
        short_description="Custom model test",
        full_description="Testing with custom model",
        text=None,
    )

    async def mock_parse(*args, **kwargs):
        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    result = await provider.vision(
        image=SAMPLE_IMAGE_BYTES,
        mime_type="image/png",
        model="pixtral-large-latest",
    )

    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_vision_with_text_extraction():
    """Test vision method when image contains text."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    mock_result = VisionResultSchema(
        short_description="Document with text",
        full_description="A document containing important information",
        text="Important information extracted from document",
    )

    async def mock_parse(*args, **kwargs):
        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

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

    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_parse(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = []
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_no_parsed_content_raises_error():
    """Test that missing parsed content raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()

    async def mock_parse(*args, **kwargs):
        mock_message = MagicMock()
        mock_message.parsed = None

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    with pytest.raises(ServiceResponseError) as exc_info:
        await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_base64_encoding():
    """Test that image is correctly encoded to base64."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_kwargs = {}

    mock_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text=None,
    )

    async def mock_parse(*args, **kwargs):
        captured_kwargs.update(kwargs)

        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    test_image = b"Hello World"
    expected_base64 = "SGVsbG8gV29ybGQ="

    await provider.vision(image=test_image, mime_type="image/png")

    image_url = captured_kwargs["messages"][0]["content"][1]["image_url"]["url"]

    assert expected_base64 in image_url
    assert image_url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_vision_default_model():
    """Test that default vision model is used when none provided."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_model = None

    mock_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text=None,
    )

    async def mock_parse(*args, **kwargs):
        nonlocal captured_model
        captured_model = kwargs.get("model")

        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert captured_model == "pixtral-12b-2409"


@pytest.mark.asyncio
async def test_vision_with_custom_prompt():
    """Test vision method with custom prompt parameter."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_prompt = None

    mock_result = VisionResultSchema(
        short_description="Custom prompt test",
        full_description="Testing with custom prompt",
        text=None,
    )

    async def mock_parse(*args, **kwargs):
        nonlocal captured_prompt
        messages = kwargs.get("messages", [])
        # Extract prompt from messages
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                for content in msg["content"]:
                    if isinstance(content, dict) and content.get("type") == "text":
                        captured_prompt = content.get("text")

        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    custom_prompt = "What objects are in this image?"
    result = await provider.vision(
        image=SAMPLE_IMAGE_BYTES,
        mime_type="image/png",
        prompt=custom_prompt,
    )

    assert result.short_description == "Custom prompt test"
    assert captured_prompt == custom_prompt


@pytest.mark.asyncio
async def test_vision_default_prompt():
    """Test that default prompt is used when none provided."""
    provider = MistralAI(token=TEST_TOKEN)

    mock_client = MagicMock()
    captured_prompt = None

    mock_result = VisionResultSchema(
        short_description="Test",
        full_description="Test",
        text=None,
    )

    async def mock_parse(*args, **kwargs):
        nonlocal captured_prompt
        messages = kwargs.get("messages", [])
        # Extract prompt from messages
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                for content in msg["content"]:
                    if isinstance(content, dict) and content.get("type") == "text":
                        captured_prompt = content.get("text")

        mock_message = MagicMock()
        mock_message.parsed = mock_result
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.parse_async = mock_parse

    provider.__dict__["_client"] = mock_client

    # Call without prompt argument
    await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    # Verify default prompt is used
    assert captured_prompt == "Describe the image in detail."
