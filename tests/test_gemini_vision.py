from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.gemini_native import Gemini

# Sample image bytes for testing
SAMPLE_IMAGE_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


# Use a test token for provider instantiation
TEST_TOKEN = "test-token-for-vision"


@pytest.mark.asyncio
async def test_vision_basic_image_analysis():
    """Test basic image analysis with vision method."""
    provider = Gemini(token=TEST_TOKEN)

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "A test image", "full_description": "This is a test image description", "text": null}'
    )

    # Patch _generate_content to return our mock response
    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        result = await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    # Verify the result
    assert isinstance(result, VisionResultSchema)
    assert result.short_description == "A test image"
    assert result.full_description == "This is a test image description"


@pytest.mark.asyncio
async def test_vision_with_custom_model():
    """Test vision method with custom model parameter."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Custom model test", "full_description": "Testing with custom model", "text": null}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        result = await provider.vision(
            image=SAMPLE_IMAGE_BYTES,
            mime_type="image/png",
            model="custom-vision-model",
        )

    assert captured_model == "custom-vision-model"
    assert result.short_description == "Custom model test"


@pytest.mark.asyncio
async def test_vision_with_custom_prompt():
    """Test vision method with custom prompt argument."""
    provider = Gemini(token=TEST_TOKEN)

    captured_prompt = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Custom prompt test", "full_description": "Testing custom prompt", "text": null}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_prompt
        # Contents is a list, second element is the prompt text
        if isinstance(contents, list) and len(contents) > 1:
            captured_prompt = contents[1]
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        custom_prompt = "What colors are in this image?"
        result = await provider.vision(
            image=SAMPLE_IMAGE_BYTES,
            mime_type="image/png",
            prompt=custom_prompt,
        )

    assert captured_prompt == custom_prompt
    assert result.short_description == "Custom prompt test"


@pytest.mark.asyncio
async def test_vision_default_prompt():
    """Test that default prompt is used when no prompt is provided."""
    provider = Gemini(token=TEST_TOKEN)

    captured_prompt = None

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Default prompt test", "full_description": "Testing default prompt", "text": null}'
    )

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_prompt
        if isinstance(contents, list) and len(contents) > 1:
            captured_prompt = contents[1]
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        # Call without prompt argument
        result = await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert captured_prompt == "Describe the image in detail"
    assert result.short_description == "Default prompt test"


@pytest.mark.asyncio
async def test_vision_with_text_extraction():
    """Test vision method when image contains text."""
    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = (
        '{"short_description": "Document with text", '
        '"full_description": "A document containing important information", '
        '"text": "Important information extracted from document"}'
    )

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

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

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = None

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_parse_error_raises_error():
    """Test that parse error raises ServiceResponseError."""
    from chibi.services.providers.provider import ServiceResponseError

    provider = Gemini(token=TEST_TOKEN)

    mock_response = MagicMock()
    mock_response.text = "invalid json that cannot be parsed"

    with patch.object(provider, "_generate_content", new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response

        with pytest.raises(ServiceResponseError) as exc_info:
            await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert "parse" in str(exc_info.value).lower() or "analyze" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_vision_default_model():
    """Test that default vision model is used when none specified."""
    provider = Gemini(token=TEST_TOKEN)

    captured_model = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": null}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_model
        captured_model = model
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        await provider.vision(image=SAMPLE_IMAGE_BYTES, mime_type="image/png")

    assert captured_model == provider.default_vision_model


@pytest.mark.asyncio
async def test_vision_image_bytes_passed_correctly():
    """Test that image bytes are correctly passed to the API."""
    provider = Gemini(token=TEST_TOKEN)

    captured_contents = None

    mock_response = MagicMock()
    mock_response.text = '{"short_description": "Test", "full_description": "Test", "text": null}'

    async def mock_generate_content(self, model, contents, config):
        nonlocal captured_contents
        captured_contents = contents
        return mock_response

    with patch.object(Gemini, "_generate_content", mock_generate_content):
        test_image = b"Hello World"
        await provider.vision(image=test_image, mime_type="image/png")

    # Verify that contents is a list with the image part
    assert isinstance(captured_contents, list)
    assert len(captured_contents) >= 1
