"""Tests for describe_image function in chibi/services/user.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.schemas.app import VisionResultSchema
from chibi.services import user as user_service


@pytest.mark.asyncio
async def test_describe_image_without_prompt():
    """Test describe_image works without prompt parameter (backward compatibility)."""
    mock_provider = MagicMock()
    mock_user = MagicMock()
    mock_user.vision_provider = mock_provider

    mock_result = VisionResultSchema(
        short_description="Test image",
        full_description="A test image description",
        text=None,
    )
    mock_provider.vision = AsyncMock(return_value=mock_result)

    # Use the actual function path and patch the database dependency
    with patch("chibi.storage.database._db_provider") as mock_db_provider:
        mock_db = MagicMock()
        mock_db.get_or_create_user = AsyncMock(return_value=mock_user)
        mock_db_provider.get_database = AsyncMock(return_value=mock_db)

        # Call the function - inject_database will inject the db
        result = await user_service.describe_image(
            user_id=12345,
            image=b"fake_image_bytes",
            mime_type="image/png",
        )

    # Verify provider.vision was called without prompt
    mock_provider.vision.assert_called_once()
    call_kwargs = mock_provider.vision.call_args.kwargs
    assert call_kwargs["image"] == b"fake_image_bytes"
    assert call_kwargs["mime_type"] == "image/png"
    assert call_kwargs.get("prompt") is None

    assert result.short_description == "Test image"


@pytest.mark.asyncio
async def test_describe_image_with_prompt():
    """Test describe_image passes prompt to provider.vision."""
    mock_provider = MagicMock()
    mock_user = MagicMock()
    mock_user.vision_provider = mock_provider

    mock_result = VisionResultSchema(
        short_description="Custom prompt result",
        full_description="Analysis based on custom prompt",
        text=None,
    )
    mock_provider.vision = AsyncMock(return_value=mock_result)

    with patch("chibi.storage.database._db_provider") as mock_db_provider:
        mock_db = MagicMock()
        mock_db.get_or_create_user = AsyncMock(return_value=mock_user)
        mock_db_provider.get_database = AsyncMock(return_value=mock_db)

        custom_prompt = "What colors are in this image?"
        result = await user_service.describe_image(
            user_id=12345,
            image=b"fake_image_bytes",
            mime_type="image/png",
            prompt=custom_prompt,
        )

    # Verify provider.vision was called with prompt
    mock_provider.vision.assert_called_once()
    call_kwargs = mock_provider.vision.call_args.kwargs
    assert call_kwargs["image"] == b"fake_image_bytes"
    assert call_kwargs["mime_type"] == "image/png"
    assert call_kwargs["prompt"] == custom_prompt

    assert result.short_description == "Custom prompt result"


@pytest.mark.asyncio
async def test_describe_image_with_model_and_prompt():
    """Test describe_image works with both model and prompt parameters."""
    mock_provider = MagicMock()
    mock_user = MagicMock()
    mock_user.vision_provider = mock_provider

    mock_result = VisionResultSchema(
        short_description="Full options test",
        full_description="Testing with model and prompt",
        text=None,
    )
    mock_provider.vision = AsyncMock(return_value=mock_result)

    with patch("chibi.storage.database._db_provider") as mock_db_provider:
        mock_db = MagicMock()
        mock_db.get_or_create_user = AsyncMock(return_value=mock_user)
        mock_db_provider.get_database = AsyncMock(return_value=mock_db)

        result = await user_service.describe_image(
            user_id=12345,
            image=b"fake_image_bytes",
            mime_type="image/jpeg",
            model="gpt-4.1",
            prompt="Describe the main subject",
        )

    # Verify provider.vision was called with all parameters
    mock_provider.vision.assert_called_once()
    call_kwargs = mock_provider.vision.call_args.kwargs
    assert call_kwargs["image"] == b"fake_image_bytes"
    assert call_kwargs["mime_type"] == "image/jpeg"
    assert call_kwargs["model"] == "gpt-4.1"
    assert call_kwargs["prompt"] == "Describe the main subject"

    assert result.short_description == "Full options test"


@pytest.mark.asyncio
async def test_describe_image_prompt_none_passes_none():
    """Test describe_image with explicit prompt=None passes None to provider."""
    mock_provider = MagicMock()
    mock_user = MagicMock()
    mock_user.vision_provider = mock_provider

    mock_result = VisionResultSchema(
        short_description="None prompt test",
        full_description="Testing explicit None",
        text=None,
    )
    mock_provider.vision = AsyncMock(return_value=mock_result)

    with patch("chibi.storage.database._db_provider") as mock_db_provider:
        mock_db = MagicMock()
        mock_db.get_or_create_user = AsyncMock(return_value=mock_user)
        mock_db_provider.get_database = AsyncMock(return_value=mock_db)

        result = await user_service.describe_image(
            user_id=12345,
            image=b"fake_image_bytes",
            mime_type="image/png",
            prompt=None,
        )

    # Verify provider.vision was called with prompt=None
    mock_provider.vision.assert_called_once()
    call_kwargs = mock_provider.vision.call_args.kwargs
    assert "prompt" in call_kwargs
    assert call_kwargs["prompt"] is None

    assert result.short_description == "None prompt test"
