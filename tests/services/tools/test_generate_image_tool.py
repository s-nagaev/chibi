"""Tests for image-to-image reference handling in the generate_image tool."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chibi.config import gpt_settings
from chibi.services.providers.tools import media
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.media import (
    MAX_REFERENCE_IMAGE_BYTES,
    MAX_REFERENCE_IMAGES,
    GenerateImageTool,
    _normalize_reference_mime,
)

PROVIDER_NAME = "Gemini"
CONTENT_MODEL = "gemini-2.5-flash-image"


@pytest.fixture(scope="function")
def mock_interface() -> Mock:
    """Return a mock user interface with a fixed user id."""
    interface = Mock()
    interface.user_id = 12345
    interface.send_images = AsyncMock()
    return interface


@pytest.fixture(scope="function")
def mock_provider() -> Mock:
    """Return a mock provider that accepts image input."""
    provider = Mock()
    provider.image_to_image_ready = True
    provider.supports_image_input = Mock(return_value=True)
    return provider


@pytest.fixture(scope="function")
def mock_user(mock_provider: Mock) -> Mock:
    """Return a mock user whose provider manager yields the mock provider."""
    user = Mock()
    user.providers.get = Mock(return_value=mock_provider)
    return user


@pytest.fixture(scope="function")
def mock_storage() -> AsyncMock:
    """Return a mock file storage serving a JPEG reference image."""
    storage = AsyncMock()
    storage.get_file_info = AsyncMock(return_value={"mime_type": "image/jpeg"})
    storage.get_bytes = AsyncMock(return_value=b"file-bytes")
    return storage


@pytest.fixture(scope="function")
def generate_mock(mock_user: Mock, mock_storage: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch media-module collaborators and return the generate_image mock."""
    monkeypatch.setattr(media, "user_has_reached_images_generation_limit", AsyncMock(return_value=False))
    monkeypatch.setattr(media, "get_chibi_user", AsyncMock(return_value=mock_user))
    monkeypatch.setattr(media, "get_file_storage", Mock(return_value=mock_storage))
    generate = AsyncMock(return_value=[])
    monkeypatch.setattr(media, "generate_image", generate)
    return generate


async def call_tool(interface: Mock, **overrides: object) -> dict[str, str]:
    """Invoke GenerateImageTool.function with standard arguments.

    Args:
        interface: Mock user interface passed to the tool.
        **overrides: Optional overrides for any tool argument (e.g. reference inputs).

    Returns:
        The tool result dictionary.
    """
    arguments: dict[str, Any] = {
        "provider": PROVIDER_NAME,
        "image_model": CONTENT_MODEL,
        "prompt": "a cat",
        "interface": interface,
        "user_id": 12345,
    }
    arguments.update(overrides)
    return await GenerateImageTool.function(**arguments)


def test_normalize_reference_mime_falls_back_to_png() -> None:
    """Unknown MIME types map to image/png, known ones pass through."""
    assert _normalize_reference_mime(mime=None) == "image/png"
    assert _normalize_reference_mime(mime="") == "image/png"
    assert _normalize_reference_mime(mime="application/octet-stream") == "image/png"
    assert _normalize_reference_mime(mime="image/webp") == "image/webp"


@pytest.mark.asyncio
async def test_file_id_resolution_passes_bytes_and_mime(mock_interface: Mock, generate_mock: AsyncMock) -> None:
    """A reference_file_id resolves to bytes plus MIME and reaches generate_image."""
    result = await call_tool(mock_interface, reference_file_ids=["abc123"])

    assert result == {"detail": "Image was successfully generated and sent to user."}
    generate_mock.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model=CONTENT_MODEL,
        prompt="a cat",
        images=[(b"file-bytes", "image/jpeg")],
    )


@pytest.mark.asyncio
async def test_reference_path_expands_tilde(
    mock_interface: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A tilde path resolves against the user home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)
    (tmp_path / "home-ref.png").write_bytes(b"tilde-bytes")

    await call_tool(mock_interface, reference_paths=["~/home-ref.png"])

    generate_mock.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model=CONTENT_MODEL,
        prompt="a cat",
        images=[(b"tilde-bytes", "image/png")],
    )


@pytest.mark.asyncio
async def test_reference_path_resolves_relative(
    mock_interface: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A relative path resolves against the current working directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)
    (tmp_path / "rel-ref.png").write_bytes(b"relative-bytes")

    await call_tool(mock_interface, reference_paths=["rel-ref.png"])

    generate_mock.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model=CONTENT_MODEL,
        prompt="a cat",
        images=[(b"relative-bytes", "image/png")],
    )


@pytest.mark.asyncio
async def test_missing_reference_path_raises_tool_exception(
    mock_interface: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A non-existent reference path raises ToolException naming the path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)
    missing = tmp_path / "missing.png"

    with pytest.raises(ToolException, match="missing.png"):
        await call_tool(mock_interface, reference_paths=[str(missing)])


@pytest.mark.asyncio
async def test_missing_file_id_raises_tool_exception(
    mock_interface: Mock, mock_storage: AsyncMock, generate_mock: AsyncMock
) -> None:
    """An unresolvable file ID raises ToolException naming the ID."""
    mock_storage.get_file_info = AsyncMock(side_effect=FileNotFoundError("no such file"))

    with pytest.raises(ToolException, match="abc123"):
        await call_tool(mock_interface, reference_file_ids=["abc123"])


@pytest.mark.asyncio
async def test_reference_paths_require_filesystem_access(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """Local reference paths are rejected when filesystem access is disabled."""
    monkeypatch.setattr(gpt_settings, "filesystem_access", False)

    with pytest.raises(ToolException, match="filesystem"):
        await call_tool(mock_interface, reference_paths=["/tmp/whatever.png"])


@pytest.mark.asyncio
async def test_combined_file_ids_and_paths_preserve_order(
    mock_interface: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """Combined file IDs and paths resolve in order: file IDs first, then paths."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)
    (tmp_path / "second.png").write_bytes(b"path-bytes")

    await call_tool(mock_interface, reference_file_ids=["abc123"], reference_paths=["second.png"])

    generate_mock.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model=CONTENT_MODEL,
        prompt="a cat",
        images=[(b"file-bytes", "image/jpeg"), (b"path-bytes", "image/png")],
    )


@pytest.mark.asyncio
async def test_unknown_file_mime_falls_back_to_png(
    mock_interface: Mock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A reference file with an unknown extension passes image/png to generate_image."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpt_settings, "filesystem_access", True)
    (tmp_path / "mystery.bin").write_bytes(b"mystery-bytes")

    await call_tool(mock_interface, reference_paths=["mystery.bin"])

    generate_mock.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model=CONTENT_MODEL,
        prompt="a cat",
        images=[(b"mystery-bytes", "image/png")],
    )


@pytest.mark.asyncio
async def test_too_many_reference_images_raises(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """More than the allowed number of reference images raises ToolException."""
    monkeypatch.setattr(media, "_resolve_reference_from_file_id", AsyncMock(return_value=(b"x", "image/png")))
    file_ids = [f"id-{index}" for index in range(MAX_REFERENCE_IMAGES + 1)]

    with pytest.raises(ToolException, match="Too many reference images"):
        await call_tool(mock_interface, reference_file_ids=file_ids)


@pytest.mark.asyncio
async def test_oversized_reference_image_raises(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A reference image above the size limit raises ToolException."""
    oversized = b"x" * (MAX_REFERENCE_IMAGE_BYTES + 1)
    monkeypatch.setattr(media, "_resolve_reference_from_file_id", AsyncMock(return_value=(oversized, "image/png")))

    with pytest.raises(ToolException, match="10 MB"):
        await call_tool(mock_interface, reference_file_ids=["abc123"])


@pytest.mark.asyncio
async def test_disallowed_reference_mime_raises(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """A non-image reference MIME type raises ToolException."""
    monkeypatch.setattr(
        media,
        "_resolve_reference_from_file_id",
        AsyncMock(return_value=(b"%PDF-1.4", "application/pdf")),
    )

    with pytest.raises(ToolException, match="application/pdf"):
        await call_tool(mock_interface, reference_file_ids=["abc123"])


@pytest.mark.asyncio
async def test_disallowed_image_mime_raises(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch, generate_mock: AsyncMock
) -> None:
    """An image type outside PNG/JPEG/WebP raises ToolException."""
    monkeypatch.setattr(media, "_resolve_reference_from_file_id", AsyncMock(return_value=(b"bytes", "image/tiff")))

    with pytest.raises(ToolException, match="image/tiff"):
        await call_tool(mock_interface, reference_file_ids=["abc123"])


@pytest.mark.asyncio
async def test_provider_without_image_input_support_raises(
    mock_interface: Mock, mock_user: Mock, generate_mock: AsyncMock
) -> None:
    """A provider that is not image-to-image ready is named in the error."""
    provider = SimpleNamespace(image_to_image_ready=False, supports_image_input=Mock(return_value=False))
    mock_user.providers.get = Mock(return_value=provider)

    with pytest.raises(ToolException, match=f"Provider '{PROVIDER_NAME}'"):
        await call_tool(mock_interface, reference_file_ids=["abc123"])


@pytest.mark.asyncio
async def test_model_without_image_input_support_raises(
    mock_interface: Mock, mock_provider: Mock, generate_mock: AsyncMock
) -> None:
    """An image-to-image ready provider still rejects models without image input."""
    mock_provider.supports_image_input = Mock(return_value=False)

    with pytest.raises(ToolException, match="dall-e-3"):
        await call_tool(mock_interface, reference_file_ids=["abc123"], image_model="dall-e-3")


@pytest.mark.asyncio
async def test_no_reference_input_skips_capability_checks(
    mock_interface: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without reference input the tool behaves as before and never inspects capabilities."""
    monkeypatch.setattr(media, "user_has_reached_images_generation_limit", AsyncMock(return_value=False))
    get_user_mock = AsyncMock(return_value=Mock())
    monkeypatch.setattr(media, "get_chibi_user", get_user_mock)
    generate = AsyncMock(return_value=[])
    monkeypatch.setattr(media, "generate_image", generate)

    result = await call_tool(mock_interface, image_model="dall-e-3")

    assert result == {"detail": "Image was successfully generated and sent to user."}
    generate.assert_awaited_once_with(
        interface=mock_interface,
        provider_name=PROVIDER_NAME,
        model="dall-e-3",
        prompt="a cat",
        images=None,
    )
    get_user_mock.assert_not_awaited()
