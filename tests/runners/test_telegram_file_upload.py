"""Tests for the ChibiBot.file_upload handler caption behavior."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.constants import ChatType

from chibi.runners.telegram import ChibiBot


def make_update(
    chat_type: ChatType = ChatType.PRIVATE,
    chat_id: int = 42,
    caption: str | None = None,
    document: MagicMock | None = None,
    photo: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a mocked Telegram update suitable for the file_upload handler.

    Args:
        chat_type: Type of the chat the message arrives in.
        chat_id: Identifier of the chat and the sending user.
        caption: Caption text attached to the message.
        document: Document metadata, if a document was uploaded.
        photo: Photo size variants, if a photo was uploaded.

    Returns:
        The mocked Update object.
    """
    update = MagicMock()
    update.update_id = 1
    update.effective_user = MagicMock(is_bot=False, id=chat_id, username="tester", name="Tester")
    update.effective_chat = MagicMock(
        type=chat_type,
        id=chat_id,
        effective_name="Test chat",
        leave=AsyncMock(),
    )
    message = MagicMock()
    message.caption = caption
    message.document = document
    message.photo = photo
    message.message_thread_id = None
    message.chat = update.effective_chat
    update.effective_message = message
    return update


def make_document(file_name: str, mime_type: str) -> MagicMock:
    """Build a mocked Telegram document.

    Args:
        file_name: Name of the uploaded document.
        mime_type: MIME type reported for the document.

    Returns:
        The mocked Document object.
    """
    document = MagicMock()
    document.file_name = file_name
    document.mime_type = mime_type
    document.to_dict.return_value = {"file_name": file_name, "mime_type": mime_type}
    return document


def make_context() -> MagicMock:
    """Build a mocked PTB callback context.

    Returns:
        The mocked Context object.
    """
    context = MagicMock()
    context.bot = MagicMock(id=999, first_name="Chibi", username="chibibot")
    context.user_data = {}
    return context


@pytest.fixture()
def allowance_settings():
    """Patch the Telegram settings consulted by the allowance decorator.

    Yields:
        The mocked settings object.
    """
    with patch("chibi.utils.telegram.telegram_settings") as settings:
        settings.users_whitelist = set()
        settings.groups_whitelist = {100}
        settings.allow_bots = False
        settings.message_for_disallowed_users = "not allowed"
        yield settings


class TestFileUploadDocumentCaption:
    """Tests for the caption JSON produced for document uploads."""

    @pytest.mark.usefixtures("allowance_settings")
    async def test_document_caption_contains_all_keys(self) -> None:
        """The document caption JSON carries all five backward-compatible keys."""
        update = make_update(document=make_document("notes.md", "text/plain"))
        context = make_context()

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch("chibi.runners.telegram.handle_user_prompt"),
        ):
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            storage_cls.return_value.save = AsyncMock(return_value="file-id-123")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        interface_cls.return_value.set_caption.assert_called_once()
        caption = json.loads(interface_cls.return_value.set_caption.call_args[0][0])
        assert set(caption.keys()) == {"user_caption", "file_id", "file_name", "mime_type", "is_text_file"}
        assert caption["user_caption"] == "no data"
        assert caption["file_id"] == "file-id-123"
        assert caption["file_name"] == "notes.md"
        assert caption["mime_type"] == "text/plain"
        assert caption["is_text_file"] is True

    @pytest.mark.usefixtures("allowance_settings")
    async def test_document_markdown_text_mime_is_text_file(self) -> None:
        """A .md upload with a text MIME type is flagged as a text file."""
        update = make_update(document=make_document("readme.md", "text/plain"))
        context = make_context()

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch("chibi.runners.telegram.handle_user_prompt"),
        ):
            storage_cls.return_value.save = AsyncMock(return_value="file-id-md")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        interface_cls.return_value.set_caption.assert_called_once()
        caption = json.loads(interface_cls.return_value.set_caption.call_args[0][0])
        assert caption["is_text_file"] is True

    @pytest.mark.usefixtures("allowance_settings")
    async def test_document_zip_binary_is_not_text_file(self) -> None:
        """A .zip upload with a binary MIME type is flagged as non-text."""
        update = make_update(document=make_document("archive.zip", "application/zip"))
        context = make_context()

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch("chibi.runners.telegram.handle_user_prompt"),
        ):
            storage_cls.return_value.save = AsyncMock(return_value="file-id-zip")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        interface_cls.return_value.set_caption.assert_called_once()
        caption = json.loads(interface_cls.return_value.set_caption.call_args[0][0])
        assert caption["is_text_file"] is False

    @pytest.mark.usefixtures("allowance_settings")
    async def test_document_with_user_caption_is_preserved(self) -> None:
        """The user caption text is kept verbatim in the enriched JSON."""
        update = make_update(caption="/ask summarize this", document=make_document("doc.txt", "text/plain"))
        context = make_context()

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch("chibi.runners.telegram.handle_user_prompt"),
        ):
            storage_cls.return_value.save = AsyncMock(return_value="file-id-cap")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        interface_cls.return_value.set_caption.assert_called_once()
        caption = json.loads(interface_cls.return_value.set_caption.call_args[0][0])
        assert caption["user_caption"] == "/ask summarize this"


class TestFileUploadPhotoBranch:
    """Tests ensuring the photo branch behavior is unchanged."""

    @pytest.mark.usefixtures("allowance_settings")
    async def test_photo_caption_has_no_document_keys(self) -> None:
        """A photo upload produces the photo caption shape without document keys."""
        photo = MagicMock()
        photo.file_unique_id = "photo-unique-1"
        photo.to_dict.return_value = {"file_unique_id": "photo-unique-1"}
        update = make_update(caption="what is this?", photo=[photo])
        context = make_context()
        vision_result = MagicMock()
        vision_result.full_description = "full"
        vision_result.short_description = "short"
        vision_result.text = ""

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch(
                "chibi.runners.telegram.handle_image_understanding",
                new=AsyncMock(return_value=vision_result),
            ),
        ):
            storage_cls.return_value.save = AsyncMock(return_value="photo-id-1")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        interface_cls.return_value.set_caption.assert_called_once()
        caption = json.loads(interface_cls.return_value.set_caption.call_args[0][0])
        assert set(caption.keys()) == {"user_caption", "photo_short_desc", "file_id"}
        assert caption["file_id"] == "photo-id-1"
        assert caption["photo_short_desc"] == "short"

    @pytest.mark.usefixtures("allowance_settings")
    async def test_document_upload_does_not_trigger_photo_branch(self) -> None:
        """A document upload never invokes the image-understanding path."""
        update = make_update(document=make_document("notes.md", "text/plain"))
        context = make_context()

        with (
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
            patch("chibi.runners.telegram.TelegramInterface") as interface_cls,
            patch("chibi.runners.telegram.task_manager"),
            patch("chibi.runners.telegram.handle_user_prompt"),
            patch(
                "chibi.runners.telegram.handle_image_understanding",
                new=AsyncMock(),
            ) as vision_mock,
        ):
            storage_cls.return_value.save = AsyncMock(return_value="file-id-123")
            interface_cls.return_value.get_caption = AsyncMock(return_value=None)
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        vision_mock.assert_not_called()


class TestFileUploadGroupGating:
    """Tests for the group-chat gating in the file_upload handler."""

    @pytest.mark.usefixtures("allowance_settings")
    async def test_group_upload_without_mention_returns_early(self) -> None:
        """A group upload without /ask or bot interaction is ignored entirely."""
        update = make_update(
            chat_type=ChatType.GROUP,
            chat_id=100,
            caption="just a file",
            document=make_document("notes.md", "text/plain"),
        )
        context = make_context()

        with (
            patch("chibi.runners.telegram.user_interacts_with_bot", return_value=False),
            patch("chibi.runners.telegram.TelegramFileStorage") as storage_cls,
        ):
            await ChibiBot.file_upload(self=None, update=update, context=context)  # type: ignore[arg-type]

        storage_cls.assert_not_called()
