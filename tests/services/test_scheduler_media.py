"""Media delivery and chat actions of the Telegram SchedulerInterface."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InputMediaDocument, InputMediaPhoto
from telegram.constants import ChatAction, FileSizeLimit

import chibi.services.scheduler_interface as scheduler_interface_module
from chibi.constants import AUDIO_UPLOAD_TIMEOUT, FILE_UPLOAD_TIMEOUT, IMAGE_UPLOAD_TIMEOUT
from chibi.services.scheduler_interface import SchedulerInterface

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_scheduler_bot_singleton():
    """Reset the shared scheduler Bot singleton around every test."""
    scheduler_interface_module._scheduler_bot = None
    yield
    scheduler_interface_module._scheduler_bot = None


@pytest.fixture
def scheduler_bot() -> AsyncMock:
    """Install an AsyncMock as the shared scheduler Bot for the duration of a test."""
    bot = AsyncMock()
    scheduler_interface_module._scheduler_bot = bot
    return bot


@pytest.fixture
def interface() -> SchedulerInterface:
    """A scheduler interface bound to a non-threaded chat."""
    return SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=0)


class TestChatActions:
    """Chat actions go through the scheduler bot and never fail the job."""

    @pytest.mark.parametrize(
        ("method_name", "action"),
        [
            ("send_action_typing", ChatAction.TYPING),
            ("send_action_uploading_photo", ChatAction.UPLOAD_PHOTO),
            ("send_action_recording", ChatAction.RECORD_VOICE),
        ],
    )
    async def test_chat_action_forwarded_to_bot(self, interface, scheduler_bot, method_name, action) -> None:
        """Each chat action sends the matching action with thread_id=None when thread is 0."""
        await getattr(interface, method_name)()

        scheduler_bot.send_chat_action.assert_awaited_once_with(chat_id=555, action=action, message_thread_id=None)

    async def test_chat_action_uses_thread_id_when_set(self, scheduler_bot) -> None:
        """Non-zero thread ids are forwarded as message_thread_id."""
        interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=42)

        await interface.send_action_typing()

        scheduler_bot.send_chat_action.assert_awaited_once_with(
            chat_id=555, action=ChatAction.TYPING, message_thread_id=42
        )

    async def test_chat_action_failure_is_swallowed(self, interface, scheduler_bot) -> None:
        """A failing chat action is logged, not raised — the job must keep running."""
        scheduler_bot.send_chat_action.side_effect = RuntimeError("telegram down")

        await interface.send_action_typing()  # must not raise

        scheduler_bot.send_chat_action.assert_awaited_once()


class TestMediaDelivery:
    """Audio, video and documents mirror TelegramInterface delivery."""

    async def test_send_audio_forwards_all_params(self, interface, scheduler_bot) -> None:
        """send_audio forwards every parameter with HTML parse mode and audio timeouts."""
        await interface.send_audio(
            audio=b"audio-bytes",
            title="Track",
            performer="Artist",
            caption="caption",
            duration=120,
            thumbnail=b"thumb",
            filename="track.mp3",
        )

        scheduler_bot.send_audio.assert_awaited_once_with(
            chat_id=555,
            audio=b"audio-bytes",
            title="Track",
            performer="Artist",
            caption="caption",
            duration=120,
            thumbnail=b"thumb",
            filename="track.mp3",
            parse_mode="HTML",
            message_thread_id=None,
            read_timeout=AUDIO_UPLOAD_TIMEOUT,
            write_timeout=AUDIO_UPLOAD_TIMEOUT,
        )

    async def test_send_video_forwards_all_params(self, interface, scheduler_bot) -> None:
        """send_video forwards every parameter with HTML parse mode and file timeouts."""
        await interface.send_video(
            video=b"video-bytes",
            caption="caption",
            duration=30,
            thumbnail=b"thumb",
            filename="clip.mp4",
        )

        scheduler_bot.send_video.assert_awaited_once_with(
            chat_id=555,
            video=b"video-bytes",
            caption="caption",
            duration=30,
            thumbnail=b"thumb",
            filename="clip.mp4",
            message_thread_id=None,
            parse_mode="HTML",
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )

    async def test_send_document_forwards_all_params(self, interface, scheduler_bot) -> None:
        """send_document forwards every parameter with file timeouts and thread None."""
        await interface.send_document(
            document=b"doc-bytes", filename="report.pdf", caption="caption", thumbnail=b"thumb"
        )

        scheduler_bot.send_document.assert_awaited_once_with(
            chat_id=555,
            document=b"doc-bytes",
            filename="report.pdf",
            caption="caption",
            thumbnail=b"thumb",
            message_thread_id=None,
            read_timeout=FILE_UPLOAD_TIMEOUT,
            write_timeout=FILE_UPLOAD_TIMEOUT,
        )

    async def test_media_uses_thread_id_when_set(self, scheduler_bot) -> None:
        """Non-zero thread ids are forwarded to every media send."""
        interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=555, thread_id=42)

        await interface.send_audio(audio=b"audio")

        scheduler_bot.send_audio.assert_awaited_once()
        assert scheduler_bot.send_audio.await_args.kwargs["message_thread_id"] == 42


class TestSendImagesFromUrls:
    """URL image lists are downloaded and delivered as a photo media group."""

    async def test_urls_sent_as_media_group(self, interface, scheduler_bot) -> None:
        """URLs are downloaded and each becomes an InputMediaPhoto in the media group."""
        with patch("chibi.services.scheduler_interface.download_image", new=AsyncMock(return_value=b"img")):
            await interface.send_images(images=["http://example.com/1.png", "http://example.com/2.png"])

        scheduler_bot.send_media_group.assert_awaited_once()
        kwargs = scheduler_bot.send_media_group.await_args.kwargs
        assert kwargs["chat_id"] == 555
        assert kwargs["message_thread_id"] is None
        assert kwargs["read_timeout"] == IMAGE_UPLOAD_TIMEOUT
        assert kwargs["write_timeout"] == IMAGE_UPLOAD_TIMEOUT
        media = kwargs["media"]
        assert all(isinstance(m, InputMediaPhoto) for m in media)
        assert len(media) == 2

    async def test_media_group_failure_falls_back_to_url_text_message(self, interface, scheduler_bot) -> None:
        """When the media group fails, the original URLs are sent as a text message."""
        scheduler_bot.send_media_group.side_effect = RuntimeError("upload failed")
        urls = ["http://example.com/1.png", "http://example.com/2.png"]

        with patch("chibi.services.scheduler_interface.download_image", new=AsyncMock(return_value=b"img")):
            await interface.send_images(images=urls)

        scheduler_bot.send_message.assert_awaited()
        fallback_text = scheduler_bot.send_message.await_args.kwargs["text"]
        # send_message delivers MarkdownV2 (dots escaped) and markdownify
        # appends a trailing newline — normalize both before comparing.
        assert fallback_text.replace("\\", "").strip() == "\n".join(urls)

    async def test_empty_list_is_a_noop(self, interface, scheduler_bot) -> None:
        """An empty image list sends nothing."""
        await interface.send_images(images=[])

        scheduler_bot.send_media_group.assert_not_awaited()
        scheduler_bot.send_message.assert_not_awaited()


class TestSendImagesFromBytesIO:
    """BytesIO image buffers are partitioned by size."""

    def _buffer(self, size: int) -> BytesIO:
        """Build a BytesIO buffer of the given size."""
        return BytesIO(b"x" * size)

    async def test_small_buffers_sent_as_photos(self, interface, scheduler_bot) -> None:
        """Buffers under PHOTOSIZE_UPLOAD go into the photos media group."""
        small = self._buffer(1024)

        await interface.send_images(images=[small])

        scheduler_bot.send_media_group.assert_awaited_once()
        kwargs = scheduler_bot.send_media_group.await_args.kwargs
        assert all(isinstance(m, InputMediaPhoto) for m in kwargs["media"])
        assert kwargs["write_timeout"] == IMAGE_UPLOAD_TIMEOUT
        assert "read_timeout" not in kwargs

    async def test_large_buffers_sent_as_documents(self, interface, scheduler_bot) -> None:
        """Buffers between PHOTOSIZE_UPLOAD and FILESIZE_UPLOAD are sent as documents."""
        large = self._buffer(FileSizeLimit.PHOTOSIZE_UPLOAD + 1)

        await interface.send_images(images=[large])

        scheduler_bot.send_media_group.assert_awaited_once()
        kwargs = scheduler_bot.send_media_group.await_args.kwargs
        media = kwargs["media"]
        assert all(isinstance(m, InputMediaDocument) for m in media)
        assert len(media) == 1
        assert kwargs["write_timeout"] == FILE_UPLOAD_TIMEOUT

    async def test_oversized_buffers_are_skipped(self, interface, scheduler_bot) -> None:
        """Buffers above FILESIZE_UPLOAD are skipped without raising."""
        oversized = MagicMock(spec=BytesIO)

        # Make the size probe report something beyond the hard file size limit.
        oversized.tell.return_value = FileSizeLimit.FILESIZE_UPLOAD + 1

        await interface.send_images(images=[oversized])

        scheduler_bot.send_media_group.assert_not_awaited()
