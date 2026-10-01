"""Runner-aware scheduler interface selection and stdio failure delivery."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import chibi.services.jobs.agent_task as agent_task_module
from chibi.constants import IDE_STORAGE_ID
from chibi.schemas.scheduler import AgentJobPayload
from chibi.services.jobs.agent_task import _build_interface, _notify_failure
from chibi.services.scheduler_interface import (
    SchedulerInterface,
    StdioSchedulerInterface,
    clear_stdio_delivery_emitter,
    get_stdio_delivery_emitter,
    set_stdio_delivery_emitter,
)

pytestmark = pytest.mark.asyncio

STDIO_PAYLOAD_KWARGS = {
    "user_id": IDE_STORAGE_ID,
    "storage_id": IDE_STORAGE_ID,
    "chat_id": IDE_STORAGE_ID,
    "thread_id": 42,
    "title": "Stdio heartbeat",
    "action": {"type": "notify", "message": "Stdio heartbeat"},
}


@pytest.fixture(autouse=True)
def _clean_emitter():
    """Ensure the module-level emitter registry is clean around every test."""
    clear_stdio_delivery_emitter()
    yield
    clear_stdio_delivery_emitter()


@pytest.fixture(autouse=True)
def _reset_anti_flood_state():
    """Reset the in-memory anti-flood state around every test."""
    agent_task_module._failure_notify_timestamps.clear()
    yield
    agent_task_module._failure_notify_timestamps.clear()


class TestBuildInterfaceSelection:
    """_build_interface switches on application_settings.client."""

    async def test_telegram_client_gets_telegram_interface(self) -> None:
        """The Telegram process keeps delivering through SchedulerInterface."""
        settings = SimpleNamespace(client="telegram")
        with patch("chibi.services.jobs.agent_task.application_settings", settings):
            interface = _build_interface(
                AgentJobPayload.model_validate(
                    {
                        "job_id": "agent:1:job",
                        **{
                            "user_id": 1,
                            "storage_id": 1,
                            "chat_id": 555,
                            "thread_id": 42,
                            "title": "T",
                            "action": {"type": "notify", "message": "x"},
                        },
                    }
                )
            )

        assert isinstance(interface, SchedulerInterface)
        assert interface.user_id == 1
        assert interface.storage_id == 1
        assert interface.chat_id == 555
        assert interface.thread_id == 42

    @pytest.mark.parametrize("client", ["tui", "vscode", "pycharm", "neovim"])
    async def test_stdio_clients_get_stdio_interface(self, client: str) -> None:
        """Every non-telegram client (stdio/IDE) gets StdioSchedulerInterface."""
        settings = SimpleNamespace(client=client)
        with patch("chibi.services.jobs.agent_task.application_settings", settings):
            interface = _build_interface(
                AgentJobPayload.model_validate({"job_id": "agent:-10000000000000000:job", **STDIO_PAYLOAD_KWARGS})
            )

        assert isinstance(interface, StdioSchedulerInterface)
        assert interface.user_id == IDE_STORAGE_ID
        assert interface.storage_id == IDE_STORAGE_ID
        assert interface.thread_id == 42


class TestStdioSchedulerInterface:
    """Delivery behavior of the stdio scheduler interface."""

    async def test_send_message_uses_registered_emitter(self) -> None:
        """send_message hands (thread_id, content) to the session emitter."""
        delivered: list[tuple[int, str]] = []

        async def emitter(thread_id: int, content: str) -> None:
            delivered.append((thread_id, content))

        set_stdio_delivery_emitter(emitter)
        interface = StdioSchedulerInterface(
            user_id=IDE_STORAGE_ID, storage_id=IDE_STORAGE_ID, chat_id=IDE_STORAGE_ID, thread_id=42
        )

        await interface.send_message(message="Job answer")

        assert delivered == [(42, "Job answer")]

    async def test_send_message_without_session_drops_quietly(self) -> None:
        """With no stdio session connected the message is dropped, not raised."""
        interface = StdioSchedulerInterface(
            user_id=IDE_STORAGE_ID, storage_id=IDE_STORAGE_ID, chat_id=IDE_STORAGE_ID, thread_id=42
        )

        await interface.send_message(message="Nobody is listening")

        assert get_stdio_delivery_emitter() is None

    async def test_media_delivery_unsupported(self) -> None:
        """Media delivery mirrors the Telegram scheduler interface: rejected."""
        interface = StdioSchedulerInterface(
            user_id=IDE_STORAGE_ID, storage_id=IDE_STORAGE_ID, chat_id=IDE_STORAGE_ID, thread_id=42
        )

        with pytest.raises(NotImplementedError):
            await interface.send_document(document=b"doc")
        with pytest.raises(NotImplementedError):
            await interface.send_images(images=["a.png"])
        with pytest.raises(NotImplementedError):
            await interface.send_audio(audio=b"audio")
        with pytest.raises(NotImplementedError):
            await interface.send_video(video=b"video")


class TestStdioNotifyFailure:
    """_notify_failure works in stdio without any Telegram configuration."""

    async def test_failure_notification_reaches_session_emitter(self) -> None:
        """The failure note is delivered as a message frame, no ConfigurationError."""
        delivered: list[tuple[int, str]] = []

        async def emitter(thread_id: int, content: str) -> None:
            delivered.append((thread_id, content))

        set_stdio_delivery_emitter(emitter)
        with (
            patch(
                "chibi.services.jobs.agent_task.application_settings",
                SimpleNamespace(client="vscode", scheduler_failure_notify=True),
            ),
            patch(
                "chibi.services.scheduler_interface._get_scheduler_bot",
                side_effect=AssertionError("Telegram bot must not be touched in stdio delivery"),
            ),
        ):
            payload = AgentJobPayload.model_validate(
                {"job_id": "agent:-10000000000000000:broken", **STDIO_PAYLOAD_KWARGS}
            )
            await _notify_failure(payload, reason="RuntimeError: boom")

        assert len(delivered) == 1
        thread_id, message = delivered[0]
        assert thread_id == 42
        assert "agent:-10000000000000000:broken" in message
        assert "RuntimeError: boom" in message

    async def test_failure_notification_without_session_drops_quietly(self) -> None:
        """A failed job in a stdio process without a connected session never raises."""
        with patch(
            "chibi.services.jobs.agent_task.application_settings",
            SimpleNamespace(client="vscode", scheduler_failure_notify=True),
        ):
            payload = AgentJobPayload.model_validate(
                {"job_id": "agent:-10000000000000000:broken", **STDIO_PAYLOAD_KWARGS}
            )
            await _notify_failure(payload, reason="RuntimeError: boom")


class TestUploadedFileStorageFlag:
    """Scheduler interfaces must opt out of uploaded-file storage lookups."""

    async def test_scheduler_interface_disables_uploaded_file_storage(self) -> None:
        """Self-wake turns have no update context, so get_file_storage must not be consulted."""
        interface = SchedulerInterface(user_id=1, storage_id=1, chat_id=1)
        assert interface.uses_uploaded_file_storage is False

    async def test_stdio_scheduler_interface_disables_uploaded_file_storage(self) -> None:
        """The stdio counterpart likewise declares no uploaded-files storage."""
        interface = StdioSchedulerInterface(user_id=1, storage_id=1, chat_id=1)
        assert interface.uses_uploaded_file_storage is False
