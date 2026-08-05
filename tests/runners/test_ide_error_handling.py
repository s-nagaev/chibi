"""Regression tests for actionable IDE error handling."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import chibi.config  # noqa: F401
from chibi.exceptions import ConfigurationError, NoApiKeyProvidedError, StorageError
from chibi.runners.ide_transport import IDEInterface, IDEStdioRunner
from chibi.storage.local import LocalStorage
from chibi.utils.app import handle_gpt_exceptions


class OutputRecorder:
    """Capture emitted JSONL protocol frames."""

    def __init__(self) -> None:
        """Initialize an empty frame list."""
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, frame: dict[str, Any]) -> None:
        """Store one emitted frame.

        Args:
            frame: Protocol frame to capture.
        """
        self.frames.append(frame)


def request(request_id: str, prompt: str = "hello") -> dict[str, Any]:
    """Build a valid IDE request frame.

    Args:
        request_id: Request correlation identifier.
        prompt: User prompt or slash command.

    Returns:
        A valid request frame.
    """
    return {
        "type": "request",
        "request_id": request_id,
        "thread_id": 1,
        "prompt": prompt,
        "workspace_root": "/tmp",
        "active_file": None,
        "selection": None,
        "cursor_position": None,
        "language_id": None,
    }


async def provider_configuration_failure(interface: IDEInterface) -> None:
    """Raise a provider configuration exception containing a secret-like detail.

    Args:
        interface: Interface passed to the decorated handler.
    """
    raise NoApiKeyProvidedError(detail="secret-token-must-not-reach-the-client")


@pytest.mark.asyncio
async def test_handled_provider_failure_is_sanitized_ide_error() -> None:
    """Handled provider exceptions become correlated IDE errors without raw details."""
    decorated_handler = handle_gpt_exceptions(provider_configuration_failure)
    recorder = OutputRecorder()
    runner = IDEStdioRunner()
    with (
        patch.object(runner, "_write", recorder),
        patch("chibi.runners.ide_transport.handle_user_prompt", decorated_handler),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request("provider-failure"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["request_id"] == "provider-failure"
    assert error["code"] == "backend_error"
    assert "API key" in error["message"]
    assert "secret-token-must-not-reach-the-client" not in error["message"]


@pytest.mark.asyncio
async def test_model_storage_failure_is_actionable_transport_error() -> None:
    """A /model storage failure stays distinct from a handled provider error."""
    recorder = OutputRecorder()
    runner = IDEStdioRunner()
    with (
        patch.object(runner, "_write", recorder),
        patch(
            "chibi.runners.ide_transport.get_models_available", new=AsyncMock(side_effect=OSError("disk unavailable"))
        ),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request("storage-failure", "/model"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["code"] == "request_failed"
    assert "output channel" in error["message"]
    assert "disk unavailable" not in error["message"]


def test_local_storage_creates_missing_host_directory(tmp_path: Path) -> None:
    """Local storage initializes a configured directory absent on a fresh host.

    Args:
        tmp_path: Per-test temporary directory supplied by pytest.
    """
    storage_path = tmp_path / "fresh-host" / "storage"

    LocalStorage(str(storage_path))

    assert storage_path.is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["99", "bogus"])
async def test_invalid_model_selection_is_actionable_and_lists_available_models(selection: str) -> None:
    """Invalid /model arguments use the existing typed IDE error path."""
    from chibi.schemas.app import ModelChangeSchema

    models = [
        ModelChangeSchema(provider="openai", name="gpt-example", display_name="GPT Example", image_generation=False),
    ]
    recorder = OutputRecorder()
    runner = IDEStdioRunner()
    with (
        patch.object(runner, "_write", recorder),
        patch("chibi.runners.ide_transport.get_models_available", new=AsyncMock(return_value=models)),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request(f"invalid-model-{selection}", f"/model {selection}"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["code"] == "invalid_request"
    assert error["code"] != "request_failed"
    assert "Unknown model selection" in error["message"]
    assert "GPT Example" in error["message"]


@pytest.mark.asyncio
async def test_rate_limited_helper_emits_canonical_frame() -> None:
    """The transport-level rate-limit helper emits a frontend-shaped error frame."""
    recorder = OutputRecorder()
    runner = IDEStdioRunner()
    with patch.object(runner, "_write", recorder):
        await runner.emit_rate_limited("Slow down", 5, request_id="rate-limit-1")

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["request_id"] == "rate-limit-1"
    assert error["code"] == "rate_limited"
    assert error["message"] == "Slow down"
    assert error["retry_after"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_file_storage", ["telegram", "local"])
async def test_ide_prompt_skips_uploaded_file_storage_for_all_storage_configurations(
    selected_file_storage: str,
) -> None:
    """IDE prompts do not resolve Telegram-only uploaded-file storage."""
    from types import SimpleNamespace

    from chibi.runners.ide_transport import IDEInterface
    from chibi.services.providers.utils import prepare_system_prompt

    interface = IDEInterface(1, "ping", {}, lambda _: None)
    user = SimpleNamespace(
        working_dir="/tmp",
        approximate_context_size=lambda thread_id: 0,
        id=1,
        info="",
        llm_skills={},
    )
    with (
        patch("chibi.services.providers.utils.get_chibi_user", new=AsyncMock(return_value=user)),
        patch("chibi.services.providers.utils.get_builtin_skill_names", return_value=[]),
        patch("chibi.services.providers.utils.get_file_storage") as get_file_storage,
        patch("chibi.services.providers.utils.application_settings.selected_file_storage", selected_file_storage),
    ):
        prompt = await prepare_system_prompt("base", 1, interface)

    assert "last_uploaded_files" not in prompt
    get_file_storage.assert_not_called()


@pytest.mark.asyncio
async def test_storage_error_emits_tailored_message_and_typed_cause() -> None:
    """StorageError in _run_request emits a tailored message and cause field."""
    recorder = OutputRecorder()
    runner = IDEStdioRunner()

    async def raise_storage_error(interface: IDEInterface) -> None:
        raise StorageError()

    with (
        patch.object(runner, "_write", recorder),
        patch("chibi.runners.ide_transport.handle_user_prompt", raise_storage_error),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request("storage-error"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["code"] == "request_failed"
    assert error["cause"] == "StorageError"
    assert "Storage is not configured" in error["message"]


@pytest.mark.asyncio
async def test_configuration_error_emits_tailored_message_and_typed_cause() -> None:
    """ConfigurationError in _run_request emits a tailored message and cause field."""
    recorder = OutputRecorder()
    runner = IDEStdioRunner()

    async def raise_config_error(interface: IDEInterface) -> None:
        raise ConfigurationError()

    with (
        patch.object(runner, "_write", recorder),
        patch("chibi.runners.ide_transport.handle_user_prompt", raise_config_error),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request("config-error"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["code"] == "request_failed"
    assert error["cause"] == "ConfigurationError"
    assert "not fully configured" in error["message"]


@pytest.mark.asyncio
async def test_generic_exception_emits_cause_with_exception_class_name() -> None:
    """Generic exceptions in _run_request emit cause with the exception class name."""
    recorder = OutputRecorder()
    runner = IDEStdioRunner()

    async def raise_runtime_error(interface: IDEInterface) -> None:
        raise RuntimeError("something went wrong")

    with (
        patch.object(runner, "_write", recorder),
        patch("chibi.runners.ide_transport.handle_user_prompt", raise_runtime_error),
    ):
        await runner._handle_message({"type": "initialize", "protocol_version": 1})
        await runner._handle_message(request("generic-error"))
        for _ in range(100):
            if any(frame.get("type") == "error" for frame in recorder.frames):
                break
            await asyncio.sleep(0.01)

    error = next(frame for frame in recorder.frames if frame.get("type") == "error")
    assert error["code"] == "request_failed"
    assert error["cause"] == "RuntimeError"
    assert "something went wrong" not in error["message"]
