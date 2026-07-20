"""JSONL stdio transport for IDE clients."""

import asyncio
import inspect
import json
import sys
from io import BytesIO
from typing import Any, Callable

from loguru import logger

from chibi.constants import IDE_STORAGE_ID
from chibi.services.bot import handle_image_generation, handle_reset, handle_user_prompt
from chibi.services.interface import UserInterface
from chibi.services.task_manager import task_manager
from chibi.services.user import get_info, get_models_available, set_active_model

PROTOCOL_VERSION = 1
COMMANDS = ["/reset", "/model", "/imagine", "/info", "/help", "/quit", "/exit"]


class IDEInterface(UserInterface):
    """User-interface adapter that collects Chibi responses for one IDE request."""

    uses_uploaded_file_storage = False

    def __init__(self, thread_id: int, prompt: str, context: dict[str, Any], emit: Callable[[str], Any]) -> None:
        """Initialize an IDE request interface.

        Args:
            thread_id: Existing IDE chat thread identifier.
            prompt: User prompt for this request.
            context: Editor context supplied by the client.
            emit: Callback receiving assistant text responses.
        """
        self._thread_id = thread_id
        self._prompt = prompt
        self.context = context
        self._emit = emit
        self.response_model: str | None = None
        self.response_provider: str | None = None
        self.error_code: str | None = None
        self.error_message: str | None = None

    @property
    def chat_id(self) -> int:
        """Return the stable IDE chat identifier."""
        return IDE_STORAGE_ID

    @property
    def user_id(self) -> int:
        """Return the stable IDE storage identity."""
        return IDE_STORAGE_ID

    @property
    def storage_id(self) -> int:
        """Return the stable IDE storage identity."""
        return IDE_STORAGE_ID

    @property
    def thread_id(self) -> int:
        """Return the request's chat thread identifier."""
        return self._thread_id

    @property
    def user_data(self) -> str:
        """Return a diagnostic user description."""
        return "IDE user"

    @property
    def chat_data(self) -> str:
        """Return a diagnostic chat description."""
        return f"IDE chat, thread #{self.thread_id}"

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Return no document attachment for JSONL requests."""
        return None

    @property
    def attached_document_caption(self) -> str | None:
        """Return no attachment caption for JSONL requests."""
        return None

    async def get_text_prompt(self) -> str | None:
        """Return the request prompt."""
        return self._prompt

    async def get_voice_prompt(self) -> BytesIO | None:
        """Return no voice attachment for JSONL requests."""
        return None

    async def get_caption(self) -> str | None:
        """Return no attachment caption."""
        return None

    def set_caption(self, caption: str) -> None:
        """Ignore captions, which are not part of the IDE transport."""

    async def send_action_typing(self) -> None:
        """Ignore typing indicators on stdio."""

    async def send_action_uploading_photo(self) -> None:
        """Ignore upload indicators on stdio."""

    async def send_action_recording(self) -> None:
        """Ignore recording indicators on stdio."""

    async def send_reaction(self, reaction: str) -> None:
        """Ignore reactions on stdio."""

    async def delete_last_user_message(self) -> None:
        """Ignore message deletion on stdio."""

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Emit assistant text through the request-local callback."""
        result = self._emit(message)
        if inspect.isawaitable(result):
            await result

    async def send_audio(
        self,
        audio: bytes | str,
        reply: bool = True,
        title: str | None = None,
        caption: str | None = None,
        performer: str | None = None,
        duration: int | None = None,
        thumbnail: bytes | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Ignore binary audio delivery on the text protocol."""

    async def send_video(
        self,
        video: bytes | str,
        reply: bool = True,
        title: str | None = None,
        caption: str | None = None,
        duration: int | None = None,
        thumbnail: bytes | None = None,
        filename: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Ignore binary video delivery on the text protocol."""

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Emit image references as text on the JSONL protocol."""
        result = self._emit("\n".join(str(image) for image in images))
        if inspect.isawaitable(result):
            await result

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Ignore binary document delivery on the text protocol."""

    async def create_thread(self, name: str) -> int:
        """Reject creation of threads, which are allocated by the IDE."""
        raise NotImplementedError("IDE threads are allocated by the client")

    async def rename_thread(self, new_name: str) -> bool:
        """Reject Telegram-only thread renaming."""
        raise NotImplementedError("IDE threads cannot be renamed")

    async def delete_thread(self) -> bool:
        """Reject deletion of IDE threads."""
        raise NotImplementedError("IDE threads are allocated by the client")


class IDEStdioRunner:
    """Run the versioned Chibi IDE JSONL protocol over stdin and stdout."""

    def __init__(self) -> None:
        """Initialize the transport state."""
        self._initialized = False
        self._stopping = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._thread_requests: dict[int, int] = {}
        self.exit_code = 0
        self._stdout_lock = asyncio.Lock()

    async def _write(self, message: dict[str, Any]) -> None:
        """Write one protocol frame to stdout.

        Args:
            message: JSON-compatible protocol message.
        """
        async with self._stdout_lock:
            sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    async def _error(self, code: str, message: str, request_id: str | None = None, **extra: Any) -> None:
        """Write a correlated protocol error."""
        await self._write({"type": "error", "request_id": request_id, "code": code, "message": message, **extra})

    async def _read_line(self) -> str:
        """Read one stdin line without blocking the event loop."""
        return await asyncio.to_thread(sys.stdin.readline)

    @staticmethod
    def _valid_request(message: dict[str, Any]) -> bool:
        """Validate required request fields and their basic types."""
        return (
            isinstance(message.get("request_id"), str)
            and bool(message["request_id"])
            and isinstance(message.get("thread_id"), int)
            and message["thread_id"] >= 0
            and isinstance(message.get("prompt"), str)
            and bool(message["prompt"].strip())
            and isinstance(message.get("workspace_root"), str)
            and isinstance(message.get("active_file"), (str, type(None)))
            and isinstance(message.get("selection"), (dict, type(None)))
            and isinstance(message.get("cursor_position"), (dict, type(None)))
            and isinstance(message.get("language_id"), (str, type(None)))
        )

    async def _run_request(self, message: dict[str, Any]) -> None:
        """Route and complete one request."""
        request_id = message["request_id"]
        thread_id = message["thread_id"]
        prompt = message["prompt"].strip()
        responses: list[str] = []
        interface = IDEInterface(thread_id, prompt, message, responses.append)
        await self._write({"type": "status", "request_id": request_id, "state": "running"})
        try:
            if prompt.startswith("/"):
                parts = prompt.split(maxsplit=1)
                command, args = parts[0].lower(), parts[1] if len(parts) > 1 else ""
                if command in ("/quit", "/exit"):
                    self._stopping = True
                elif command == "/help":
                    responses.append("Available commands: " + ", ".join(COMMANDS))
                elif command == "/reset":
                    await handle_reset(interface=interface)
                elif command == "/imagine":
                    if not args:
                        raise ValueError("Please provide a prompt: /imagine <description>")
                    await handle_image_generation(prompt=args, interface=interface)
                elif command == "/info":
                    models = await get_models_available(user_id=IDE_STORAGE_ID, thread_id=thread_id)
                    active = next((model.display_name for model in models if "🟢" in model.display_name), "N/A")
                    user_info = await get_info(user_id=IDE_STORAGE_ID)
                    responses.append(f"User ID: {IDE_STORAGE_ID}\nActive model: {active}\nUser info: {user_info}")
                elif command == "/model":
                    models = await get_models_available(user_id=IDE_STORAGE_ID, thread_id=thread_id)
                    if args.isdigit() and 1 <= int(args) <= len(models):
                        await set_active_model(interface=interface, model=models[int(args) - 1])
                        responses.append(
                            f"Selected model: {models[int(args) - 1].display_name} ({models[int(args) - 1].provider})"
                        )
                    elif args:
                        selected = next(
                            (model for model in models if model.name == args or model.display_name == args), None
                        )
                        if selected is None:
                            available_models = ", ".join(model.display_name for model in models) or "none"
                            interface.error_code = "invalid_argument"
                            interface.error_message = f"Unknown model selection. Available models: {available_models}."
                        else:
                            await set_active_model(interface=interface, model=selected)
                            responses.append(f"Selected model: {selected.display_name} ({selected.provider})")
                    else:
                        responses.append(
                            "\n".join(
                                f"{index}. {model.display_name} ({model.provider})"
                                for index, model in enumerate(models, 1)
                            )
                            or "No models available."
                        )
                else:
                    raise ValueError(f"Unknown command: {command}")
            else:
                await handle_user_prompt(interface=interface)
            content = "\n".join(responses)
            if interface.error_code is not None:
                await self._error(interface.error_code, interface.error_message or "Request failed.", request_id)
                return
            result: dict[str, Any] = {"type": "result", "request_id": request_id, "content": content}
            if interface.response_model is not None and interface.response_provider is not None:
                result["model"] = interface.response_model
                result["provider"] = interface.response_provider
            await self._write(result)
        except asyncio.CancelledError:
            await self._error("cancelled", "Request cancelled.", request_id)
            raise
        except Exception:
            logger.exception("IDE request failed")
            await self._error(
                "request_failed",
                "The backend could not complete this request. Check the Chibi output channel for details, "
                "then verify storage and provider configuration.",
                request_id,
            )
        finally:
            self._tasks.pop(request_id, None)
            remaining = self._thread_requests.get(thread_id, 1) - 1
            if remaining <= 0:
                self._thread_requests.pop(thread_id, None)
            else:
                self._thread_requests[thread_id] = remaining

    async def _handle_message(self, message: Any) -> None:
        """Validate and dispatch one decoded protocol message."""
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            await self._error("malformed_request", "Message must be a JSON object with a type.")
            return
        message_type = message["type"]
        request_id = message.get("request_id") if isinstance(message.get("request_id"), str) else None
        if message_type == "initialize":
            if self._initialized or not isinstance(message.get("protocol_version"), int):
                await self._error("malformed_request", "Invalid initialize message.")
            elif message["protocol_version"] != PROTOCOL_VERSION:
                await self._error(
                    "unsupported_protocol_version",
                    f"Unsupported protocol version: {message['protocol_version']}.",
                    None,
                    server_protocol_version=PROTOCOL_VERSION,
                )
                self._stopping = True
                self.exit_code = 1
            else:
                self._initialized = True
                await self._write(
                    {
                        "type": "ready",
                        "protocol_version": PROTOCOL_VERSION,
                        "server": {"name": "chibi", "version": "1.0.0"},
                        "capabilities": {"commands": COMMANDS},
                    }
                )
        elif message_type == "request":
            if not self._initialized:
                await self._error("not_initialized", "Not initialized.", request_id)
            elif not self._valid_request(message):
                await self._error("malformed_request", "Missing required request fields.", request_id)
            elif request_id in self._tasks:
                await self._error("malformed_request", "request_id is already in use.", request_id)
            else:
                assert request_id is not None
                thread_id = message["thread_id"]
                contended = self._thread_requests.get(thread_id, 0) > 0
                self._thread_requests[thread_id] = self._thread_requests.get(thread_id, 0) + 1
                if contended:
                    await self._write({"type": "status", "request_id": request_id, "state": "queued"})
                task = task_manager.run_task(
                    coro=self._run_request(message), user_id=IDE_STORAGE_ID, thread_id=thread_id
                )
                if task is None:
                    self._thread_requests[thread_id] -= 1
                    await self._error("request_failed", "Transport is shutting down.", request_id)
                    return
                self._tasks[request_id] = task
        elif message_type == "cancel":
            if not self._initialized:
                await self._error("not_initialized", "Not initialized.", request_id)
            elif request_id is None or request_id not in self._tasks:
                await self._error("unknown_request", "Unknown request id.", request_id)
            else:
                self._tasks[request_id].cancel()
        elif message_type == "shutdown":
            self._stopping = True
        else:
            await self._error("unknown_message", f"Unknown message type: {message_type}.", request_id)

    async def run(self) -> int:
        """Run until shutdown or stdin EOF, then clean up in-flight work."""
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        try:
            while not self._stopping:
                line = await self._read_line()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    await self._error("malformed_request", "Input is not valid JSON.")
                    continue
                await self._handle_message(decoded)
        finally:
            self._stopping = True
            await task_manager.shutdown()
        return self.exit_code
