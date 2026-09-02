"""JSONL stdio transport for IDE clients."""

import asyncio
import inspect
import json
import sys
import warnings
from io import BytesIO
from typing import Any, Callable

from loguru import logger
from openai.types import CompletionUsage

from chibi.config.gpt import gpt_settings
from chibi.constants import IDE_STORAGE_ID
from chibi.exceptions import ConfigurationError, StorageError
from chibi.models import Message, get_model_context_window
from chibi.schemas.app import UsageSchema
from chibi.services.bot import handle_image_generation, handle_reset, handle_user_prompt
from chibi.services.interface import EditorContextProvider, UserInterface
from chibi.services.task_manager import task_manager
from chibi.services.user import (
    clone_thread_messages,
    get_info,
    get_models_available,
    save_thread_name,
    set_active_model,
)
from chibi.storage.abstract import Database
from chibi.storage.database import inject_database

PROTOCOL_VERSION = 1
COMMANDS = ["/reset", "/new_thread_with_current_context", "/model", "/imagine", "/info", "/help", "/quit", "/exit"]


def build_usage_payload(
    usage: UsageSchema | CompletionUsage | None, provider: str | None, model: str | None
) -> dict[str, Any] | None:
    """Shape provider usage data into the result-frame ``usage`` object.

    The numbers are exactly what the provider reported for the request that
    produced this answer, never an estimate. Input tokens include Anthropic
    cache reads and writes when they are reported separately, mirroring how
    ``UsageCacheStore`` computes the real prompt size. The emitted
    ``context_window`` is the effective ceiling ``min(model window,
    MAX_HISTORY_TOKENS)``: proactive summarization fires at
    ``MAX_HISTORY_TOKENS`` (default 100000), so the denominator the client
    sees must reflect that ceiling rather than the raw model window. When the
    backend has no curated context window for the model, ``context_window``
    is null and stays null (no clamp possible, nothing fabricated).

    Args:
        usage: Usage object attached to the chat response, if the provider
            reported one.
        provider: Name of the provider that served the response.
        model: Name of the model that served the response.

    Returns:
        A dict with ``input_tokens``, ``output_tokens`` and ``context_window``,
        or None when no usage data is available.
    """
    if usage is None:
        return None
    input_tokens = usage.prompt_tokens
    cache_creation = getattr(usage, "cache_creation_input_tokens", None) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    # Anthropic-compatible APIs (Anthropic itself and MiniMax, which rides the
    # same Anthropic Messages path) report cached input outside input_tokens,
    # so the cache parts must be added. Every other provider includes cached
    # tokens inside its prompt token count already.
    if provider and provider.lower() in ("anthropic", "minimax"):
        input_tokens += cache_creation + cache_read
    context_window = get_model_context_window(model)
    if context_window is not None:
        # Effective ceiling: summarization triggers at MAX_HISTORY_TOKENS, so
        # cap the advertised window there. Unknown models keep null.
        context_window = min(context_window, gpt_settings.max_history_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": usage.completion_tokens,
        "context_window": context_window,
    }


# Maps backend-internal error codes to the closest frontend-facing code.
# Codes without a clean frontend analog (malformed_request, not_initialized,
# unsupported_protocol_version, unknown_request, unknown_message, cancelled,
# request_failed) are kept as backend-internal codes and emitted verbatim.
_FRONTEND_CODE_MAP: dict[str, str] = {
    # Command / validation failures that map cleanly to the frontend contract.
    "invalid_argument": "invalid_request",
    "missing_model": "invalid_request",
    "missing_provider": "invalid_request",
    # Provider-side failures surfaced by the shared exception handler.
    "provider_error": "backend_error",
    "provider_configuration": "backend_error",
    "provider_authorization": "backend_error",
    "runtime_error": "backend_error",
    "provider_rate_limit": "rate_limited",
    # Transport-level codes with no clean frontend analog remain backend-internal.
    "malformed_request": "malformed_request",
    "unsupported_protocol_version": "unsupported_protocol_version",
    "not_initialized": "not_initialized",
    "unknown_request": "unknown_request",
    "unknown_message": "unknown_message",
    "request_failed": "request_failed",
    "cancelled": "cancelled",
    # Canonical frontend code used for future rate limiters.
    "rate_limited": "rate_limited",
}


def _frontend_code(code: str) -> str:
    """Translate a backend-internal error code to its frontend-facing form."""
    return _FRONTEND_CODE_MAP.get(code, code)


@inject_database
async def _get_thread_messages(db: Database, storage_id: int, thread_id: int) -> list[Message]:
    """Return the conversation history stored for one thread.

    Args:
        db: Database instance injected by :func:`chibi.storage.database.inject_database`.
        storage_id: The storage identity owning the thread.
        thread_id: The thread whose messages should be returned.

    Returns:
        The stored conversation messages; empty for unknown or empty threads.
    """
    user = await db.get_or_create_user(user_id=storage_id)
    return await db.get_conversation_messages(user=user, thread_id=thread_id)


class IDEInterface(UserInterface, EditorContextProvider):
    """User-interface adapter that collects Chibi responses for one IDE request."""

    uses_uploaded_file_storage = False

    def __init__(
        self,
        thread_id: int,
        prompt: str,
        context: dict[str, Any],
        emit: Callable[[str], Any],
        background_emit: Callable[[int, str, str | None, str | None], Any] | None = None,
    ) -> None:
        """Initialize an IDE request interface.

        Args:
            thread_id: Existing IDE chat thread identifier.
            prompt: User prompt for this request.
            context: Editor context supplied by the client.
            emit: Callback receiving assistant text responses.
            background_emit: Optional session-level callback delivering
                assistant text as out-of-band background message frames
                after the owning request has finished.
        """
        self._thread_id = thread_id
        self._prompt = prompt
        self._context = context
        self._emit = emit
        self._background_emit = background_emit
        self._closed = False
        self.response_model: str | None = None
        self.response_provider: str | None = None
        self.response_usage: UsageSchema | CompletionUsage | None = None
        self.error_code: str | None = None
        self.error_message: str | None = None

    def mark_closed(self) -> None:
        """Flag the owning request as finished.

        After this, assistant text delivered by background tool tasks is
        routed to the out-of-band background emitter instead of the dead
        request's response buffer.
        """
        self._closed = True

    @property
    def editor_context(self) -> dict[str, Any] | None:
        """Return editor context supplied with this IDE request."""
        return self._context

    @property
    def context(self) -> dict[str, Any] | None:
        """Return editor context supplied with this IDE request.

        .. deprecated::
            Use :attr:`editor_context` instead. This property remains for
            backward compatibility and will be removed in a future release.
        """
        warnings.warn(
            "IDEInterface.context is deprecated; use IDEInterface.editor_context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.editor_context

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

    async def send_tool_answer(self, content: str, model: str | None = None, provider: str | None = None) -> None:
        """Deliver an answer produced by a background tool task.

        While the owning request is still open, the text is appended to the
        request's response buffer exactly as before. Once the request has
        finished, the text is routed to the session-level background emitter
        so it reaches the client as a ``message`` frame instead of being
        silently lost.

        Args:
            content: Assistant text being delivered.
            model: Model that produced the answer, when known.
            provider: Provider that produced the answer, when known.
        """
        if self._closed:
            if self._background_emit is None:
                logger.debug(
                    "Dropping background tool answer for thread {}: client did not declare background_messages",
                    self._thread_id,
                )
                return
            result = self._background_emit(self._thread_id, content, model, provider)
            if inspect.isawaitable(result):
                await result
            return
        result = self._emit(content)
        if inspect.isawaitable(result):
            await result

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
        """Persist the thread name in backend storage.

        IDE threads have no external topic to edit, so the name is registered
        in the user's thread-name map (the same place the clone command uses).

        Args:
            new_name: The new display name for this request's thread.

        Returns:
            True once the name is persisted.
        """
        await save_thread_name(storage_id=self.storage_id, thread_id=self.thread_id, name=new_name)
        return True

    async def delete_thread(self) -> bool:
        """Reject deletion of IDE threads."""
        raise NotImplementedError("IDE threads are allocated by the client")


class IDEStdioRunner:
    """Run the versioned Chibi IDE JSONL protocol over stdin and stdout."""

    def __init__(self) -> None:
        """Initialize the transport state."""
        self._initialized = False
        self.client_name: str | None = None
        self.client_version: str | None = None
        self._stopping = False
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._thread_requests: dict[int, int] = {}
        self._active_clones: set[int] = set()
        self._background_messages_enabled = False
        self.exit_code = 0
        self._stdout_lock = asyncio.Lock()

    async def _write(self, message: dict[str, Any]) -> None:
        """Write one protocol frame to stdout as a UTF-8 encoded JSONL line.

        The frame is written to the binary stdout buffer so the wire protocol
        stays strictly UTF-8 regardless of the console locale or code page
        (a cp1252 console on Windows cannot encode Cyrillic text otherwise).
        Streams without a binary buffer fall back to a plain text write.

        Args:
            message: JSON-compatible protocol message.
        """
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._stdout_lock:
            buffer = getattr(sys.stdout, "buffer", None)
            if buffer is None:
                sys.stdout.write(line)
                sys.stdout.flush()
                return
            buffer.write(line.encode("utf-8"))
            buffer.flush()

    async def _error(self, code: str, message: str, request_id: str | None = None, **extra: Any) -> None:
        """Write a correlated protocol error."""
        await self._write(
            {"type": "error", "request_id": request_id, "code": _frontend_code(code), "message": message, **extra}
        )

    async def _emit_background_message(
        self, thread_id: int, content: str, model: str | None, provider: str | None
    ) -> None:
        """Write one out-of-band background message frame to stdout.

        The frame shares the regular stdout write lock, so wire lines are
        never interleaved. On a write failure the message is logged and
        dropped: background delivery is best effort and there is no
        request id to correlate an error frame to.

        Args:
            thread_id: Thread the background task was spawned from.
            content: Assistant text being delivered.
            model: Model that produced the answer, if known.
            provider: Provider that produced the answer, if known.
        """
        frame: dict[str, Any] = {"type": "message", "thread_id": thread_id, "content": content}
        if model is not None:
            frame["model"] = model
        if provider is not None:
            frame["provider"] = provider
        try:
            await self._write(frame)
        except Exception:
            logger.exception("Failed to deliver a background message for thread {}", thread_id)

    async def emit_rate_limited(self, message: str, retry_after: int, request_id: str | None = None) -> None:
        """Emit a canonical rate-limited error frame with a retry hint.

        Args:
            message: Human-readable reason for the rate limit.
            retry_after: Seconds the client should wait before retrying.
            request_id: Correlated request id, if any.
        """
        await self._error("rate_limited", message, request_id, retry_after=retry_after)

    async def _read_line(self) -> str:
        """Read one stdin line without blocking the event loop."""
        return await asyncio.to_thread(sys.stdin.readline)

    @staticmethod
    def _valid_request(message: dict[str, Any]) -> bool:
        """Validate required request fields and their basic types."""
        selection = message.get("selection")
        if selection is not None:
            if not isinstance(selection, dict):
                return False
            if "start" in selection or "end" in selection:
                return False
            if not isinstance(selection.get("start_line"), int) or selection["start_line"] < 0:
                return False
            if not isinstance(selection.get("end_line"), int) or selection["end_line"] < 0:
                return False
            text = selection.get("text")
            if text is not None and not isinstance(text, str):
                return False
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

    async def _handle_new_thread_with_current_context(
        self, interface: IDEInterface, args: str, responses: list[str]
    ) -> None:
        """Clone a source thread's history into the request's (new) thread.

        The request frame arrives on the destination thread's identity (the
        client minted the new thread id, mirroring how /reset arrives on the
        thread it resets); the source thread id is taken from the command
        arguments: ``/new_thread_with_current_context <source_thread_id> [name]``.
        The clone re-keys the source conversation onto the destination bucket
        via :func:`chibi.services.user.clone_thread_messages`, which also
        registers the thread name.

        Args:
            interface: Request-local interface bound to the destination thread.
            args: Raw command arguments (source thread id and optional name).
            responses: Accumulated response lines for the result frame.
        """
        parts = args.split(maxsplit=1)
        raw_source = parts[0] if parts else ""
        name = parts[1].strip() if len(parts) > 1 else ""

        source = int(raw_source) if raw_source.isascii() and raw_source.isdigit() else None
        if source is None or source < 0:
            interface.error_code = "invalid_argument"
            interface.error_message = (
                f"Invalid source thread id: {raw_source!r}. "
                "Usage: /new_thread_with_current_context <source_thread_id> [name]"
            )
            return
        if self._thread_requests.get(source, 0) > 0:
            interface.error_code = "invalid_argument"
            interface.error_message = f"Source thread {source} is busy. Wait for it to finish before cloning."
            return
        if source in self._active_clones:
            interface.error_code = "invalid_argument"
            interface.error_message = f"A clone of thread {source} is already in progress."
            return

        self._active_clones.add(source)
        try:
            try:
                source_history = await _get_thread_messages(storage_id=IDE_STORAGE_ID, thread_id=source)
                if not source_history:
                    interface.error_code = "invalid_argument"
                    interface.error_message = f"Source thread {source} has no message history to clone."
                    return
                destination_history = await _get_thread_messages(
                    storage_id=IDE_STORAGE_ID, thread_id=interface.thread_id
                )
                if destination_history:
                    interface.error_code = "invalid_argument"
                    interface.error_message = (
                        f"Destination thread {interface.thread_id} already has message history. "
                        "Clone into an empty thread instead."
                    )
                    return
                cloned_messages = await clone_thread_messages(
                    storage_id=IDE_STORAGE_ID,
                    old_thread_id=source,
                    new_thread_id=interface.thread_id,
                    name=name or None,
                )
            except StorageError as exc:
                logger.exception("IDE thread clone failed")
                interface.error_code = "request_failed"
                interface.error_message = f"Failed to clone thread {source}: {exc.detail}"
                return
            except Exception:
                logger.exception("IDE thread clone failed")
                interface.error_code = "request_failed"
                interface.error_message = (
                    f"Failed to clone thread {source}. Check the Chibi output channel for details."
                )
                return
        finally:
            self._active_clones.discard(source)

        responses.append(
            f"✅ Thread cloned: {name or str(interface.thread_id)} (ID: {interface.thread_id}). "
            f"{cloned_messages} messages copied."
        )

    async def _run_request(self, message: dict[str, Any]) -> None:
        """Route and complete one request."""
        request_id = message["request_id"]
        thread_id = message["thread_id"]
        prompt = message["prompt"].strip()
        responses: list[str] = []
        background_emit = self._emit_background_message if self._background_messages_enabled else None
        interface = IDEInterface(thread_id, prompt, message, responses.append, background_emit=background_emit)
        await self._write({"type": "status", "request_id": request_id, "state": "running"})
        try:
            if prompt.startswith("/"):
                parts = prompt.split(maxsplit=1)
                command, args = parts[0].lower(), parts[1] if len(parts) > 1 else ""
                if command == "/help":
                    responses.append("Available commands: " + ", ".join(COMMANDS))
                elif command == "/reset":
                    await handle_reset(interface=interface)
                elif command == "/new_thread_with_current_context":
                    await self._handle_new_thread_with_current_context(
                        interface=interface, args=args, responses=responses
                    )
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
            usage_payload = build_usage_payload(
                usage=interface.response_usage, provider=interface.response_provider, model=interface.response_model
            )
            if usage_payload is not None:
                result["usage"] = usage_payload
            await self._write(result)
        except asyncio.CancelledError:
            await self._error("cancelled", "Request cancelled.", request_id)
            raise
        except StorageError as exc:
            logger.exception("IDE request failed")
            await self._error(
                "request_failed",
                exc.detail,
                request_id,
                cause="StorageError",
            )
        except ConfigurationError as exc:
            logger.exception("IDE request failed")
            await self._error(
                "request_failed",
                exc.detail,
                request_id,
                cause="ConfigurationError",
            )
        except Exception as exc:
            logger.exception("IDE request failed")
            await self._error(
                "request_failed",
                "The backend could not complete this request. Check the Chibi output channel for details, "
                "then verify storage and provider configuration.",
                request_id,
                cause=type(exc).__name__,
            )
        finally:
            interface.mark_closed()
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
                raw_client = message.get("client")
                client = raw_client if isinstance(raw_client, dict) else {}
                raw_name = client.get("name")
                raw_version = client.get("version")
                self.client_name = raw_name if isinstance(raw_name, str) else None
                self.client_version = raw_version if isinstance(raw_version, str) else None
                raw_capabilities = message.get("capabilities")
                capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
                self._background_messages_enabled = capabilities.get("background_messages") is True
                if self._background_messages_enabled:
                    logger.debug("client declared the background_messages capability")
                logger.info(
                    "client_handshake name={} version={} protocol_version={}",
                    self.client_name or "<unknown>",
                    self.client_version or "<unknown>",
                    message["protocol_version"],
                )
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
            elif message["prompt"].strip().split(maxsplit=1)[0].lower() in ("/quit", "/exit"):
                await self._write({"type": "status", "request_id": request_id, "state": "running"})
                await self._write({"type": "result", "request_id": request_id, "content": "Bye!"})
                self._stopping = True
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
        from chibi.config.logging import use_stderr_logging

        # stdout is the JSONL protocol channel; loguru must never write there.
        use_stderr_logging()
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
