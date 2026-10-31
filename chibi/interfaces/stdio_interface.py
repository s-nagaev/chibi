"""User-interface adapter for the JSONL stdio transport."""

import inspect
import warnings
from io import BytesIO
from typing import Any, Callable

from loguru import logger
from openai.types import CompletionUsage

from chibi.constants import IDE_STORAGE_ID
from chibi.interfaces.interface import EditorContextProvider, UserInterface
from chibi.schemas.app import UsageSchema
from chibi.services.user import save_thread_name


class StdioInterface(UserInterface, EditorContextProvider):
    """User-interface adapter that collects Chibi responses for one IDE request."""

    uses_uploaded_file_storage = False
    captures_llm_thoughts: bool = True

    def __init__(
        self,
        thread_id: int,
        prompt: str,
        context: dict[str, Any],
        emit: Callable[[str], Any],
        background_emit: Callable[[int, str, str | None, str | None, str | None], Any] | None = None,
        delta_emit: Callable[[str], Any] | None = None,
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
            delta_emit: Optional session-level callback delivering live
                response chunks as coalesced ``delta`` frames while the
                owning request is still running.
        """
        self._thread_id = thread_id
        self._prompt = prompt
        self._context = context
        self._emit = emit
        self._background_emit = background_emit
        self._delta_emit = delta_emit
        self._closed = False
        self.streaming_enabled = delta_emit is not None
        self.delta_emitted = False
        self.response_model: str | None = None
        self.response_provider: str | None = None
        self.response_usage: UsageSchema | CompletionUsage | None = None
        self.response_thoughts: str | None = None
        # Set by handle_user_prompt via setattr for every turn (task 5):
        # "hidden" when the turn ended via ack_silently() and its thoughts
        # are private CoT that must not travel on the wire.
        self.response_visibility: str = "visible"
        self._thoughts_at_close: str | None = None
        self.error_code: str | None = None
        self.error_message: str | None = None

    def mark_closed(self) -> None:
        """Flag the owning request as finished.

        After this, assistant text delivered by background tool tasks is
        routed to the out-of-band background emitter instead of the dead
        request's response buffer. The reasoning already attached to the
        request's result frame is snapshotted so a background tool task can
        later report ONLY the reasoning its own continuation turn added.
        """
        self._closed = True
        self._thoughts_at_close = self.response_thoughts

    def _continuation_thoughts(self) -> str | None:
        """Return the reasoning captured after the request closed.

        The background tool task's continuation LLM call appends its
        reasoning to the same request-local buffer via
        :meth:`send_llm_thoughts`; everything beyond the snapshot taken at
        :meth:`mark_closed` belongs to the continuation turn alone.

        Returns:
            The continuation turn's reasoning text, or None when nothing
            new was captured after the request closed.
        """
        captured = self.response_thoughts
        if not captured:
            return None
        baseline = self._thoughts_at_close
        if baseline and captured.startswith(baseline):
            delta = captured[len(baseline) :].lstrip("\n")
            return delta or None
        return captured

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
            "StdioInterface.context is deprecated; use StdioInterface.editor_context instead.",
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
        silently lost; the continuation turn's reasoning rides along so the
        client can render it with the answer.

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
            result = self._background_emit(self._thread_id, content, model, provider, self._continuation_thoughts())
            if inspect.isawaitable(result):
                await result
            return
        result = self._emit(content)
        if inspect.isawaitable(result):
            await result

    async def send_llm_thoughts(self, thoughts: str) -> None:
        """Capture LLM reasoning for the result frame instead of emitting it as text.

        Providers may report reasoning more than once during agentic tool
        loops; the pieces are accumulated in arrival order and joined with
        newlines. Captured text lives only on this request-local object and
        is never written to thread history.

        Args:
            thoughts: The LLM reasoning text to capture.
        """
        if not thoughts:
            return
        if self.response_thoughts is None:
            self.response_thoughts = thoughts
        else:
            self.response_thoughts = f"{self.response_thoughts}\n{thoughts}"

    async def absorb_hidden_turn_reasoning(self) -> None:
        """Re-baseline the continuation snapshot after a hidden turn (task 5).

        A background continuation turn that ended via ``ack_silently()``
        delivered nothing (the bot-level gate suppressed it), so its private
        reasoning was captured after the :meth:`mark_closed` snapshot and
        would otherwise ride the next background continuation's delta as if
        it belonged to that turn. Absorbing it into the baseline keeps the
        hidden turn's CoT off the wire without changing any frame shape.
        """
        self._thoughts_at_close = self.response_thoughts

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Emit assistant text through the request-local callback."""
        result = self._emit(message)
        if inspect.isawaitable(result):
            await result

    async def send_delta(self, text: str) -> None:
        """Forward a live response chunk to the transport's delta buffer.

        The chunk is handed to the session-level ``delta_emit`` callback,
        which appends it to the transport-owned coalescing buffer keyed by
        this request. Chunks arriving after the owning request has finished
        (or from an interface created for a non-streaming client) are
        dropped silently: a late delta must never surface as a ``delta``
        frame after the terminal one, and it must never be routed into the
        background emitter.

        Args:
            text: The partial text chunk to stream.
        """
        if self._closed or self._delta_emit is None or not text:
            return
        result = self._delta_emit(text)
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
