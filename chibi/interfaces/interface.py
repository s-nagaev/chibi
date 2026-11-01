from abc import ABC
from io import BytesIO
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EditorContextProvider(Protocol):
    """Protocol for interfaces that provide IDE editor context."""

    @property
    def editor_context(self) -> dict[str, Any] | None:
        """Return editor context supplied by an IDE client, if available."""
        ...


class UserInterface(ABC):
    captures_llm_thoughts: bool = False
    delta_emitted: bool = False
    streaming_enabled: bool = False
    non_streaming_fallback_error: tuple[str, BaseException] | None = None

    @property
    def chat_id(self) -> str | int:
        """Returns the unique identifier for the current chat.

        Returns:
            The chat identifier.
        """
        raise NotImplementedError

    @property
    def user_id(self) -> int:
        """Returns the unique identifier for the current user.

        Returns:
            The user identifier.
        """
        raise NotImplementedError

    @property
    def storage_id(self) -> int:
        """Returns the storage key for persisting conversation/context.

        Private chats -> user_id (== chat_id). Group chats -> chat_id (shared).

        Returns:
            The storage identifier.
        """
        raise NotImplementedError

    @property
    def thread_id(self) -> int:
        """Returns the message thread ID for the current chat context.

        Returns:
            The thread identifier, or 0 for non-threaded chats.
        """
        raise NotImplementedError

    @property
    def user_data(self) -> str:
        """Returns a string representation of the user data.

        Returns:
            The user data string.
        """
        raise NotImplementedError

    @property
    def chat_data(self) -> str:
        """Returns a string representation of the chat data.

        Returns:
            The chat data string.
        """
        raise NotImplementedError

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Returns the attached document data if present, otherwise None.

        Returns:
            The document data dictionary or None.
        """
        raise NotImplementedError

    @property
    def attached_document_caption(self) -> str | None:
        """Returns the caption of the attached document if present, otherwise None.

        Returns:
            The caption string or None.
        """
        raise NotImplementedError

    async def get_text_prompt(self) -> str | None:
        """Retrieves the text prompt from the current message.

        Returns:
            The text prompt string or None.
        """
        raise NotImplementedError

    async def get_voice_prompt(self) -> BytesIO | None:
        """Retrieves the voice prompt as a BytesIO object if present, otherwise None.

        Returns:
            The voice prompt BytesIO object or None.
        """
        raise NotImplementedError

    async def get_caption(self) -> str | None:
        """Retrieve the caption attached to the current message.

        Returns:
            The caption string, or None if no caption is present.
        """
        raise NotImplementedError

    def set_caption(self, caption: str) -> None:
        """Store a caption for use during subsequent media sends.

        Args:
            caption: The caption text to associate with the next media message.
        """
        raise NotImplementedError

    async def send_action_typing(self) -> None:
        """Sends a typing action to the user."""
        raise NotImplementedError

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the user."""
        raise NotImplementedError

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the user."""
        raise NotImplementedError

    async def send_reaction(self, reaction: str) -> None:
        """Sends a reaction to the user's message.

        Args:
            reaction: The reaction to send.
        """
        raise NotImplementedError

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user."""
        raise NotImplementedError

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Sends a text message to the user.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the message sending function.
        """
        raise NotImplementedError

    async def send_tool_answer(self, content: str, model: str | None = None, provider: str | None = None) -> None:
        """Delivers an answer produced by a background tool task.

        The default delivery is a regular message. Interfaces with an
        out-of-band background channel may override this to route the
        answer differently once the originating request has finished.

        Args:
            content: The assistant text to deliver.
            model: The model that produced the answer, when known.
            provider: The provider that produced the answer, when known.
        """
        await self.send_message(message=content)

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
        """Sends an audio file to the user.

        Args:
            audio: The audio data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the audio.
            caption: The caption for the audio.
            performer: The performer of the audio.
            duration: The duration of the audio in seconds.
            thumbnail: The thumbnail data for the audio.
            filename: The filename for the audio.
            **kwargs: Additional arguments for the audio sending function.
        """
        raise NotImplementedError

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
        """Sends a video file to the user.

        Args:
            video: The video data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video in seconds.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments for the video sending function.
        """
        raise NotImplementedError

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments for the image sending function.
        """
        raise NotImplementedError

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Sends a document file to the user.

        Args:
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments for the document sending function.
        """
        raise NotImplementedError

    async def create_thread(self, name: str) -> int:
        """Create a new thread/topic.

        Args:
            name: The name of the thread to create.

        Returns:
            The ID of the created thread.
        """
        raise NotImplementedError

    async def rename_thread(self, new_name: str) -> bool:
        """Rename the current thread/topic.

        Args:
            new_name: The new name for the thread.

        Returns:
            True if the thread was successfully renamed, False otherwise.
        """
        raise NotImplementedError

    async def delete_thread(self) -> bool:
        """Delete the current thread/topic.

        Returns:
            True if the thread was successfully deleted, False otherwise.
        """
        raise NotImplementedError

    async def send_delta(self, text: str) -> None:
        """Send a live partial response chunk (delta) to the user.

        Default implementation is a no-op: runners that do not support
        streaming (Telegram, terminal) inherit it unchanged and discard
        deltas. IDE/transport interfaces override this method to forward
        the chunk over the wire.

        Args:
            text: The partial text chunk to send.
        """
        return None

    async def send_llm_thoughts(self, thoughts: str) -> None:
        """Send LLM thinking/thoughts to the user.

        Default implementation sends as plain text.
        Telegram overrides with native <tg-thinking> animation.

        Args:
            thoughts: The LLM reasoning text to display.
        """
        await self.send_message(f"💡💭 {thoughts}", reply=False)
        return None

    async def absorb_hidden_turn_reasoning(self) -> None:
        """Drop reasoning captured for a turn that ended hidden.

        Called by the render-time suppression gate (task 5) after a turn
        whose messages are hidden delivered nothing: interfaces that buffer
        reasoning for later delivery must make sure the hidden turn's private
        CoT can never ride a later frame. Default implementation is a no-op;
        the IDE transport interface overrides it to re-baseline its
        continuation snapshot.
        """
        return None
