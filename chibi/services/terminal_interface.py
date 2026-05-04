"""Terminal interface implementation for Chibi AI assistant."""

from io import BytesIO
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from chibi.services.interface import UserInterface

# Module-level shared console for consistent output
_console = Console()


class TerminalInterface(UserInterface):
    """Terminal-based implementation of UserInterface for REPL usage.

    This interface provides a terminal-friendly way to interact with Chibi,
    supporting text input/output and basic commands.
    """

    def __init__(
        self,
        user_id: int,
        user_data_dict: dict | None = None,
        chat_data_dict: dict | None = None,
    ) -> None:
        """Initialize the terminal interface.

        Args:
            user_id: The unique identifier for the current user.
            user_data_dict: Optional dictionary for user data storage.
            chat_data_dict: Optional dictionary for chat data storage.
        """
        self._user_id = user_id
        self._thread_id = 0
        self._user_data_dict = user_data_dict or {}
        self._chat_data_dict = chat_data_dict or {}
        self._last_user_message: str | None = None

    @property
    def chat_id(self) -> str | int:
        """Returns the unique identifier for the current chat.

        In terminal mode, we use a fixed chat ID.

        Returns:
            The chat identifier (terminal session ID).
        """
        return f"terminal_{self._user_id}"

    @property
    def user_id(self) -> int:
        """Returns the unique identifier for the current user.

        Returns:
            The user identifier.
        """
        return self._user_id

    @property
    def thread_id(self) -> int:
        """Returns the thread ID.

        In terminal mode, thread ID is always 0.

        Returns:
            The thread identifier (always 0).
        """
        return self._thread_id

    @property
    def user_data(self) -> str:
        """Returns a string representation of the user data.

        Returns:
            The user data string.
        """
        return f"User #{self._user_id}"

    @property
    def chat_data(self) -> str:
        """Returns a string representation of the chat data.

        Returns:
            The chat data string.
        """
        return f"Terminal chat #{self.chat_id}, thread #{self._thread_id}"

    @property
    def attached_document(self) -> dict[str, str] | None:
        """Returns the attached document data if present, otherwise None.

        Returns:
            None (attachments not supported in terminal mode).
        """
        return None

    @property
    def attached_document_caption(self) -> str | None:
        """Returns the caption of the attached document if present, otherwise None.

        Returns:
            None (attachments not supported in terminal mode).
        """
        return None

    async def get_text_prompt(self) -> str | None:
        """Retrieves the text prompt from the current message.

        Returns:
            The text prompt string or None.
        """
        return self._last_user_message

    async def get_voice_prompt(self) -> BytesIO | None:
        """Retrieves the voice prompt as a BytesIO object if present.

        In terminal mode, voice prompts are not supported.

        Returns:
            None (voice not supported in terminal).
        """
        return None

    async def get_caption(self) -> str | None:
        """Retrieves the caption from the attached document.

        Returns:
            None (attachments not supported in terminal mode).
        """
        return None

    def set_caption(self, caption: str) -> None:
        """Sets the caption for the next message.

        In terminal mode, this is a no-op (attachments not supported).

        Args:
            caption: The caption to set.
        """
        pass

    async def send_action_typing(self) -> None:
        """Sends a typing action to the user.

        In terminal mode, this is a no-op (typing indicator not needed).
        """
        pass

    async def send_action_uploading_photo(self) -> None:
        """Sends an uploading photo action to the user.

        In terminal mode, this is a no-op.
        """
        pass

    async def send_action_recording(self) -> None:
        """Sends a recording voice action to the user.

        In terminal mode, this is a no-op.
        """
        pass

    async def send_reaction(self, reaction: str) -> None:
        """Sends a reaction to the user's message.

        In terminal mode, this prints the reaction.

        Args:
            reaction: The reaction to send.
        """
        _console.print(f"[dim]{reaction}[/dim]")

    async def delete_last_user_message(self) -> None:
        """Deletes the last message sent by the user.

        In terminal mode, this is a no-op (can't delete terminal history).
        """
        pass

    async def send_message(self, message: str, reply: bool = True, **kwargs: Any) -> None:
        """Sends a text message to the user.

        Args:
            message: The text content to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments (unused in terminal).
        """
        # Try to render as markdown, fall back to plain text
        try:
            md = Markdown(message)
            _console.print(md)
        except Exception:
            _console.print(message)

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

        In terminal mode, just prints the path/filename.

        Args:
            audio: The audio data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the audio.
            caption: The caption for the audio.
            performer: The performer of the audio.
            duration: The duration of the audio in seconds.
            thumbnail: The thumbnail data for the audio.
            filename: The filename for the audio.
            **kwargs: Additional arguments (unused in terminal).
        """
        audio_info = f"Audio: {filename or 'unknown'}"
        if title:
            audio_info += f" - {title}"
        if performer:
            audio_info += f" by {performer}"
        if isinstance(audio, str):
            audio_info += f" (path: {audio})"
        _console.print(f"[dim]{audio_info}[/dim]")
        if caption:
            _console.print(f"[dim]Caption: {caption}[/dim]")

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

        In terminal mode, just prints the path/filename.

        Args:
            video: The video data or path to send.
            reply: Whether to reply to the user's message.
            title: The title of the video.
            caption: The caption for the video.
            duration: The duration of the video in seconds.
            thumbnail: The thumbnail data for the video.
            filename: The filename for the video.
            **kwargs: Additional arguments (unused in terminal).
        """
        video_info = f"Video: {filename or 'unknown'}"
        if title:
            video_info += f" - {title}"
        if isinstance(video, str):
            video_info += f" (path: {video})"
        _console.print(f"[dim]{video_info}[/dim]")
        if caption:
            _console.print(f"[dim]Caption: {caption}[/dim]")

    async def send_images(self, images: list[BytesIO] | list[str], reply: bool = True, **kwargs: Any) -> None:
        """Sends a list of images to the user.

        In terminal mode, just prints the paths/URLs.

        Args:
            images: A list of image data or paths to send.
            reply: Whether to reply to the user's message.
            **kwargs: Additional arguments (unused in terminal).
        """
        _console.print("[bold]Generated image(s):[/bold]")
        for i, img in enumerate(images, 1):
            if isinstance(img, str):
                _console.print(f"  {i}. {img}")
            else:
                _console.print(f"  {i}. [image data in memory]")

    async def send_document(
        self,
        document: bytes | BytesIO,
        filename: str | None = None,
        caption: str | None = None,
        thumbnail: bytes | None = None,
        **kwargs: Any,
    ) -> None:
        """Sends a document file to the user.

        In terminal mode, just prints the filename.

        Args:
            document: The document data to send.
            filename: The filename for the document.
            caption: The caption for the document.
            thumbnail: The thumbnail data for the document.
            **kwargs: Additional arguments (unused in terminal).
        """
        doc_info = f"Document: {filename or 'unknown'}"
        _console.print(f"[dim]{doc_info}[/dim]")
        if caption:
            _console.print(f"[dim]Caption: {caption}[/dim]")

    def set_last_message(self, message: str | None) -> None:
        """Sets the last user message for prompt retrieval.

        Args:
            message: The user's message text, or None to clear.
        """
        self._last_user_message = message
