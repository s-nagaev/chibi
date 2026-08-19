"""Regression coverage for UTF-8 framing of IDE protocol writes."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

import chibi.config  # noqa: F401
from chibi.runners.ide_transport import IDEStdioRunner

NON_ASCII_CONTENT = "Привет, мир"


class Cp1252Stdout:
    """Text stream emulating a Windows console restricted to cp1252."""

    encoding = "cp1252"

    def __init__(self) -> None:
        """Initialize the emulated stream with a real binary buffer."""
        self.buffer = io.BytesIO()
        self.text: list[str] = []
        self.flushed = 0

    def write(self, data: str) -> int:
        """Write text, rejecting anything cp1252 cannot encode.

        Args:
            data: Text to write to the emulated console.

        Returns:
            The number of characters accepted.

        Raises:
            UnicodeEncodeError: If the text contains non-cp1252 characters.
        """
        data.encode(self.encoding)
        self.text.append(data)
        return len(data)

    def flush(self) -> None:
        """Record a flush of the text layer."""
        self.flushed += 1


class TextOnlyStdout:
    """Text stream without a binary buffer, forcing the text fallback path."""

    encoding = "utf-8"
    buffer = None

    def __init__(self) -> None:
        """Initialize the fallback stream capture."""
        self.text: list[str] = []
        self.flushed = 0

    def write(self, data: str) -> int:
        """Capture written text.

        Args:
            data: Text to write.

        Returns:
            The number of characters accepted.
        """
        self.text.append(data)
        return len(data)

    def flush(self) -> None:
        """Record a flush of the text layer."""
        self.flushed += 1


def expected_line(message: dict[str, Any]) -> bytes:
    """Build the expected UTF-8 wire representation of a frame.

    Args:
        message: Protocol frame to encode.

    Returns:
        The UTF-8 encoded compact JSONL line.
    """
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


@pytest.mark.asyncio
async def test_non_ascii_frame_is_written_as_utf8_on_cp1252_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-ASCII content reaches stdout as UTF-8 bytes on a cp1252 console."""
    stdout = Cp1252Stdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    runner = IDEStdioRunner()
    frame: dict[str, Any] = {"type": "result", "content": NON_ASCII_CONTENT}

    await runner._write(frame)

    assert stdout.buffer.getvalue() == expected_line(frame)
    assert stdout.text == []
    assert json.loads(stdout.buffer.getvalue().decode("utf-8"))["content"] == NON_ASCII_CONTENT


@pytest.mark.asyncio
async def test_ascii_frame_is_written_as_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASCII frames use the same binary path and stay byte-identical."""
    stdout = Cp1252Stdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    runner = IDEStdioRunner()
    frame: dict[str, Any] = {"type": "status", "request_id": "abc", "state": "running"}

    await runner._write(frame)

    assert stdout.buffer.getvalue() == expected_line(frame)
    assert stdout.text == []


@pytest.mark.asyncio
async def test_consecutive_frames_are_newline_delimited(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sequential writes produce one UTF-8 JSONL line per frame."""
    stdout = Cp1252Stdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    runner = IDEStdioRunner()
    first: dict[str, Any] = {"type": "result", "content": NON_ASCII_CONTENT}
    second: dict[str, Any] = {"type": "result", "content": "ok"}

    await runner._write(first)
    await runner._write(second)

    assert stdout.buffer.getvalue() == expected_line(first) + expected_line(second)
    lines = stdout.buffer.getvalue().decode("utf-8").splitlines()
    assert [json.loads(line)["content"] for line in lines] == [NON_ASCII_CONTENT, "ok"]


@pytest.mark.asyncio
async def test_stream_without_buffer_falls_back_to_text_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streams exposing no binary buffer still receive the frame as text."""
    stdout = TextOnlyStdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    runner = IDEStdioRunner()
    frame: dict[str, Any] = {"type": "result", "content": NON_ASCII_CONTENT}

    await runner._write(frame)

    assert stdout.text == [expected_line(frame).decode("utf-8")]
    assert stdout.flushed == 1
