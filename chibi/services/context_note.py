"""The volatile per-turn context note: assembly helpers, attach and strip.

The note carries the volatile keys of a turn (the approximate context size and
both context warnings). It used to travel inside the system payload, which it
invalidated on every turn; it now travels as text appended to the END of the
last user message of the CONVERTED provider payload, and is never persisted:
the persistence boundary (:func:`chibi.memory.chroma.with_chroma_archival`)
strips it on write.

This module is deliberately dependency-free (standard library only): it is
imported by both the provider layer and the storage wrapper, which sit on
opposite sides of an import cycle otherwise.
"""

import json
import re
from typing import Any

CONTEXT_NOTE_OPEN = "<context_note>"
CONTEXT_NOTE_CLOSE = "</context_note>"

# The note is attached as ``text + "\n\n" + note``; the strip removes the note
# together with the whitespace that preceded it, so a stripped content is
# byte-identical to the pre-attach content.
_CONTEXT_NOTE_RE = re.compile(r"\n*<context_note>.*?</context_note>", re.DOTALL)


def format_context_note(lines: list[str]) -> str:
    """Wrap the volatile note lines into the delimited block.

    Args:
        lines: The note lines in display order.

    Returns:
        The delimited note text, or an empty string when there are no lines.
    """
    if not lines:
        return ""
    return CONTEXT_NOTE_OPEN + "\n" + "\n".join(lines) + "\n" + CONTEXT_NOTE_CLOSE


def note_is_present(text: str) -> bool:
    """Check whether a text already carries the note marker.

    Args:
        text: The text to inspect.

    Returns:
        True when the opening marker occurs in the text.
    """
    return CONTEXT_NOTE_OPEN in text


def strip_context_note(text: str) -> str:
    """Remove the note from a message text.

    Idempotent on clean input: a text without the marker is returned
    byte-for-byte unchanged.

    Args:
        text: The message content to clean.

    Returns:
        The text without the note (and without the separator that preceded it).
    """
    if not note_is_present(text):
        return text
    return _CONTEXT_NOTE_RE.sub("", text)


def attach_context_note(messages: list[Any], note: str) -> None:
    """Attach the note to the LAST user message of a converted provider payload.

    The payload must already be in provider-native format (the fresh, call-owned
    structures produced by ``Message.to_openai`` / ``to_anthropic`` / ``to_google``
    / ``to_mistral`` / ``to_responses_items``); internal ``Message`` objects are
    never touched, so the caller's conversation list stays clean and a retry that
    re-converts the same messages cannot inherit an old note.

    Handled shapes:
      - dict with string ``content`` (OpenAI Chat Completions, Mistral dicts);
      - dict with a list ``content`` of blocks carrying a ``text`` key
        (Anthropic ``TextBlockParam``, Responses ``input_text`` blocks);
      - dict with a ``parts`` list carrying ``text`` keys (Google ``ContentDict``);
      - Mistral SDK message objects with a string ``content`` attribute.

    Tool-result envelopes that merely share the user role (Anthropic
    ``tool_result`` blocks, Google ``function_response`` parts) are skipped: the
    note never lands on a tool result.

    Idempotent: a target message that already carries the marker receives no
    second copy, so a payload that passes two attach sites (the OpenAI Responses
    fallback into the generic path) keeps exactly one note.

    Args:
        messages: The converted payload message list.
        note: The note text; an empty note is a no-op.
    """
    if not note:
        return
    for message in reversed(messages):
        if not _is_user_message(message):
            continue
        if _note_already_present(message):
            return
        if _is_tool_result_envelope(message):
            continue
        _append_text(message, note)
        return


def _is_user_message(message: Any) -> bool:
    """Check the user role across dict payloads and SDK message objects."""
    if isinstance(message, dict):
        return message.get("role") == "user"
    return getattr(message, "role", None) == "user"


def _is_tool_result_envelope(message: Any) -> bool:
    """Detect a user-role envelope that only carries tool results.

    Anthropic converts a ``role="tool"`` Message into a user-role param whose
    blocks are ``tool_result``; Google does the same with ``function_response``
    parts. Neither is a place for the note.
    """
    if not isinstance(message, dict):
        return False
    blocks = message.get("content") or message.get("parts")
    if not isinstance(blocks, list) or not blocks:
        return False
    return all(
        isinstance(block, dict) and (block.get("type") == "tool_result" or "function_response" in block)
        for block in blocks
    )


def _note_already_present(message: Any) -> bool:
    """Check the whole textual content of a converted message for the marker."""
    if isinstance(message, dict):
        candidates: list[Any] = [message.get("content"), message.get("parts")]
        for content in candidates:
            if isinstance(content, str):
                if note_is_present(content):
                    return True
            elif content is not None:
                if note_is_present(json.dumps(content, default=str)):
                    return True
        return False
    content = getattr(message, "content", None)
    return isinstance(content, str) and note_is_present(content)


def _append_text(message: Any, note: str) -> None:
    """Append the note to the text of a converted user message, by shape."""
    if isinstance(message, dict):
        if "parts" in message:
            _append_to_block_list(blocks=message.setdefault("parts", []), note=note, default_type="text")
            return
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + "\n\n" + note
            return
        if isinstance(content, list):
            _append_to_block_list(blocks=content, note=note, default_type="text")
            return
        return
    content = getattr(message, "content", None)
    if isinstance(content, str):
        message.content = content + "\n\n" + note


def _append_to_block_list(blocks: list[Any], note: str, default_type: str) -> None:
    """Append the note to the last text block, or add a fresh text block.

    The fresh block copies the ``type`` of the existing text blocks so the
    payload stays protocol-valid (``input_text`` for Responses items, ``text``
    for Anthropic blocks and Google parts).
    """
    block_type = default_type
    for block in blocks:
        if isinstance(block, dict) and isinstance(block.get("type"), str) and isinstance(block.get("text"), str):
            block_type = block["type"]
    for block in reversed(blocks):
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] = block["text"] + "\n\n" + note
            return
    blocks.append({"type": block_type, "text": note})
