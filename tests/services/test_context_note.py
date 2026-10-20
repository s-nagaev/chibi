"""Tests for the volatile context note: attach, strip and shape coverage."""

import json
from typing import Any

from mistralai.models import UserMessage as MistralUserMessage

from chibi.models import Message
from chibi.services.context_note import (
    CONTEXT_NOTE_OPEN,
    attach_context_note,
    format_context_note,
    note_is_present,
    strip_context_note,
)

NOTE = format_context_note(["approximate_context_size: 42 tokens (0% of 200,000 limit)"])
assert NOTE, "sanity: the test note must be non-empty"

USER_TEXT = "What is the context size?"


def count_notes(text: str) -> int:
    return text.count(CONTEXT_NOTE_OPEN)


# --- format / strip primitives -------------------------------------------------


def test_format_context_note_empty_lines_produce_empty_note() -> None:
    assert format_context_note([]) == ""


def test_strip_context_note_clean_input_is_byte_identical() -> None:
    """A message without the note passes through byte for byte."""
    for clean in [
        USER_TEXT,
        json.dumps({"user": "telegram", "prompt": "hi", "type": "user message"}),
        "",
    ]:
        assert strip_context_note(clean) == clean


def test_strip_context_note_removes_note_and_separator() -> None:
    dirty = USER_TEXT + "\n\n" + NOTE
    assert strip_context_note(dirty) == USER_TEXT


def test_strip_context_note_is_idempotent() -> None:
    dirty = USER_TEXT + "\n\n" + NOTE
    once = strip_context_note(dirty)
    assert strip_context_note(once) == once == USER_TEXT


def test_note_is_present() -> None:
    assert note_is_present(USER_TEXT + NOTE)
    assert not note_is_present(USER_TEXT)
    assert not note_is_present("context_note without brackets")


# --- attach per converted payload shape ----------------------------------------


def test_attach_openai_chat_shape_string_content() -> None:
    payload = [Message(role="user", content=USER_TEXT).to_openai()]
    attach_context_note(list(payload), NOTE)
    assert payload[-1]["content"] == USER_TEXT + "\n\n" + NOTE
    # The internal Message object is untouched.
    assert Message(role="user", content=USER_TEXT).to_openai()["content"] == USER_TEXT


def test_attach_anthropic_shape_block_list_content() -> None:
    payload = [Message(role="user", content=USER_TEXT).to_anthropic()]
    attach_context_note(list(payload), NOTE)
    content = payload[-1]["content"]
    assert isinstance(content, list)
    # A single text block: the note is appended to it in place.
    assert content[0]["text"] == USER_TEXT + "\n\n" + NOTE
    assert len(content) == 1


def test_attach_google_shape_parts() -> None:
    payload = [Message(role="user", content=USER_TEXT).to_google()]
    attach_context_note(list(payload), NOTE)
    parts = payload[-1]["parts"] or []
    # A single text part: the note is appended to it in place.
    assert parts[0]["text"] == USER_TEXT + "\n\n" + NOTE
    assert len(parts) == 1


def test_attach_mistral_shape_sdk_object() -> None:
    payload: list[Any] = [Message(role="user", content=USER_TEXT).to_mistral()]
    attach_context_note(list(payload), NOTE)
    assert isinstance(payload[-1], MistralUserMessage)
    assert payload[-1].content == USER_TEXT + "\n\n" + NOTE


def test_attach_responses_shape_input_text_blocks() -> None:
    payload = Message(role="user", content=USER_TEXT).to_responses_items()
    attach_context_note(list(payload), NOTE)
    item = payload[-1]
    assert item["type"] == "message" and item["role"] == "user"
    blocks = item["content"]
    # A single input_text block: the note is appended to it in place.
    assert blocks[0] == {"type": "input_text", "text": USER_TEXT + "\n\n" + NOTE}
    assert len(blocks) == 1


def test_attach_lands_on_the_last_user_message_not_on_the_seed() -> None:
    """After a compaction the request is [seed user, seed assistant, fresh user turn].

    The note must land on the fresh turn only: the seed is persisted history.
    """
    seed_user = Message(role="user", content="What we were talking about?", source="compaction_seed")
    seed_assistant = Message(role="assistant", content="Summary of everything.", source="compaction_seed")
    fresh_turn = Message(role="user", content="And what happened next?")
    for converter in (lambda m: m.to_openai(), lambda m: m.to_anthropic(), lambda m: m.to_google()):
        payload = [converter(seed_user), converter(seed_assistant), converter(fresh_turn)]
        attach_context_note(payload, NOTE)
        attached = payload[-1]
        text = json.dumps(attached)
        assert count_notes(text) == 1
        assert CONTEXT_NOTE_OPEN in text
        seed_text = json.dumps(payload[0]) + json.dumps(payload[1])
        assert CONTEXT_NOTE_OPEN not in seed_text


def test_attach_skips_user_role_tool_result_envelopes() -> None:
    """Anthropic turns a role="tool" result into a user-role tool_result param.

    The note must never land on it, even when it is the last user-role item.
    """
    tool_result_param = Message(role="tool", content='{"ok": true}', tool_call_id="call_1").to_anthropic()
    assert tool_result_param["role"] == "user"
    payload: list[Any] = [Message(role="user", content=USER_TEXT).to_anthropic(), tool_result_param]
    attach_context_note(payload, NOTE)
    assert count_notes(json.dumps(payload[0])) == 1
    assert CONTEXT_NOTE_OPEN not in json.dumps(payload[1])


def test_attach_empty_note_is_noop() -> None:
    payload = [Message(role="user", content=USER_TEXT).to_openai()]
    attach_context_note(payload, "")
    content = payload[-1]["content"]
    assert isinstance(content, str)
    assert content == USER_TEXT


def test_attach_without_any_user_message_is_noop() -> None:
    payload: list[Any] = [Message(role="assistant", content="hi").to_openai()]
    attach_context_note(payload, NOTE)
    assert CONTEXT_NOTE_OPEN not in json.dumps(payload, default=str)


# --- idempotency ---------------------------------------------------------------


def test_double_attach_is_single_note() -> None:
    """A payload that already carries the note receives no second copy.

    This is the contract the OpenAI Responses -> Chat Completions fallback and
    every retry rely on: the same list must never end up with two notes.
    """
    payload = [Message(role="user", content=USER_TEXT).to_openai()]
    attach_context_note(payload, NOTE)
    attach_context_note(payload, NOTE)
    attach_context_note(payload, NOTE)
    content = payload[-1]["content"]
    assert isinstance(content, str)
    assert count_notes(content) == 1
    assert content == USER_TEXT + "\n\n" + NOTE


def test_double_attach_on_block_and_parts_shapes_is_single_note() -> None:
    for converter in (lambda m: m.to_anthropic(), lambda m: m.to_google(), lambda m: m.to_responses_items()[0]):
        payload = [converter(Message(role="user", content=USER_TEXT))]
        attach_context_note(payload, NOTE)
        attach_context_note(payload, NOTE)
        assert count_notes(json.dumps(payload, default=str)) == 1


def test_double_attach_mistral_object_is_single_note() -> None:
    payload: list[Any] = [Message(role="user", content=USER_TEXT).to_mistral()]
    attach_context_note(payload, NOTE)
    attach_context_note(payload, NOTE)
    mistral_content = payload[-1].content
    assert isinstance(mistral_content, str)
    assert count_notes(mistral_content) == 1


def test_note_on_another_message_still_attaches_to_the_last_user() -> None:
    """Idempotency is per target message: a note elsewhere does not block a fresh attach."""
    older = Message(role="user", content=USER_TEXT).to_openai()
    older["content"] = USER_TEXT + "\n\n" + NOTE
    fresh = Message(role="user", content="Newer question").to_openai()
    payload: list[Any] = [older, fresh]
    attach_context_note(payload, NOTE)
    fresh_content = fresh["content"]
    older_content = older["content"]
    assert isinstance(fresh_content, str) and isinstance(older_content, str)
    assert count_notes(fresh_content) == 1
    assert count_notes(older_content) == 1
