"""Tests for Markdown table detection and Rich Message rendering in telegram utils."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update

import chibi.config  # noqa: F401
from chibi.utils.telegram import (
    _render_ascii_table,
    _split_table_parts,
    detect_markdown_table,
    render_table_rich,
    send_long_message,
)


def test_detect_markdown_table_returns_parsed_rows() -> None:
    """A Markdown table with a separator row is parsed into rows."""
    text = "Here is data:\n| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"

    rows = detect_markdown_table(text)

    assert rows == [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]


def test_detect_markdown_table_strips_whitespace_from_cells() -> None:
    """Cells are stripped of surrounding whitespace."""
    text = "|  col1  |  col2  |\n|  ---  |  ---  |\n|  a  |  b  |"

    rows = detect_markdown_table(text)

    assert rows == [["col1", "col2"], ["a", "b"]]


def test_detect_markdown_table_returns_none_without_table() -> None:
    """Plain text without pipe rows yields None."""
    assert detect_markdown_table("Just some plain text\nNo tables here") is None


def test_detect_markdown_table_single_pipe_line_is_not_table() -> None:
    """A single pipe-delimited line is not treated as a table."""
    assert detect_markdown_table("| just a pipe |") is None


def test_detect_markdown_table_separator_only_block_is_not_table() -> None:
    """A block consisting only of separator rows is not a table."""
    assert detect_markdown_table("| --- | --- |") is None


def test_detect_markdown_table_handles_alignment_separators() -> None:
    """Separator rows with colons are skipped."""
    text = "| A | B |\n|:---|:---:|\n| 1 | 2 |"

    rows = detect_markdown_table(text)

    assert rows == [["A", "B"], ["1", "2"]]


def test_detect_markdown_table_bold_cells_kept() -> None:
    """Bold markdown inside cells is preserved as-is."""
    text = "| **Name** | Age |\n| --- | --- |\n| Alice | 30 |"

    rows = detect_markdown_table(text)

    assert rows == [["**Name**", "Age"], ["Alice", "30"]]


def test_split_table_parts_interleaves_text_and_tables() -> None:
    """Multiple tables are extracted while surrounding text is preserved."""
    text = "Before\n| A | B |\n| - | - |\n| 1 | 2 |\nBetween\n| C | D |\n| - | - |\n| 3 | 4 |\nAfter"

    parts = _split_table_parts(text)

    assert parts == [
        ("Before", None),
        (None, [["A", "B"], ["1", "2"]]),
        ("Between", None),
        (None, [["C", "D"], ["3", "4"]]),
        ("After", None),
    ]


def test_split_table_parts_text_only() -> None:
    """A message without tables yields a single text part."""
    parts = _split_table_parts("Plain text only")

    assert parts == [("Plain text only", None)]


def test_render_ascii_table_formats_columns() -> None:
    """ASCII table aligns columns and adds a header separator."""
    rendered = _render_ascii_table([["Name", "Age"], ["Alice", "30"], ["Bob", "25"]])

    assert rendered == ("| Name  | Age |\n|-------|-----|\n| Alice | 30  |\n| Bob   | 25  |")


def test_render_ascii_table_empty_returns_empty_string() -> None:
    """An empty table renders as an empty string."""
    assert _render_ascii_table([]) == ""


async def _make_context() -> MagicMock:
    """Build a mocked context whose bot exposes do_api_request and send_message."""
    context = MagicMock()
    context.bot.do_api_request = AsyncMock(return_value={"ok": True})
    context.bot.send_message = AsyncMock(return_value=None)
    return context


@pytest.mark.asyncio
async def test_render_table_rich_sends_rich_message_with_header() -> None:
    """A table with a non-empty first row sends a header-bearing Rich Message."""
    context = await _make_context()

    await render_table_rich(
        table_rows=[["Name", "Age"], ["Alice", "30"]],
        context=context,
        chat_id=12345,
        thread_id=None,
    )

    context.bot.do_api_request.assert_awaited_once_with(
        "sendRichMessage",
        api_kwargs={
            "chat_id": 12345,
            "rich_message": {
                "blocks": [
                    {
                        "type": "table",
                        "cells": [
                            [{"text": "Name"}, {"text": "Age"}],
                            [{"text": "Alice"}, {"text": "30"}],
                        ],
                    }
                ],
            },
        },
    )


@pytest.mark.asyncio
async def test_render_table_rich_falls_back_to_ascii_on_failure() -> None:
    """A failing sendRichMessage falls back to an ASCII plain-text table."""
    context = await _make_context()
    context.bot.do_api_request = AsyncMock(side_effect=Exception("API error"))

    await render_table_rich(
        table_rows=[["Name", "Age"], ["Alice", "30"]],
        context=context,
        chat_id=12345,
        thread_id=7,
    )

    context.bot.send_message.assert_awaited_once_with(
        chat_id=12345,
        text="| Name  | Age |\n|-------|-----|\n| Alice | 30  |",
        message_thread_id=7,
    )


def _make_update() -> Update:
    """Build a minimal update with a chat and a message."""
    return cast(
        Update,
        SimpleNamespace(
            effective_chat=SimpleNamespace(id=12345),
            effective_message=SimpleNamespace(message_id=111, message_thread_id=None),
        ),
    )


@pytest.mark.asyncio
async def test_send_long_message_routes_table_to_rich_and_text_to_normal() -> None:
    """Tables go to Rich Messages while surrounding text uses the normal path."""
    update = _make_update()
    context = await _make_context()

    with patch("chibi.utils.telegram.send_message", new=AsyncMock()) as mock_send:
        await send_long_message(
            message="Results:\n| Name | Age |\n| --- | --- |\n| Alice | 30 |",
            update=update,
            context=context,
            parse_mode="MarkdownV2",
            normalize_md=True,
        )

    context.bot.do_api_request.assert_awaited_once()
    call = context.bot.do_api_request.await_args
    assert call is not None
    assert call.args[0] == "sendRichMessage"
    assert call.kwargs["api_kwargs"]["chat_id"] == 12345
    mock_send.assert_awaited_once()
    send_call = mock_send.await_args
    assert send_call is not None
    assert send_call.kwargs["text"].startswith("Results:")


@pytest.mark.asyncio
async def test_send_long_message_without_table_uses_normal_path() -> None:
    """Non-table messages do not trigger Rich Message calls."""
    update = _make_update()
    context = await _make_context()

    with patch("chibi.utils.telegram.send_message", new=AsyncMock()) as mock_send:
        await send_long_message(
            message="Just a normal message",
            update=update,
            context=context,
            parse_mode="MarkdownV2",
            normalize_md=True,
        )

    context.bot.do_api_request.assert_not_awaited()
    mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_long_message_plain_text_path_with_table() -> None:
    """normalize_md=False path still routes tables to Rich Messages."""
    update = _make_update()
    context = await _make_context()

    with patch("chibi.utils.telegram.send_message", new=AsyncMock()) as mock_send:
        await send_long_message(
            message="Table:\n| A | B |\n| - | - |\n| 1 | 2 |\nend",
            update=update,
            context=context,
            normalize_md=False,
        )

    context.bot.do_api_request.assert_awaited_once()
    assert mock_send.await_count == 2


@pytest.mark.asyncio
async def test_send_long_message_multiple_tables_each_get_rich_message() -> None:
    """Each table in a multi-table message gets its own sendRichMessage."""
    update = _make_update()
    context = await _make_context()

    with patch("chibi.utils.telegram.send_message", new=AsyncMock()) as mock_send:
        await send_long_message(
            message=("T1\n| A | B |\n| - | - |\n| 1 | 2 |\nT2\n| C | D |\n| - | - |\n| 3 | 4 |"),
            update=update,
            context=context,
            normalize_md=False,
        )

    assert context.bot.do_api_request.await_count == 2
    assert mock_send.await_count == 2
