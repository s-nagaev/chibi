"""
RichMessageBuilder — constructs Telegram Bot API payloads for Rich Messages.

This is a pure dict builder with no PTB dependency and no HTTP calls.
Supports:
    - sendRichMessageDraft (LLM thinking/thoughts blocks)
    - sendRichMessage (table blocks)

References:
    - Telegram Bot API 10.1/10.2
    - InputRichMessage, InputRichBlock, InputRichBlockTableCell
"""

from __future__ import annotations

from typing import Any


class RichMessageBuilder:
    """Builder for Telegram Rich Message payloads.

    All methods return plain Python dicts for bot.do_api_request().
    """

    @staticmethod
    def build_thinking_draft(
        thoughts: str,
        chat_id: int | str,
        thread_id: int | None,
    ) -> dict[str, Any]:
        """Build a ``sendRichMessageDraft`` payload for an LLM thinking block.

        Args:
            thoughts: The LLM's thoughts/reasoning content.
            chat_id: Target chat ID.
            thread_id: Message thread ID (``None`` to omit).

        Returns:
            Dict for ``bot.do_api_request("sendRichMessageDraft", api_kwargs=payload)``.

        Note:
            Caller must set ``draft_id`` (non-zero int) on the returned dict.
        """
        payload: dict[str, Any] = {"chat_id": chat_id}

        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        payload["rich_message"] = {
            "blocks": [{"type": "thinking", "text": thoughts}],
        }

        return payload

    @staticmethod
    def build_table_message(
        table_data: list[list[str]],
        chat_id: int | str,
        thread_id: int | None,
        headers: list[str] | None = None,
        *,
        caption: str | None = None,
    ) -> dict[str, Any]:
        """Build a ``sendRichMessage`` payload with a table block.

        Args:
            table_data: List of rows, each row is list of cell strings.
            chat_id: Target chat ID.
            thread_id: Message thread ID (``None`` to omit).
            headers: Optional header row prepended before data rows.
            caption: Optional text paragraph before the table.

        Returns:
            Dict for ``bot.do_api_request("sendRichMessage", api_kwargs=payload)``.
        """
        col_count = RichMessageBuilder._compute_column_count(table_data, headers)

        blocks: list[dict[str, Any]] = []

        if caption:
            blocks.append({"type": "paragraph", "text": caption})

        table_cells: list[list[dict[str, Any]]] = []

        if headers is not None:
            table_cells.append(RichMessageBuilder._build_cells(headers, col_count))

        for row in table_data:
            if RichMessageBuilder._is_empty_row(row):
                continue
            table_cells.append(RichMessageBuilder._build_cells(row, col_count))

        blocks.append({"type": "table", "cells": table_cells})

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "rich_message": {"blocks": blocks},
        }

        if thread_id is not None:
            payload["message_thread_id"] = thread_id

        return payload

    @staticmethod
    def _compute_column_count(table_data: list[list[str]], headers: list[str] | None) -> int:
        """Compute the maximum column count across table data and optional headers.

        Args:
            table_data: List of table rows.
            headers: Optional header row.

        Returns:
            Maximum number of columns, at least 1.
        """
        max_cols = 0
        if headers is not None:
            max_cols = len(headers)
        for row in table_data:
            if row:
                max_cols = max(max_cols, len(row))
        return max_cols if max_cols > 0 else 1

    @staticmethod
    def _is_empty_row(row: list[str]) -> bool:
        """Check whether all cells in a row are empty.

        Args:
            row: List of cell strings.

        Returns:
            True if every cell is empty or whitespace-only.
        """
        return all(cell.strip() == "" for cell in row)

    @staticmethod
    def _pad_row(row: list[str], col_count: int) -> list[str]:
        """Pad or trim a row to exactly *col_count* cells.

        Args:
            row: List of cell strings.
            col_count: Target number of columns.

        Returns:
            Row padded with empty strings or trimmed to the target length.
        """
        if len(row) >= col_count:
            return list(row[:col_count])
        return list(row) + [""] * (col_count - len(row))

    @staticmethod
    def _build_cells(
        cells: list[str],
        col_count: int,
    ) -> list[dict[str, Any]]:
        """Build a table row for the Rich Message API.

        Args:
            cells: List of cell strings.
            col_count: Target number of columns.

        Returns:
            List of ``{"text": str}`` dicts representing a table row.
        """
        padded = RichMessageBuilder._pad_row(cells, col_count)
        return [{"text": cell} for cell in padded]
