"""
File editing utilities for the project.
This module provides functions for efficiently editing files through LLM tools.
"""

import os
import re
from pathlib import Path
from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import gpt_settings
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions


class ReplaceInFileTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="replace_in_file",
            description="Replace occurrences of text in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to be replaced.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of replacements to make (-1 for all).",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "old_text", "new_text"],
            },
        ),
    )
    name = "replace_in_file"

    @classmethod
    async def function(
        cls,
        full_path: str,
        old_text: str,
        new_text: str,
        count: int = -1,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, int]:
        """
        Replace occurrences of text in a file.

        Args:
            full_path: Path to the file
            old_text: Text to be replaced
            new_text: Replacement text
            count: Maximum number of replacements to make (-1 for all)
            encoding: File encoding

        Returns:
            Dict containing the number of replacements made
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                content = f.read()

            new_content, replacements = content, 0
            if count == -1:
                new_content, replacements = content.replace(old_text, new_text), content.count(old_text)
            else:
                new_content, replacements = re.subn(
                    re.escape(old_text), new_text.replace("\\", "\\\\"), content, count=count
                )

            if replacements > 0:
                with path.open("w", encoding=encoding) as f:
                    f.write(new_content)
                logger.log("TOOL", f"Made {replacements} replacements in {path}")
            else:
                logger.log("TOOL", f"No occurrences of '{old_text}' found in {path}")

            return {"replacements": replacements}
        except Exception as e:
            logger.error(f"Error replacing text in {full_path}: {e}")
            raise


class ReplaceInFileRegexTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="replace_in_file_regex",
            description="Replace text in a file using regex patterns.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to match.",
                    },
                    "replacement": {
                        "type": "string",
                        "description": "Replacement text (can include regex group references).",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of replacements to make (-1 for all).",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "pattern", "replacement"],
            },
        ),
    )
    name = "replace_in_file_regex"

    @classmethod
    async def function(
        cls,
        full_path: str,
        pattern: str,
        replacement: str,
        count: int = -1,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, int]:
        """Replace text in a file using regex patterns.

        Args:
            full_path: Path to the file
            pattern: Regex pattern to match
            replacement: Replacement text (can include regex group references)
            count: Maximum number of replacements to make (-1 for all)
            encoding: File encoding

        Returns:
            Dict containing the number of replacements made
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                content = f.read()

            new_content, replacements = re.subn(pattern, replacement, content, count=count)

            if replacements > 0:
                with path.open("w", encoding=encoding) as f:
                    f.write(new_content)
                logger.log("TOOL", f"Made {replacements} regex replacements in {path}")
            else:
                logger.log("TOOL", f"No matches for pattern '{pattern}' found in {path}")

            return {"replacements": replacements}
        except Exception as e:
            logger.error(f"Error replacing text with regex in {full_path}: {e}")
            raise


class InsertAtLineTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="insert_at_line",
            description="Insert content at a specific line number in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number where to insert (0-indexed).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "line_number", "content"],
            },
        ),
    )
    name = "insert_at_line"

    @classmethod
    async def function(
        cls,
        full_path: str,
        line_number: int,
        content: str,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, bool]:
        """Insert content at a specific line number in a file.

        Args:
            full_path: Path to the file
            line_number: Line number where to insert (0-indexed)
            content: Content to insert
            encoding: File encoding

        Returns:
            Dict indicating success
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                lines = f.readlines()

            line_count = len(lines)
            if line_number < 0:
                line_number = max(0, line_count + line_number)
            elif line_number > line_count:
                line_number = line_count

            if content and not content.endswith("\n"):
                content += "\n"

            lines.insert(line_number, content)

            with path.open("w", encoding=encoding) as f:
                f.writelines(lines)

            logger.log("TOOL", f"Inserted content at line {line_number} in {path}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Error inserting at line {line_number} in {full_path}: {e}")
            raise


class ReplaceLinesTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="replace_lines",
            description="Replace a range of lines in a file with new content.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (0-indexed).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace (inclusive).",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to insert.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "start_line", "end_line", "new_content"],
            },
        ),
    )
    name = "replace_lines"

    @classmethod
    async def function(
        cls,
        full_path: str,
        start_line: int,
        end_line: int,
        new_content: str,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, int]:
        """Replace a range of lines in a file with new content.

        Args:
            full_path: Path to the file
            start_line: First line to replace (0-indexed)
            end_line: Last line to replace (inclusive)
            new_content: New content to insert
            encoding: File encoding

        Returns:
            Dict containing the number of lines replaced
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                lines = f.readlines()

            line_count = len(lines)

            if start_line < 0:
                start_line = max(0, line_count + start_line)
            if end_line < 0:
                end_line = max(0, line_count + end_line)

            start_line = min(max(0, start_line), line_count)
            end_line = min(max(start_line, end_line), line_count - 1)

            if new_content and not new_content.endswith("\n"):
                new_content += "\n"
            new_lines = new_content.splitlines(True)

            lines[start_line : end_line + 1] = new_lines
            lines_replaced = end_line - start_line + 1

            with path.open("w", encoding=encoding) as f:
                f.writelines(lines)

            logger.log("TOOL", f"Replaced {lines_replaced} lines in {path}")
            return {"lines_replaced": lines_replaced}

        except Exception as e:
            logger.error(f"Error replacing lines in {full_path}: {e}")
            raise


class FindAndReplaceSectionTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="find_and_replace_section",
            description="Find and replace a marked section in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "start_marker": {
                        "type": "string",
                        "description": "Text that marks the beginning of the section.",
                    },
                    "end_marker": {
                        "type": "string",
                        "description": "Text that marks the end of the section.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to replace the section with.",
                    },
                    "include_markers": {
                        "type": "boolean",
                        "description": "Whether to include the markers in the replacement.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "start_marker", "end_marker", "new_content"],
            },
        ),
    )
    name = "find_and_replace_section"

    @classmethod
    async def function(
        cls,
        full_path: str,
        start_marker: str,
        end_marker: str,
        new_content: str,
        include_markers: bool = True,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, bool]:
        """Find and replace a marked section in a file.

        Args:
            full_path: Path to the file
            start_marker: Text that marks the beginning of the section
            end_marker: Text that marks the end of the section
            new_content: New content to replace the section with
            include_markers: Whether to include the markers in the replacement
            encoding: File encoding

        Returns:
            Dict indicating success and whether the section was found
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                content = f.read()

            if start_marker not in content or end_marker not in content:
                logger.log("TOOL", f"Section markers not found in {path}")
                return {"success": False, "section_found": False}

            start_pos = content.find(start_marker)
            end_pos = content.find(end_marker, start_pos + len(start_marker))

            if start_pos == -1 or end_pos == -1:
                logger.log("TOOL", f"Section markers not found in correct order in {path}")
                return {"success": False, "section_found": False}

            if include_markers:
                end_pos += len(end_marker)
                body = new_content

                if body.strip().startswith(start_marker.strip()):
                    body = body.strip()[len(start_marker.strip()) :].lstrip("\n")

                if body.strip().endswith(end_marker.strip()):
                    body = body.strip()[: -len(end_marker.strip())].rstrip("\n")

                # Check if markers are on the same line (no newline between them)
                section_content = content[start_pos : end_pos + len(end_marker)]
                if "\n" not in section_content:
                    # Single line case - no newlines around content
                    replacement = f"{start_marker}{body}{end_marker}"
                else:
                    # Multi-line case - preserve newlines
                    replacement = f"{start_marker}\n{body}\n{end_marker}"
            else:
                end_pos += len(end_marker)
                replacement = new_content

            new_content = content[:start_pos] + replacement + content[end_pos:]

            with path.open("w", encoding=encoding) as f:
                f.write(new_content)

            logger.log("TOOL", f"Replaced section in {path}")
            return {"success": True, "section_found": True}

        except Exception as e:
            logger.error(f"Error replacing section in {full_path}: {e}")
            raise


class AppendToFileTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="append_to_file",
            description="Append content to the end of a file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append.",
                    },
                    "ensure_newline": {
                        "type": "boolean",
                        "description": "Ensure file ends with a newline before appending.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "content"],
            },
        ),
    )
    name = "append_to_file"

    @classmethod
    async def function(
        cls,
        full_path: str,
        content: str,
        ensure_newline: bool = True,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, bool]:
        """Append content to the end of a file.

        Args:
            full_path: Path to the file
            content: Content to append
            ensure_newline: Ensure file ends with a newline before appending
            encoding: File encoding

        Returns:
            Dict indicating success
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            needs_newline = False
            if ensure_newline and os.path.getsize(path) > 0:
                with path.open("rb") as f:
                    f.seek(-1, os.SEEK_END)
                    last_char = f.read(1)
                    needs_newline = last_char != b"\n"

            with path.open("a", encoding=encoding) as f:
                if needs_newline:
                    f.write("\n")
                f.write(content)

            logger.log("TOOL", f"Appended content to {path}")
            return {"success": True}

        except Exception as e:
            logger.error(f"Error appending to {full_path}: {e}")
            raise


class InsertAfterPatternTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="insert_after_pattern",
            description="Insert content after a specific pattern (string or regex) in the file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert after the pattern.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Whether to treat pattern as regex.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of insertions to make.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "pattern", "content"],
            },
        ),
    )
    name = "insert_after_pattern"

    @classmethod
    async def function(
        cls,
        full_path: str,
        pattern: str,
        content: str,
        regex: bool = False,
        count: int = 1,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, int]:
        """Insert content after a specific pattern (string or regex) in the file.

        Args:
            full_path: Path to the file
            pattern: Pattern to search for
            content: Content to insert after the pattern
            regex: Whether to treat pattern as regex
            count: Maximum number of insertions to make
            encoding: File encoding

        Returns:
            Dict containing the number of insertions made
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                file_content = f.read()

            if regex:
                matches = list(re.finditer(pattern, file_content))
                if not matches:
                    logger.log("TOOL", f"Pattern '{pattern}' not found in {path}")
                    return {"insertions": 0}

                matches = matches[:count]  # Limit to the specified count

                new_content = file_content
                for match in reversed(matches):
                    pos = match.end()
                    new_content = new_content[:pos] + content + new_content[pos:]

                insertions = len(matches)
            else:
                insertions = 0
                current_pos = 0
                new_content = file_content

                for _ in range(count):
                    pos = new_content.find(pattern, current_pos)
                    if pos == -1:
                        break

                    insert_pos = pos + len(pattern)
                    new_content = new_content[:insert_pos] + content + new_content[insert_pos:]
                    current_pos = insert_pos + len(content)
                    insertions += 1

            if insertions > 0:
                with path.open("w", encoding=encoding) as f:
                    f.write(new_content)
                logger.log("TOOL", f"Made {insertions} insertions in {path}")

            return {"insertions": insertions}

        except Exception as e:
            logger.error(f"Error inserting after pattern in {full_path}: {e}")
            raise


class InsertBeforePatternTool(ChibiTool):
    register = gpt_settings.filesystem_access
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="insert_before_pattern",
            description="Insert content before a specific pattern (string or regex) in the file.",
            parameters={
                "type": "object",
                "properties": {
                    "full_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert before the pattern.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Whether to treat pattern as regex.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum number of insertions to make.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding.",
                    },
                },
                "required": ["full_path", "pattern", "content"],
            },
        ),
    )
    name = "insert_before_pattern"

    @classmethod
    async def function(
        cls,
        full_path: str,
        pattern: str,
        content: str,
        regex: bool = False,
        count: int = 1,
        encoding: str = "utf-8",
        **kwargs: Unpack[AdditionalOptions],
    ) -> dict[str, int]:
        """Insert content before a specific pattern (string or regex) in the file.

        Args:
            full_path: Path to the file
            pattern: Pattern to search for
            content: Content to insert before the pattern
            regex: Whether to treat pattern as regex
            count: Maximum number of insertions to make
            encoding: File encoding

        Returns:
            Dict containing the number of insertions made
        """
        try:
            path = Path(full_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")

            with path.open("r", encoding=encoding) as f:
                file_content = f.read()

            if regex:
                matches = list(re.finditer(pattern, file_content))
                if not matches:
                    logger.log("TOOL", f"Pattern '{pattern}' not found in {path}")
                    return {"insertions": 0}

                matches = matches[:count]  # Limit to the specified count

                new_content = file_content
                for match in reversed(matches):
                    pos = match.start()
                    new_content = new_content[:pos] + content + new_content[pos:]

                insertions = len(matches)
            else:
                insertions = 0
                current_pos = 0
                new_content = file_content

                for _ in range(count):
                    pos = new_content.find(pattern, current_pos)
                    if pos == -1:
                        break

                    new_content = new_content[:pos] + content + new_content[pos:]
                    current_pos = pos + len(content) + len(pattern)
                    insertions += 1

            if insertions > 0:
                with path.open("w", encoding=encoding) as f:
                    f.write(new_content)
                logger.log("TOOL", f"Made {insertions} insertions in {path}")

            return {"insertions": insertions}

        except Exception as e:
            logger.error(f"Error inserting before pattern in {full_path}: {e}")
            raise
