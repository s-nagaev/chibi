from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chibi.services.providers.tools import (
    FindAndReplaceSectionTool,
    InsertAfterPatternTool,
    InsertAtLineTool,
    InsertBeforePatternTool,
    ReadFileTool,
    ReplaceInFileRegexTool,
    ReplaceInFileTool,
)
from chibi.services.providers.tools.schemas import ToolResponse

TEST_DIR = Path(__file__).parent / "test_files"


find_and_replace_section = FindAndReplaceSectionTool.tool
insert_at_line = InsertAtLineTool.tool
insert_before_pattern = InsertBeforePatternTool.tool
insert_after_pattern = InsertAfterPatternTool.tool
replace_in_file = ReplaceInFileTool.tool
replace_in_file_regex = ReplaceInFileRegexTool.tool


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Create a directory for test files
    TEST_DIR.mkdir(exist_ok=True)
    yield
    # Teardown: Clean up test files and directory
    for f in TEST_DIR.iterdir():
        f.unlink()
    TEST_DIR.rmdir()


async def create_test_file(filename, content):
    file_path = TEST_DIR / filename
    with open(file_path, "w") as f:
        f.write(content)
    return str(file_path)


async def read_test_file(filename):
    file_path = TEST_DIR / filename
    with open(file_path, "r") as f:
        return f.read()


# ==============================================================================
# InsertAtLineTool Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_insert_at_line_empty_content():
    """Test that inserting empty content at a line works correctly."""
    content = "line 1\nline 2\nline 3"
    file_path = await create_test_file("test_empty_content.txt", content)

    result: ToolResponse = await insert_at_line(
        full_path=file_path,
        line_number=1,
        content="",
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    # Empty string should insert an empty line at position 1
    updated_content = await read_test_file("test_empty_content.txt")
    lines = updated_content.split("\n")
    assert lines[0] == "line 1"
    assert lines[1] == ""  # Empty line inserted
    assert lines[2] == "line 2"


@pytest.mark.asyncio
async def test_insert_at_line_content_without_newline():
    """Test that content without trailing newline gets newline added."""
    content = "line 1\nline 2"
    file_path = await create_test_file("test_no_newline.txt", content)

    result: ToolResponse = await insert_at_line(
        full_path=file_path,
        line_number=1,
        content="inserted",
    )
    assert result.status == "ok"

    updated_content = await read_test_file("test_no_newline.txt")
    assert "inserted\n" in updated_content


@pytest.mark.asyncio
async def test_insert_at_line_content_with_newline():
    """Test that content with trailing newline is inserted as-is."""
    content = "line 1\nline 2"
    file_path = await create_test_file("test_with_newline.txt", content)

    result: ToolResponse = await insert_at_line(
        full_path=file_path,
        line_number=1,
        content="inserted\n",
    )
    assert result.status == "ok"

    updated_content = await read_test_file("test_with_newline.txt")
    assert "inserted\n" in updated_content


# ==============================================================================
# FindAndReplaceSectionTool Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_replace_section_include_markers():
    content = "start\nSECTION_START\ncontent to replace\nSECTION_END\nend"
    file_path = await create_test_file("test_include.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker=start_marker,
        end_marker=end_marker,
        new_content=new_content,
        include_markers=True,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated_content = await read_test_file("test_include.txt")
    assert updated_content == "start\nSECTION_START\nnew content\nSECTION_END\nend"


@pytest.mark.asyncio
async def test_replace_section_exclude_markers():
    content = "start\nSECTION_START\ncontent to replace\nSECTION_END\nend"
    file_path = await create_test_file("test_exclude.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker=start_marker,
        end_marker=end_marker,
        new_content=new_content,
        include_markers=False,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated_content = await read_test_file("test_exclude.txt")
    assert updated_content == "start\nnew content\nend"


@pytest.mark.asyncio
async def test_replace_section_markers_at_boundaries_exclude():
    content = "SECTION_START\ncontent to replace\nSECTION_END"
    file_path = await create_test_file("test_boundaries_exclude.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker=start_marker,
        end_marker=end_marker,
        new_content=new_content,
        include_markers=False,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated_content = await read_test_file("test_boundaries_exclude.txt")
    assert updated_content == "new content"


@pytest.mark.asyncio
async def test_markers_not_found():
    content = "some content\nwithout markers\nin this file"
    file_path = await create_test_file("test_no_markers.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=start_marker, end_marker=end_marker, new_content=new_content
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_no_markers.txt")
    assert updated_content == content  # Content should remain unchanged


@pytest.mark.asyncio
async def test_only_start_marker_found():
    content = "start\nSECTION_START\ncontent to replace\nend"
    file_path = await create_test_file("test_only_start.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=start_marker, end_marker=end_marker, new_content=new_content
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_only_start.txt")
    assert updated_content == content  # Content should remain unchanged


@pytest.mark.asyncio
async def test_only_end_marker_found_after_start():
    content = "start\ncontent to replace\nSECTION_END\nend"
    await create_test_file("test_only_end_after.txt", content)
    new_content = "new content"
    end_marker = "SECTION_END"

    # Need a start marker for the end marker to be searched *after* it
    content_with_start = "start\nSECTION_START\ncontent to replace\nSECTION_END\nend"
    file_path_with_start = await create_test_file("test_only_end_after_with_start.txt", content_with_start)

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path_with_start,
        start_marker="NON_EXISTENT_START",
        end_marker=end_marker,
        new_content=new_content,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_only_end_after_with_start.txt")
    assert updated_content == content_with_start  # Content should remain unchanged


@pytest.mark.asyncio
async def test_end_marker_before_start_marker():
    content = "start\nSECTION_END\ncontent to replace\nSECTION_START\nend"
    file_path = await create_test_file("test_end_before_start.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=start_marker, end_marker=end_marker, new_content=new_content
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_end_before_start.txt")
    assert updated_content == content  # Content should remain unchanged


@pytest.mark.asyncio
async def test_empty_file():
    content = ""
    file_path = await create_test_file("test_empty.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=start_marker, end_marker=end_marker, new_content=new_content
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_empty.txt")
    assert updated_content == content  # Content should remain unchanged


@pytest.mark.asyncio
async def test_no_markers_in_file():
    content = "line 1\nline 2\nline 3"
    file_path = await create_test_file("test_no_markers_content.txt", content)
    new_content = "new content"
    start_marker = "START"
    end_marker = "END"

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=start_marker, end_marker=end_marker, new_content=new_content
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("section_found") is False

    updated_content = await read_test_file("test_no_markers_content.txt")
    assert updated_content == content  # Content should remain unchanged


@pytest.mark.asyncio
async def test_markers_in_single_line():
    content = "before SECTION_START content to replace SECTION_END after"
    file_path = await create_test_file("test_single_line.txt", content)
    new_content = "new content"
    start_marker = "SECTION_START"
    end_marker = "SECTION_END"

    result_include: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker=start_marker,
        end_marker=end_marker,
        new_content=new_content,
        include_markers=True,
    )
    assert result_include.status == "ok"
    assert isinstance(result_include.result, dict)
    assert result_include.result.get("success") is True
    updated_content_include = await read_test_file("test_single_line.txt")
    assert updated_content_include == "before SECTION_STARTnew contentSECTION_END after"

    # Reset file for exclude test
    await create_test_file("test_single_line.txt", content)
    result_exclude: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker=start_marker,
        end_marker=end_marker,
        new_content=new_content,
        include_markers=False,
    )
    assert result_exclude.status == "ok"
    assert isinstance(result_exclude.result, dict)
    assert result_exclude.result.get("success") is True
    updated_content_exclude = await read_test_file("test_single_line.txt")
    assert updated_content_exclude == "before new content after"


@pytest.mark.asyncio
async def test_markers_are_same():
    content = "before MARKER content to replace MARKER after"
    file_path = await create_test_file("test_same_markers.txt", content)
    new_content = "new content"
    marker = "MARKER"

    result_include: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=marker, end_marker=marker, new_content=new_content, include_markers=True
    )
    assert result_include.status == "ok"
    assert isinstance(result_include.result, dict)
    assert result_include.result.get("success") is True
    updated_content_include = await read_test_file("test_same_markers.txt")
    assert updated_content_include == "before MARKERnew contentMARKER after"

    # Reset file for exclude test
    await create_test_file("test_same_markers.txt", content)
    result_exclude: ToolResponse = await find_and_replace_section(
        full_path=file_path, start_marker=marker, end_marker=marker, new_content=new_content, include_markers=False
    )
    assert result_exclude.status == "ok"
    assert isinstance(result_exclude.result, dict)
    assert result_exclude.result.get("success") is True
    updated_content_exclude = await read_test_file("test_same_markers.txt")
    assert updated_content_exclude == "before new content after"


@pytest.mark.asyncio
async def test_single_line_section_with_trailing_newline():
    """Regression: newline after end_marker must not trick single-line detection."""
    content = "before START_MARKERold contentEND_MARKER\nafter"
    file_path = await create_test_file("test_trailing_nl.txt", content)

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker="START_MARKER",
        end_marker="END_MARKER",
        new_content="replaced",
        include_markers=True,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated = await read_test_file("test_trailing_nl.txt")
    # Single-line section: markers + content on one line, no extra \n injected
    assert updated == "before START_MARKERreplacedEND_MARKER\nafter"


@pytest.mark.asyncio
async def test_multiline_section_not_broken_by_fix():
    """Ensure multi-line sections still get newlines around body."""
    content = "header\n<!-- BEGIN -->\nold line 1\nold line 2\n<!-- END -->\nfooter"
    file_path = await create_test_file("test_multiline_ok.txt", content)

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker="<!-- BEGIN -->",
        end_marker="<!-- END -->",
        new_content="new line 1\nnew line 2",
        include_markers=True,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated = await read_test_file("test_multiline_ok.txt")
    assert updated == "header\n<!-- BEGIN -->\nnew line 1\nnew line 2\n<!-- END -->\nfooter"


@pytest.mark.asyncio
async def test_single_line_inline_markers_include():
    """Inline markers on a single line with no newline anywhere in file."""
    content = "prefix [START]value[END] suffix"
    file_path = await create_test_file("test_inline.txt", content)

    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker="[START]",
        end_marker="[END]",
        new_content="new_val",
        include_markers=True,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("success") is True

    updated = await read_test_file("test_inline.txt")
    assert updated == "prefix [START]new_val[END] suffix"


# ==============================================================================
# ReplaceInFileTool Tests


@pytest.mark.asyncio
async def test_find_and_replace_empty_markers_raises_error():
    """Test that empty start/end markers raise ValueError."""
    content = "some content\nwith markers\nin this file"
    file_path = await create_test_file("test_empty_markers.txt", content)
    new_content = "new content"

    # Test empty start_marker
    result: ToolResponse = await find_and_replace_section(
        full_path=file_path,
        start_marker="",
        end_marker="SECTION_END",
        new_content=new_content,
    )
    assert result.status == "error"
    assert "Markers cannot be empty" in result.result

    # Test empty end_marker
    result = await find_and_replace_section(
        full_path=file_path,
        start_marker="SECTION_START",
        end_marker="",
        new_content=new_content,
    )
    assert result.status == "error"
    assert "Markers cannot be empty" in result.result

    # Test both empty
    result = await find_and_replace_section(
        full_path=file_path,
        start_marker="",
        end_marker="",
        new_content=new_content,
    )
    assert result.status == "error"
    assert "Markers cannot be empty" in result.result


# ==============================================================================


@pytest.mark.asyncio
async def test_replace_in_file_empty_old_text_returns_error():
    """Test that empty old_text returns an error response."""
    content = "some content to replace"
    file_path = await create_test_file("test_empty_old_text.txt", content)

    result: ToolResponse = await replace_in_file(
        full_path=file_path,
        old_text="",
        new_text="new text",
    )
    assert result.status == "error"
    assert "old_text cannot be empty" in result.result


# ==============================================================================
# InsertBeforePatternTool Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_insert_before_pattern_count_minus_one_all_occurrences():
    """Test that count=-1 inserts before ALL occurrences of the pattern."""
    content = "line1\nTARGET\nline2\nTARGET\nline3\nTARGET\n"
    file_path = await create_test_file("test_before_count_all.txt", content)

    result: ToolResponse = await insert_before_pattern(
        full_path=file_path,
        pattern="TARGET",
        content="# INSERTED\n",
        count=-1,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 3

    updated_content = await read_test_file("test_before_count_all.txt")
    assert updated_content.count("# INSERTED") == 3


@pytest.mark.asyncio
async def test_insert_before_pattern_count_specific():
    """Test that count=N inserts before N occurrences."""
    content = "line1\nTARGET\nline2\nTARGET\nline3\nTARGET\n"
    file_path = await create_test_file("test_before_count_specific.txt", content)

    result: ToolResponse = await insert_before_pattern(
        full_path=file_path,
        pattern="TARGET",
        content="# INSERTED\n",
        count=2,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 2

    updated_content = await read_test_file("test_before_count_specific.txt")
    assert updated_content.count("# INSERTED") == 2


@pytest.mark.asyncio
async def test_insert_before_pattern_regex_count_minus_one():
    """Test that count=-1 works with regex mode."""
    content = "line1\nitem1\nitem2\nitem3\n"
    file_path = await create_test_file("test_before_regex_count_all.txt", content)

    result: ToolResponse = await insert_before_pattern(
        full_path=file_path,
        pattern=r"item\d+",
        content="# MATCHED\n",
        regex=True,
        count=-1,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 3


# ==============================================================================
# InsertAfterPatternTool Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_insert_after_pattern_count_minus_one_all_occurrences():
    """Test that count=-1 inserts after ALL occurrences of the pattern."""
    content = "line1\nTARGET\nline2\nTARGET\nline3\nTARGET\n"
    file_path = await create_test_file("test_after_count_all.txt", content)

    result: ToolResponse = await insert_after_pattern(
        full_path=file_path,
        pattern="TARGET",
        content="# INSERTED\n",
        count=-1,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 3

    updated_content = await read_test_file("test_after_count_all.txt")
    assert updated_content.count("# INSERTED") == 3


@pytest.mark.asyncio
async def test_insert_after_pattern_count_specific():
    """Test that count=N inserts after N occurrences."""
    content = "line1\nTARGET\nline2\nTARGET\nline3\nTARGET\n"
    file_path = await create_test_file("test_after_count_specific.txt", content)

    result: ToolResponse = await insert_after_pattern(
        full_path=file_path,
        pattern="TARGET",
        content="# INSERTED\n",
        count=2,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 2

    updated_content = await read_test_file("test_after_count_specific.txt")
    assert updated_content.count("# INSERTED") == 2


@pytest.mark.asyncio
async def test_insert_after_pattern_regex_count_minus_one():
    """Test that count=-1 works with regex mode."""
    content = "line1\nitem1\nitem2\nitem3\n"
    file_path = await create_test_file("test_after_regex_count_all.txt", content)

    result: ToolResponse = await insert_after_pattern(
        full_path=file_path,
        pattern=r"item\d+",
        content="# MATCHED\n",
        regex=True,
        count=-1,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("insertions") == 3


# ==============================================================================
# ReplaceInFileRegexTool Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_replace_in_file_regex_count_minus_one_all_occurrences():
    """Test that count=-1 replaces ALL occurrences of the pattern."""
    content = "foo bar foo baz foo"
    file_path = await create_test_file("test_regex_count_all.txt", content)

    result: ToolResponse = await replace_in_file_regex(
        full_path=file_path,
        pattern="foo",
        replacement="qux",
        count=-1,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("replacements") == 3

    updated_content = await read_test_file("test_regex_count_all.txt")
    assert updated_content == "qux bar qux baz qux"


@pytest.mark.asyncio
async def test_replace_in_file_regex_count_specific():
    """Test that count=N replaces N occurrences."""
    content = "foo bar foo baz foo"
    file_path = await create_test_file("test_regex_count_specific.txt", content)

    result: ToolResponse = await replace_in_file_regex(
        full_path=file_path,
        pattern="foo",
        replacement="qux",
        count=2,
    )
    assert result.status == "ok"
    assert isinstance(result.result, dict) and result.result.get("replacements") == 2

    updated_content = await read_test_file("test_regex_count_specific.txt")
    assert updated_content == "qux bar qux baz foo"


# ==============================================================================
# ReadFileTool Tests
# ==============================================================================


read_file = ReadFileTool.tool


@pytest.mark.asyncio
async def test_read_file_success():
    """Test that reading a file returns its content."""
    content = "Hello, World!\nThis is a test file."
    file_path = await create_test_file("test_read.txt", content)

    mock_moderator = AsyncMock()
    mock_moderator.name = "test_moderator"
    mock_moderator.moderate_command = AsyncMock(return_value=Mock(verdict="approved"))

    with patch("chibi.services.providers.tools.file_editor.get_moderation_provider", return_value=mock_moderator):
        result: ToolResponse = await read_file(full_path=file_path, user_id=123)

    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("content") == content
    assert result.result.get("full_path") == file_path


@pytest.mark.asyncio
async def test_read_file_not_found():
    """Test that reading a non-existent file raises an error."""
    file_path = str(TEST_DIR / "nonexistent.txt")

    mock_moderator = AsyncMock()
    mock_moderator.name = "test_moderator"
    mock_moderator.moderate_command = AsyncMock(return_value=Mock(verdict="approved"))

    with patch("chibi.services.providers.tools.file_editor.get_moderation_provider", return_value=mock_moderator):
        result: ToolResponse = await read_file(full_path=file_path, user_id=123)

    assert result.status == "error"
    assert "does not exist" in str(result.result).lower()


@pytest.mark.asyncio
async def test_read_file_directory_raises_error():
    """Test that reading a directory raises an error."""
    mock_moderator = AsyncMock()
    mock_moderator.name = "test_moderator"
    mock_moderator.moderate_command = AsyncMock(return_value=Mock(verdict="approved"))

    with patch("chibi.services.providers.tools.file_editor.get_moderation_provider", return_value=mock_moderator):
        result: ToolResponse = await read_file(full_path=str(TEST_DIR), user_id=123)

    assert result.status == "error"
    assert "not a file" in str(result.result).lower()


@pytest.mark.asyncio
async def test_read_file_with_encoding():
    """Test that reading a file with specific encoding works."""
    content = "Hello with encoding"
    file_path = await create_test_file("test_encoding.txt", content)

    mock_moderator = AsyncMock()
    mock_moderator.name = "test_moderator"
    mock_moderator.moderate_command = AsyncMock(return_value=Mock(verdict="approved"))

    with patch("chibi.services.providers.tools.file_editor.get_moderation_provider", return_value=mock_moderator):
        result: ToolResponse = await read_file(full_path=file_path, encoding="utf-8", user_id=123)

    assert result.status == "ok"
    assert isinstance(result.result, dict)
    assert result.result.get("content") == content


@pytest.mark.asyncio
async def test_read_file_moderation_declined():
    """Test that reading a file is blocked when moderator declines."""
    content = "Secret content"
    file_path = await create_test_file("test_moderation_declined.txt", content)

    mock_moderator = AsyncMock()
    mock_moderator.name = "test_moderator"
    mock_moderator.moderate_command = AsyncMock(return_value=Mock(verdict="declined", reason="Not allowed"))

    with patch("chibi.services.providers.tools.file_editor.get_moderation_provider", return_value=mock_moderator):
        result: ToolResponse = await read_file(full_path=file_path, user_id=123)

    assert result.status == "error"
    assert "declined" in str(result.result).lower()
