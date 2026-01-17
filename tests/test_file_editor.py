from pathlib import Path

import pytest

from chibi.services.providers.tools import FindAndReplaceSectionTool
from chibi.services.providers.tools.schemas import ToolResponse

TEST_DIR = Path(__file__).parent / "test_files"


find_and_replace_section = FindAndReplaceSectionTool.tool


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
