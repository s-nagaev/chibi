import pytest

from chibi.services.providers.tools.file_editor import FindAndReplaceSectionTool


@pytest.mark.asyncio
async def test_find_and_replace_section_empty_markers():
    # Create a dummy file
    with open("test_file.txt", "w") as f:
        f.write("start\ncontent\nend")

    # Attempt to use empty markers
    with pytest.raises(ValueError) as excinfo:
        await FindAndReplaceSectionTool.function(
            full_path="test_file.txt", start_marker="", end_marker="", new_content="new"
        )

    assert "Markers cannot be empty" in str(excinfo.value)
