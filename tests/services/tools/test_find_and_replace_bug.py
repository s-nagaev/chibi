import pytest

from chibi.services.providers.tools.file_editor import FindAndReplaceSectionTool


@pytest.mark.asyncio
async def test_find_and_replace_section_empty_markers(tmp_path):
    # Create a dummy file in a temporary directory (avoid littering the repo root)
    scratch_file = tmp_path / "test_file.txt"
    scratch_file.write_text("start\ncontent\nend", encoding="utf-8")

    # Attempt to use empty markers
    with pytest.raises(ValueError) as excinfo:
        await FindAndReplaceSectionTool.function(
            full_path=str(scratch_file), start_marker="", end_marker="", new_content="new"
        )

    assert "Markers cannot be empty" in str(excinfo.value)
