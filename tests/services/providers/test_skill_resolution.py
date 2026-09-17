"""Tests for the skill resolution layer in chibi.utils.app."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from chibi.services.providers.tools.memory import LoadSkillTool
from chibi.services.providers.tools.tool import RegisteredChibiTools
from chibi.utils.app import (
    PACKAGE_SKILLS_DIR,
    get_available_skills,
    get_package_skills,
    get_skill_path,
    get_user_skills,
)

STABLE_PACKAGE_SKILLS = (
    "pm_workflow_skill.md",
    "executor_workflow_skill.md",
    "reviewer_workflow_skill.md",
)


class TestGetPackageSkills:
    """Package skills are discovered from the real chibi package directory."""

    def test_package_skills_discovered_from_real_dir(self) -> None:
        skills = get_package_skills()

        assert len(skills) >= 10
        for name in STABLE_PACKAGE_SKILLS:
            assert name in skills
            assert skills[name]  # non-empty description

    def test_package_skills_descriptions_from_first_heading(self) -> None:
        skills = get_package_skills()

        first_line = (PACKAGE_SKILLS_DIR / "pm_workflow_skill.md").read_text(encoding="utf-8").splitlines()[0]
        assert skills["pm_workflow_skill.md"] == first_line.lstrip("# ").strip()


class TestGetUserSkills:
    """User skills come from the optional settings skills_dir."""

    def test_user_skills_discovered_from_settings_dir(self, tmp_path: Path) -> None:
        (tmp_path / "my_skill.md").write_text("# My Skill\n", encoding="utf-8")
        (tmp_path / "another_skill.md").write_text("# Another Skill\n", encoding="utf-8")

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            skills = get_user_skills()

        assert set(skills) == {"my_skill.md", "another_skill.md"}
        assert skills["my_skill.md"] == "My Skill"

    def test_missing_user_dir_yields_empty_dict_no_exception(self, tmp_path: Path) -> None:
        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path / "does_not_exist")
            skills = get_user_skills()

        assert skills == {}

    def test_hidden_and_non_file_entries_ignored(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden.md").write_text("# Hidden\n", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "real.md").write_text("# Real\n", encoding="utf-8")

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            skills = get_user_skills()

        assert set(skills) == {"real.md"}


class TestGetAvailableSkills:
    """Merged view: package + user, user wins on collision."""

    def test_no_collision_merged_is_package_plus_user(self, tmp_path: Path) -> None:
        (tmp_path / "user_only_skill.md").write_text("# User Only\n", encoding="utf-8")

        package_skills = get_package_skills()
        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            merged = get_available_skills()

        assert merged["user_only_skill.md"] == "User Only"
        for name in STABLE_PACKAGE_SKILLS:
            assert merged[name] == package_skills[name]
        assert set(merged) == set(package_skills) | {"user_only_skill.md"}

    def test_user_wins_on_name_collision(self, tmp_path: Path) -> None:
        collision_name = "pm_workflow_skill.md"
        (tmp_path / collision_name).write_text("# User Override\n", encoding="utf-8")

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            merged = get_available_skills()
            resolved = get_skill_path(collision_name)

        assert merged.get(collision_name) == "User Override"
        assert list(merged).count(collision_name) == 1
        assert resolved == tmp_path / collision_name

    def test_missing_user_dir_still_yields_package_skills(self, tmp_path: Path) -> None:
        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path / "does_not_exist")
            merged = get_available_skills()

        for name in STABLE_PACKAGE_SKILLS:
            assert name in merged


class TestGetSkillPath:
    """Filename resolution with user precedence."""

    def test_nonexistent_name_returns_none(self, tmp_path: Path) -> None:
        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            assert get_skill_path("no_such_skill.md") is None

    def test_package_skill_resolved_by_name(self, tmp_path: Path) -> None:
        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            resolved = get_skill_path("pm_workflow_skill.md")
        assert resolved == PACKAGE_SKILLS_DIR / "pm_workflow_skill.md"

    def test_user_skill_preferred_over_package(self, tmp_path: Path) -> None:
        name = "suno_skill.md"  # exists in the package
        user_content = "# User Suno Override\n"
        (tmp_path / name).write_text(user_content, encoding="utf-8")

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            resolved = get_skill_path(name)

        assert resolved == tmp_path / name
        assert resolved is not None
        assert resolved.read_text(encoding="utf-8") == user_content


class TestLoadSkillTool:
    """LoadSkillTool registration and loading via the resolution layer."""

    def test_registered_under_tool_name_load_skill(self) -> None:
        assert RegisteredChibiTools.tools_map.get("load_skill") is LoadSkillTool
        assert LoadSkillTool.name == "load_skill"
        assert LoadSkillTool.definition["function"]["name"] == "load_skill"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_loads_package_skill_by_name(self, tmp_path: Path) -> None:
        expected_payload = (PACKAGE_SKILLS_DIR / "pm_workflow_skill.md").read_text(encoding="utf-8")
        mock_activate = AsyncMock()

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            with patch("chibi.services.providers.tools.memory.activate_llm_skill", new=mock_activate):
                result = await LoadSkillTool.function(skill_name="pm_workflow_skill.md", user_id=1)
        assert result["status"] == "ok"
        mock_activate.assert_called_once_with(
            user_id=1,
            skill_name="pm_workflow_skill.md",
            skill_payload=expected_payload,
        )

    @pytest.mark.asyncio
    async def test_loads_user_skill_by_name(self, tmp_path: Path) -> None:
        skill_content = "# My Custom Skill\nBody with UTF-8: 你好 🎉"
        (tmp_path / "my_custom_skill.md").write_text(skill_content, encoding="utf-8")
        mock_activate = AsyncMock()

        with patch("chibi.utils.app.application_settings") as mock_settings:
            mock_settings.skills_dir = str(tmp_path)
            with patch("chibi.services.providers.tools.memory.activate_llm_skill", new=mock_activate):
                result = await LoadSkillTool.function(skill_name="my_custom_skill.md", user_id=7)

        assert result["status"] == "ok"
        mock_activate.assert_called_once_with(
            user_id=7,
            skill_name="my_custom_skill.md",
            skill_payload=skill_content,
        )
