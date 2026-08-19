"""Tests for memory tool encoding behavior."""

import builtins
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.services.providers.tools.memory import LoadBuiltinSkillTool


class TestLoadBuiltinSkillTool:
    """Tests for LoadBuiltinSkillTool file encoding."""

    @pytest.mark.asyncio
    async def test_loads_skill_as_utf8(self, tmp_path: Path, monkeypatch):
        """Skill files must be opened with encoding='utf-8'."""
        skill_name = "test_skill.md"
        skill_content = "Skill content with UTF-8: 你好, мир, 🎉"
        skill_path = tmp_path / skill_name
        skill_path.write_text(skill_content, encoding="utf-8")

        original_open = builtins.open
        captured_encoding: str | None = None

        def spy_open(*args, **kwargs):
            nonlocal captured_encoding
            if args and isinstance(args[0], str) and args[0].endswith(skill_name):
                positional_encoding = args[3] if len(args) > 3 else None
                captured_encoding = kwargs.get("encoding", positional_encoding)
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)

        mock_activate = AsyncMock()
        with patch.object(
            LoadBuiltinSkillTool,
            "get_interface",
            return_value=MagicMock(thread_id=0),
        ):
            with patch("chibi.services.providers.tools.memory.application_settings") as mock_settings:
                mock_settings.skills_dir = str(tmp_path)
                with patch("chibi.services.providers.tools.memory.activate_llm_skill", new=mock_activate):
                    result = await LoadBuiltinSkillTool.function(
                        skill_name=skill_name,
                        user_id=1,
                    )

        assert result["status"] == "ok"
        assert captured_encoding == "utf-8"
        mock_activate.assert_called_once_with(
            user_id=1,
            skill_name=skill_name,
            skill_payload=skill_content,
        )
