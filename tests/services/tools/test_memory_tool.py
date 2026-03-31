"""Tests for SearchInConversationHistoryTool."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from chibi.services.providers.tools.memory import SearchInConversationHistoryTool
from chibi.services.providers.tools.exceptions import ToolException


class TestSearchInConversationHistoryTool:
    """Tests for SearchInConversationHistoryTool."""

    @pytest.fixture
    def mock_memory(self):
        mock = MagicMock()
        mock.search = AsyncMock(return_value=[
            {"content": "Test result 1", "metadata": {"role": "user"}},
            {"content": "Test result 2", "metadata": {"role": "assistant"}},
        ])
        return mock

    @pytest.mark.asyncio
    async def test_function_returns_results(self, mock_memory):
        """Test that search returns results."""
        with patch('chibi.services.providers.tools.memory.memory', mock_memory):
            result = await SearchInConversationHistoryTool.function(
                query="test query",
                limit=5,
                user_id=123
            )

            assert "results" in result
            assert len(result["results"]) == 2
            assert result["query"] == "test query"
            assert result["count"] == 2
            mock_memory.search.assert_called_once_with(
                user_id=123,
                query="test query",
                n_results=5
            )

    @pytest.mark.asyncio
    async def test_function_uses_default_limit(self, mock_memory):
        """Test that default limit is used when not specified."""
        with patch('chibi.services.providers.tools.memory.memory', mock_memory):
            from chibi.config import application_settings
            
            result = await SearchInConversationHistoryTool.function(
                query="test",
                user_id=123
            )

            mock_memory.search.assert_called_once_with(
                user_id=123,
                query="test",
                n_results=application_settings.memory_search_limit
            )

    @pytest.mark.asyncio
    async def test_function_raises_when_no_memory(self):
        """Test that ToolException is raised when memory is not configured."""
        with patch('chibi.services.providers.tools.memory.memory', None):
            with pytest.raises(ToolException) as exc_info:
                await SearchInConversationHistoryTool.function(
                    query="test",
                    user_id=123
                )

            assert "not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_function_raises_when_no_user_id(self, mock_memory):
        """Test that ValueError is raised when user_id is missing."""
        with patch('chibi.services.providers.tools.memory.memory', mock_memory):
            with pytest.raises(ValueError) as exc_info:
                await SearchInConversationHistoryTool.function(
                    query="test"
                    # user_id not provided
                )

            assert "user_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_function_returns_empty_when_no_results(self):
        """Test that empty results are handled correctly."""
        mock_memory = MagicMock()
        mock_memory.search = AsyncMock(return_value=[])

        with patch('chibi.services.providers.tools.memory.memory', mock_memory):
            result = await SearchInConversationHistoryTool.function(
                query="nonexistent",
                user_id=123
            )

            assert result["results"] == []
            assert result["count"] == 0
            assert "No matching conversations found" in result["message"]

    def test_tool_definition(self):
        """Test that tool definition is correct."""
        assert SearchInConversationHistoryTool.name == "search_in_conversation_history"
        assert SearchInConversationHistoryTool.register is False
        
        # Check definition structure
        definition = SearchInConversationHistoryTool.definition
        assert definition["type"] == "function"
        assert "function" in definition
        assert definition["function"]["name"] == "search_in_conversation_history"
        
        # Check parameters
        params = definition["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["required"] == ["query"]