"""Tests for retention cleanup job."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.services.jobs.archive import perform_retention_cleanup


class TestPerformRetentionCleanup:
    """Tests for perform_retention_cleanup function."""

    @pytest.mark.asyncio
    async def test_does_nothing_when_memory_is_none(self):
        """Test that function returns early when memory is not configured."""
        with patch("chibi.services.jobs.archive.memory", None):
            with patch("chibi.services.jobs.archive.logger") as mock_logger:
                # Should not raise
                await perform_retention_cleanup()

                mock_logger.info.assert_called()
                assert "not configured" in mock_logger.info.call_args[0][0]

    @pytest.mark.asyncio
    async def test_calls_delete_old_with_retention_days(self):
        """Test that delete_old is called with correct retention days."""
        mock_memory = MagicMock()
        mock_memory.delete_old = AsyncMock()

        with patch("chibi.services.jobs.archive.memory", mock_memory):
            with patch("chibi.services.jobs.archive.application_settings") as mock_settings:
                mock_settings.chroma_history_retention_days = 90

                await perform_retention_cleanup()

                mock_memory.delete_old.assert_called_once_with(90)

    @pytest.mark.asyncio
    async def test_logs_start_and_completion(self):
        """Test that function logs start and completion."""
        mock_memory = MagicMock()
        mock_memory.delete_old = AsyncMock()

        with patch("chibi.services.jobs.archive.memory", mock_memory):
            with patch("chibi.services.jobs.archive.application_settings") as mock_settings:
                mock_settings.chroma_history_retention_days = 30

                with patch("chibi.services.jobs.archive.logger") as mock_logger:
                    await perform_retention_cleanup()

                    # Check for start log
                    start_calls = [call for call in mock_logger.info.call_args_list]
                    assert any("Starting retention cleanup" in str(call) for call in start_calls)
                    assert any("30 days" in str(call) for call in start_calls)

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Test that exceptions are caught and logged."""
        mock_memory = MagicMock()
        mock_memory.delete_old = AsyncMock(side_effect=Exception("Database error"))

        with patch("chibi.services.jobs.archive.memory", mock_memory):
            with patch("chibi.services.jobs.archive.application_settings") as mock_settings:
                mock_settings.chroma_history_retention_days = 90

                with patch("chibi.services.jobs.archive.logger") as mock_logger:
                    # Should not raise
                    await perform_retention_cleanup()

                    # Check error was logged
                    mock_logger.error.assert_called()
                    assert "failed" in mock_logger.error.call_args[0][0].lower()
