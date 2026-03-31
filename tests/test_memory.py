"""Tests for memory module and factory."""

from unittest.mock import MagicMock, patch

from chibi.memory.chroma import create_memory


class TestCreateMemory:
    """Tests for create_memory() factory function."""

    def test_create_memory_returns_none_when_not_configured(self):
        """Test that create_memory returns None when ChromaDB is not configured."""
        with patch("chibi.memory.chroma.application_settings") as mock_settings:
            mock_settings.is_chroma_configured = False

            result = create_memory()

            assert result is None

    def test_create_memory_returns_instance_when_configured(self):
        """Test that create_memory returns instance when ChromaDB is configured."""
        with patch("chibi.memory.chroma.application_settings") as mock_settings:
            mock_settings.is_chroma_configured = True

            with patch("chibi.memory.chroma.ChromaLongConversationMemory") as mock_chroma_class:
                mock_instance = MagicMock()
                mock_chroma_class.return_value = mock_instance

                result = create_memory()

                mock_chroma_class.assert_called_once()
                assert result is mock_instance

    def test_create_memory_returns_none_on_exception(self):
        """Test that create_memory returns None when ChromaDB initialization fails."""
        with patch("chibi.memory.chroma.application_settings") as mock_settings:
            mock_settings.is_chroma_configured = True

            with patch("chibi.memory.chroma.ChromaLongConversationMemory", side_effect=Exception("DB error")):
                with patch("chibi.memory.chroma.logger") as mock_logger:
                    result = create_memory()

                    assert result is None
                    mock_logger.error.assert_called_once()

    def test_create_memory_import(self):
        """Test that create_memory can be imported."""
        from chibi.memory.chroma import create_memory

        assert callable(create_memory)