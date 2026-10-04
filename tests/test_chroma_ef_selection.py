"""Tests for EF selection hardening in ``create_memory()``.

The silent ``DefaultEmbeddingFunction`` fallback is gone: fastembed
unavailability must disable semantic memory with a loud error instead of
silently switching embedding models (which created EF=default-persisted
collections that later broke archival with chromadb EF-conflict errors).

fastembed availability is simulated via ``sys.modules``:
- a fake module object → import succeeds (TextEmbedding is never constructed,
  the ONNX model loads lazily on first embed);
- ``None`` in ``sys.modules`` → ``from fastembed import TextEmbedding`` raises
  ``ImportError``, exactly as with the package missing.
"""

import inspect
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import chibi.memory.chroma as chroma_module
from chibi.memory.chroma import (
    ExternalChromaLongConversationMemory,
    FastEmbedEmbeddingFunction,
    InternalChromaLongConversationMemory,
    create_memory,
)

PROVIDERS = ["LOCAL", "OPENAI", "GEMINI", "MISTRALAI", "JINA"]
DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture
def fake_fastembed(monkeypatch):
    """Inject a fake ``fastembed`` module so TextEmbedding is never constructed."""
    fake_module = types.ModuleType("fastembed")
    fake_module.TextEmbedding = MagicMock(name="TextEmbedding")
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    return fake_module


@pytest.fixture
def no_fastembed(monkeypatch):
    """Make ``from fastembed import TextEmbedding`` raise ImportError."""
    monkeypatch.setitem(sys.modules, "fastembed", None)


def make_settings(embedding_function: str = "LOCAL", **overrides) -> MagicMock:
    settings = MagicMock()
    settings.is_chroma_configured = True
    settings.chroma_host = ""  # embedded mode
    settings.embedding_function = embedding_function
    settings.embedding_model = None
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


class TestLocalSelection:
    def test_local_with_fastembed_available_builds_fastembed_ef(self, fake_fastembed):
        """LOCAL + fastembed available → memory built around FastEmbedEmbeddingFunction."""
        settings = make_settings()
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as memory_cls,
        ):
            memory_cls.return_value = MagicMock()

            result = create_memory()

        assert result is memory_cls.return_value
        ef = memory_cls.call_args.kwargs["embedding_function"]
        assert isinstance(ef, FastEmbedEmbeddingFunction)
        assert ef._model_name == DEFAULT_LOCAL_MODEL

    def test_local_uses_configured_embedding_model(self, fake_fastembed):
        settings = make_settings(embedding_model="custom/embedding-model")
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as memory_cls,
        ):
            memory_cls.return_value = MagicMock()

            create_memory()

        ef = memory_cls.call_args.kwargs["embedding_function"]
        assert ef._model_name == "custom/embedding-model"

    def test_local_selection_does_not_load_onnx_model(self, fake_fastembed):
        """Task 1's lazy model construction must be unaffected: selecting the EF
        must not construct TextEmbedding (the ONNX model loads on first embed)."""
        settings = make_settings()
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as memory_cls,
        ):
            memory_cls.return_value = MagicMock()

            create_memory()

        fake_fastembed.TextEmbedding.assert_not_called()

    def test_local_without_fastembed_disables_memory_with_loud_error(self, no_fastembed):
        """LOCAL + fastembed missing → memory disabled (None) + loud error log."""
        settings = make_settings()
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.logger") as mock_logger,
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as memory_cls,
        ):
            result = create_memory()

        assert result is None
        memory_cls.assert_not_called()
        mock_logger.error.assert_called_once()
        message = str(mock_logger.error.call_args)
        assert "fastembed" in message
        assert "DISABLED" in message

    def test_local_without_fastembed_does_not_construct_any_memory_class(self, no_fastembed):
        """No fallback construction of any memory backend when fastembed is missing."""
        settings = make_settings(chroma_host="remote-host")  # external mode
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.ExternalChromaLongConversationMemory") as external_cls,
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as internal_cls,
        ):
            result = create_memory()

        assert result is None
        external_cls.assert_not_called()
        internal_cls.assert_not_called()


class TestUnknownSelectionArm:
    def test_unknown_embedding_function_disables_memory_with_loud_error(self):
        """The `_` match arm must refuse loudly instead of silently building a default EF.

        application_settings.embedding_function is a pydantic Literal, so this arm
        is unreachable with a validated config — but it must stay fail-closed.
        """
        settings = make_settings(embedding_function="DEFINITELY_NOT_A_PROVIDER")
        with (
            patch("chibi.memory.chroma.application_settings", settings),
            patch("chibi.memory.chroma.logger") as mock_logger,
            patch("chibi.memory.chroma.InternalChromaLongConversationMemory") as internal_cls,
            patch("chibi.memory.chroma.ExternalChromaLongConversationMemory") as external_cls,
        ):
            result = create_memory()

        assert result is None
        internal_cls.assert_not_called()
        external_cls.assert_not_called()
        mock_logger.error.assert_called_once()
        assert "DEFINITELY_NOT_A_PROVIDER" in str(mock_logger.error.call_args)


class TestNoSilentDefaultEF:
    def test_module_no_longer_imports_default_embedding_function(self):
        """chibi.memory.chroma must not reference DefaultEmbeddingFunction at all."""
        assert not hasattr(chroma_module, "DefaultEmbeddingFunction")

    def test_no_selection_arm_constructs_default_ef(self, fake_fastembed):
        """Regression: for every configured provider value, EF selection must
        never construct DefaultEmbeddingFunction and must always produce memory."""
        with (
            patch.object(chroma_module, "GoogleGeminiEmbeddingFunction", MagicMock()),
            patch.object(chroma_module, "OpenAIEmbeddingFunction", MagicMock()),
            patch.object(chroma_module, "MistralEmbeddingFunction", MagicMock()),
            patch.object(chroma_module, "JinaEmbeddingFunction", MagicMock()),
            patch.object(chroma_module, "InternalChromaLongConversationMemory") as memory_cls,
        ):
            memory_cls.return_value = MagicMock()

            for provider in PROVIDERS:
                settings = make_settings(embedding_function=provider)
                with patch("chibi.memory.chroma.application_settings", settings):
                    result = create_memory()
                assert result is memory_cls.return_value, f"provider {provider} must produce memory"

        assert not hasattr(chroma_module, "DefaultEmbeddingFunction")


class TestConstructorsRequireEmbeddingFunction:
    """The memory constructors must have no default EF: a caller that forgets the
    embedding_function fails loudly (TypeError / signature) instead of silently
    embedding with the default ONNX model."""

    @pytest.mark.parametrize(
        "memory_class",
        [InternalChromaLongConversationMemory, ExternalChromaLongConversationMemory],
    )
    def test_embedding_function_parameter_is_required(self, memory_class):
        signature = inspect.signature(memory_class.__init__)
        parameter = signature.parameters["embedding_function"]
        assert parameter.default is inspect.Parameter.empty
