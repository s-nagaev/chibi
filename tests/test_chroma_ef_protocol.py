"""Tests for the chromadb embedding-function protocol on FastEmbedEmbeddingFunction.

fastembed is stubbed (no model download): FastEmbedEmbeddingFunction imports
``TextEmbedding`` lazily inside ``__init__``, so injecting a fake module into
sys.modules is enough to instantiate the class cheaply.
"""

import sys
import types
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
from chromadb.utils.embedding_functions import config_to_embedding_function, known_embedding_functions

from chibi.memory.chroma import FASTEMBED_DEFAULT_MODEL, FastEmbedEmbeddingFunction

TEST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="function")
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``fastembed`` module so TextEmbedding is never constructed."""
    fake_module = types.ModuleType("fastembed")
    setattr(fake_module, "TextEmbedding", MagicMock(name="TextEmbedding"))
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    return fake_module


class TestFastEmbedProtocol:
    """FastEmbedEmbeddingFunction must satisfy chromadb's known-EF protocol."""

    def test_name_is_static_and_correct(self) -> None:
        """name() must be callable on the class itself (chromadb calls cls.name())."""
        assert FastEmbedEmbeddingFunction.name() == "fastembed"
        assert "fastembed" not in ("legacy", "default")

    def test_registered_with_chromadb(self) -> None:
        """The class must be registered in chromadb's known EF registry."""
        assert known_embedding_functions.get("fastembed") is FastEmbedEmbeddingFunction

    def test_get_config_shape(self, fake_fastembed: types.ModuleType) -> None:
        """get_config() must return exactly the configured model name."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        config = ef.get_config()
        assert config == {"model_name": TEST_MODEL}

    def test_build_from_config_roundtrip(self, fake_fastembed: types.ModuleType) -> None:
        """build_from_config(get_config()) must rebuild an equivalent EF."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        rebuilt = FastEmbedEmbeddingFunction.build_from_config(ef.get_config())
        assert isinstance(rebuilt, FastEmbedEmbeddingFunction)
        assert rebuilt._model_name == TEST_MODEL

    def test_is_legacy_false(self, fake_fastembed: types.ModuleType) -> None:
        """chromadb's is_legacy() round-trips build_from_config(get_config())."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        assert ef.is_legacy() is False

    def test_config_to_embedding_function_roundtrip(self, fake_fastembed: types.ModuleType) -> None:
        """The persisted shape used by chromadb must rebuild the EF."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        persisted = {"name": ef.name(), "type": "known", "config": ef.get_config()}
        rebuilt = config_to_embedding_function(persisted)
        assert isinstance(rebuilt, FastEmbedEmbeddingFunction)
        assert rebuilt.get_config() == {"model_name": TEST_MODEL}

    def test_call_delegates_to_fastembed(self, fake_fastembed: types.ModuleType) -> None:
        """__call__ must feed input to fastembed; chromadb's base class wrapper
        (normalize_embeddings) converts the output to validated embeddings."""
        model_instance = fake_fastembed.TextEmbedding.return_value
        model_instance.embed.return_value = iter([np.array([1.0, 2.0]), np.array([3.0, 4.0])])

        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        result = ef.__call__(["hello", "world"])

        model_instance.embed.assert_called_once_with(["hello", "world"])
        assert [np.asarray(vec).tolist() for vec in result] == [[1.0, 2.0], [3.0, 4.0]]

    def test_build_from_config_is_cheap_no_model_load(self, fake_fastembed: types.ModuleType) -> None:
        """chromadb's is_legacy() calls build_from_config(get_config()) on EVERY
        get_or_create_collection — constructing the EF must NOT load the ONNX
        model (that happens lazily on first embed)."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        fake_fastembed.TextEmbedding.assert_not_called()

        rebuilt = FastEmbedEmbeddingFunction.build_from_config(ef.get_config())

        assert isinstance(rebuilt, FastEmbedEmbeddingFunction)
        fake_fastembed.TextEmbedding.assert_not_called()

    def test_model_constructed_lazily_on_first_call(self, fake_fastembed: types.ModuleType) -> None:
        """The ONNX model must load only when the EF is actually used."""
        model_instance = fake_fastembed.TextEmbedding.return_value
        model_instance.embed.side_effect = lambda docs: iter([np.array([1.0, 2.0])] * len(docs))

        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        fake_fastembed.TextEmbedding.assert_not_called()

        ef.__call__(["hello"])
        fake_fastembed.TextEmbedding.assert_called_once_with(model_name=TEST_MODEL)

        # Second call must reuse the already-loaded model
        ef.__call__(["world"])
        fake_fastembed.TextEmbedding.assert_called_once()


class TestDefaultModel:
    """The class default must be the intentional multilingual MiniLM model."""

    def test_default_model_is_multilingual_minilm(self, fake_fastembed: types.ModuleType) -> None:
        """Constructing without model_name must select the multilingual MiniLM
        constant (not the English-only chromadb/bge default)."""
        ef = FastEmbedEmbeddingFunction()
        assert ef._model_name == FASTEMBED_DEFAULT_MODEL
        assert ef.get_config() == {"model_name": FASTEMBED_DEFAULT_MODEL}

    def test_default_model_roundtrip(self, fake_fastembed: types.ModuleType) -> None:
        """build_from_config(get_config()) of a default-constructed EF must
        preserve the default model name."""
        ef = FastEmbedEmbeddingFunction()
        rebuilt = FastEmbedEmbeddingFunction.build_from_config(ef.get_config())
        assert isinstance(rebuilt, FastEmbedEmbeddingFunction)
        assert rebuilt._model_name == FASTEMBED_DEFAULT_MODEL


class TestMeanPoolingWarningSuppression:
    """_get_model must swallow only fastembed's mean-pooling behavior notice."""

    def _ef_with_warning_emitting_model(
        self,
        ef: FastEmbedEmbeddingFunction,
        fake_fastembed: types.ModuleType,
        message: str,
    ) -> None:
        """Swap in a mocked TextEmbedding class that warns on construction."""

        def _warn_and_return(*args: object, **kwargs: object) -> object:
            warnings.warn(message, UserWarning, stacklevel=2)
            return fake_fastembed.TextEmbedding.return_value

        emitting_cls = MagicMock(name="TextEmbedding", side_effect=_warn_and_return)
        ef._text_embedding_cls = emitting_cls

    def test_mean_pooling_notice_is_suppressed(self, fake_fastembed: types.ModuleType) -> None:
        """The exact fastembed 0.8.0 mean-pooling UserWarning must not escape
        _get_model (safe: fastembed is pinned, the store is homogeneous)."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        self._ef_with_warning_emitting_model(
            ef,
            fake_fastembed,
            "The model 'BAAI/bge-small-en-v1.5' now uses mean pooling instead of CLS embedding. "
            "Recreating the model might change the results.",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ef._get_model()

    def test_other_user_warnings_still_propagate(self, fake_fastembed: types.ModuleType) -> None:
        """The narrow filter must not swallow unrelated UserWarnings."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        self._ef_with_warning_emitting_model(ef, fake_fastembed, "some unrelated deprecation notice")
        with pytest.raises(UserWarning, match="unrelated deprecation notice"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ef._get_model()
