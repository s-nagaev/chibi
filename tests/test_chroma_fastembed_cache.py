"""Tests for the persistent fastembed cache directory.

The cache directory is resolved by ``FastEmbedEmbeddingFunction`` and handed to
fastembed's ``TextEmbedding`` at model-construction time. fastembed is stubbed
here (no model download, no ONNX session, no real cache directory write): the
fake module records the kwargs the real ``TextEmbedding`` constructor would
receive, so the resolution logic is fully observable offline.
"""

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from chibi.memory.chroma import (
    FASTEMBED_CACHE_DIR_DEFAULT,
    FASTEMBED_CACHE_PATH_ENV_VAR,
    FASTEMBED_DEFAULT_MODEL,
    FastEmbedEmbeddingFunction,
)

TEST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="function")
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``fastembed`` module recording TextEmbedding kwargs.

    Args:
        monkeypatch: pytest fixture used to install the fake module.

    Returns:
        The injected fake ``fastembed`` module.
    """
    fake_module = types.ModuleType("fastembed")
    setattr(fake_module, "TextEmbedding", MagicMock(name="TextEmbedding"))
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    monkeypatch.delenv(FASTEMBED_CACHE_PATH_ENV_VAR, raising=False)
    return fake_module


def _recorded_cache_dir(fake_fastembed: types.ModuleType) -> str:
    """Return the ``cache_dir`` kwarg recorded by the fake TextEmbedding class.

    Args:
        fake_fastembed: The injected fake ``fastembed`` module.

    Returns:
        The recorded ``cache_dir`` keyword argument.
    """
    kwargs = fake_fastembed.TextEmbedding.call_args.kwargs
    return str(kwargs["cache_dir"])


class TestFastEmbedCacheDirResolution:
    """The cache directory must be persistent by default and env-overridable."""

    def test_default_cache_dir_is_persistent_not_temp(self, fake_fastembed: types.ModuleType) -> None:
        """Without FASTEMBED_CACHE_PATH the model must be cached outside $TMPDIR.

        The OS wipes the temp directory periodically, which destroys partially
        downloaded model snapshots and breaks semantic memory.
        """
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        ef._get_model()

        cache_dir = _recorded_cache_dir(fake_fastembed)
        assert cache_dir
        assert cache_dir == str(FASTEMBED_CACHE_DIR_DEFAULT)
        assert not Path(cache_dir).is_relative_to(Path(tempfile.gettempdir()))

    def test_default_cache_dir_used_for_default_model(self, fake_fastembed: types.ModuleType) -> None:
        """The default (multilingual MiniLM) model must use the persistent cache too."""
        ef = FastEmbedEmbeddingFunction()
        assert ef._model_name == FASTEMBED_DEFAULT_MODEL

        ef._get_model()

        assert _recorded_cache_dir(fake_fastembed) == str(FASTEMBED_CACHE_DIR_DEFAULT)

    def test_env_var_overrides_default(
        self, fake_fastembed: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FASTEMBED_CACHE_PATH must win over the built-in default directory."""
        custom_cache_dir = tmp_path / "custom-fastembed-cache"
        monkeypatch.setenv(FASTEMBED_CACHE_PATH_ENV_VAR, str(custom_cache_dir))

        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        ef._get_model()

        assert _recorded_cache_dir(fake_fastembed) == str(custom_cache_dir)

    def test_empty_env_var_falls_back_to_default(
        self, fake_fastembed: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty FASTEMBED_CACHE_PATH must not be passed through as a cache dir."""
        monkeypatch.setenv(FASTEMBED_CACHE_PATH_ENV_VAR, "")

        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)
        ef._get_model()

        assert _recorded_cache_dir(fake_fastembed) == str(FASTEMBED_CACHE_DIR_DEFAULT)


class TestFastEmbedLazyModelConstruction:
    """Resolving the cache dir must not make model construction eager."""

    def test_init_does_not_construct_model(self, fake_fastembed: types.ModuleType) -> None:
        """Constructing the embedding function must not create a model."""
        FastEmbedEmbeddingFunction(model_name=TEST_MODEL)

        fake_fastembed.TextEmbedding.assert_not_called()

    def test_first_get_model_constructs_once_and_reuses(self, fake_fastembed: types.ModuleType) -> None:
        """The model must be constructed on first use and then reused as-is."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)

        first_model = ef._get_model()
        fake_fastembed.TextEmbedding.assert_called_once()

        assert ef._get_model() is first_model
        fake_fastembed.TextEmbedding.assert_called_once()

        assert _recorded_cache_dir(fake_fastembed) == str(FASTEMBED_CACHE_DIR_DEFAULT)

    def test_call_passes_cache_dir_on_first_embed(self, fake_fastembed: types.ModuleType) -> None:
        """A ``__call__`` must pass the persistent cache dir to the model."""
        fake_fastembed.TextEmbedding.return_value.embed.return_value = iter([np.array([1.0, 2.0])])
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)

        assert [np.asarray(vec).tolist() for vec in ef(["hello"])] == [[1.0, 2.0]]

        fake_fastembed.TextEmbedding.assert_called_once()
        assert _recorded_cache_dir(fake_fastembed) == str(FASTEMBED_CACHE_DIR_DEFAULT)


class TestFastEmbedConfigUnchanged:
    """The persisted embedding-function identity must not include the cache dir."""

    def test_get_config_shape_is_model_name_only(self, fake_fastembed: types.ModuleType) -> None:
        """get_config() must return exactly ``{"model_name": ...}``."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)

        assert ef.get_config() == {"model_name": TEST_MODEL}

    def test_build_from_config_roundtrip(self, fake_fastembed: types.ModuleType) -> None:
        """build_from_config(get_config()) must accept the persisted shape."""
        ef = FastEmbedEmbeddingFunction(model_name=TEST_MODEL)

        rebuilt = FastEmbedEmbeddingFunction.build_from_config(ef.get_config())

        assert isinstance(rebuilt, FastEmbedEmbeddingFunction)
        assert rebuilt.get_config() == {"model_name": TEST_MODEL}
