"""Tests for models_whitelist and models_blacklist functionality in GPTSettings."""

from chibi.config.gpt import GPTSettings


class TestGPTSettingsWhitelistParsing:
    """Tests for models_whitelist parsing."""

    def test_whitelist_parsing_empty(self):
        """Test that empty whitelist_raw returns empty list."""
        settings = GPTSettings(MODELS_WHITELIST=None)
        assert settings.models_whitelist == []

    def test_whitelist_parsing_single_value(self):
        """Test parsing whitelist with single model."""
        settings = GPTSettings(MODELS_WHITELIST="gpt-4o")
        assert settings.models_whitelist == ["gpt-4o"]

    def test_whitelist_parsing_multiple_values(self):
        """Test parsing whitelist with multiple models."""
        settings = GPTSettings(MODELS_WHITELIST="gpt-4o, gpt-4o-mini, claude-3-5-sonnet")
        assert settings.models_whitelist == ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]

    def test_whitelist_parsing_strips_whitespace(self):
        """Test that whitespace is stripped from model names."""
        settings = GPTSettings(MODELS_WHITELIST="gpt-4o , gpt-4o-mini ,  claude-3")
        assert settings.models_whitelist == ["gpt-4o", "gpt-4o-mini", "claude-3"]


class TestGPTSettingsBlacklistParsing:
    """Tests for models_blacklist parsing."""

    def test_blacklist_parsing_empty(self):
        """Test that empty blacklist_raw returns empty list."""
        settings = GPTSettings(MODELS_BLACKLIST=None)
        assert settings.models_blacklist == []

    def test_blacklist_parsing_single_value(self):
        """Test parsing blacklist with single model."""
        settings = GPTSettings(MODELS_BLACKLIST="gpt-4-turbo")
        assert settings.models_blacklist == ["gpt-4-turbo"]

    def test_blacklist_parsing_multiple_values(self):
        """Test parsing blacklist with multiple models."""
        settings = GPTSettings(MODELS_BLACKLIST="gpt-4-turbo, gpt-3.5-turbo, unknown-model")
        assert settings.models_blacklist == ["gpt-4-turbo", "gpt-3.5-turbo", "unknown-model"]

    def test_blacklist_parsing_strips_whitespace(self):
        """Test that whitespace is stripped from model names."""
        settings = GPTSettings(MODELS_BLACKLIST="model-1 , model-2 ,  model-3")
        assert settings.models_blacklist == ["model-1", "model-2", "model-3"]


class TestModelFilteringPriority:
    """Tests for whitelist/blacklist filtering priority.

    The priority is: whitelist > blacklist
    - If whitelist is set, only models in whitelist are returned
    - If whitelist is empty and blacklist is set, models in blacklist are excluded
    """

    def test_whitelist_takes_priority_over_blacklist(self):
        """Test that whitelist is applied even when blacklist is also set."""
        settings = GPTSettings(
            MODELS_WHITELIST="gpt-4o, gpt-4o-mini",
            MODELS_BLACKLIST="gpt-4o",
        )

        # Whitelist should take priority
        assert settings.models_whitelist == ["gpt-4o", "gpt-4o-mini"]
        assert settings.models_blacklist == ["gpt-4o"]

        # The filtering logic in provider.py checks whitelist first
        all_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet"]
        filtered = [m for m in all_models if m in settings.models_whitelist]
        assert filtered == ["gpt-4o", "gpt-4o-mini"]

    def test_blacklist_works_when_whitelist_empty(self):
        """Test that blacklist is applied when whitelist is empty."""
        settings = GPTSettings(
            MODELS_WHITELIST=None,
            MODELS_BLACKLIST="gpt-4-turbo, gpt-3.5-turbo",
        )

        all_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        filtered = [m for m in all_models if m not in settings.models_blacklist]
        assert filtered == ["gpt-4o", "gpt-4o-mini"]

    def test_blacklist_excludes_only_listed_models(self):
        """Test that blacklist only excludes models that are explicitly listed."""
        settings = GPTSettings(MODELS_BLACKLIST="gpt-4-turbo")

        all_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet"]
        filtered = [m for m in all_models if m not in settings.models_blacklist]
        assert filtered == ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]


class TestBlacklistFilteringWithUnknownModels:
    """Tests for blacklist filtering behavior with unknown model names."""

    def test_blacklist_with_unknown_models_still_works(self):
        """Test that blacklist containing unknown models doesn't break filtering."""
        settings = GPTSettings(MODELS_BLACKLIST="unknown-model-1, known-model")

        all_models = ["known-model", "another-known", "yet-another"]
        filtered = [m for m in all_models if m not in settings.models_blacklist]
        # Unknown models in blacklist are simply not matched
        assert filtered == ["another-known", "yet-another"]

    def test_blacklist_unknown_model_does_not_affect_whitelist_priority(self):
        """Test that unknown models in blacklist don't affect whitelist priority."""
        settings = GPTSettings(
            MODELS_WHITELIST="model-a, model-b",
            MODELS_BLACKLIST="unknown-model, model-a",
        )

        # When whitelist is set, blacklist is ignored
        all_models = ["model-a", "model-b", "model-c", "unknown-model"]
        filtered = [m for m in all_models if m in settings.models_whitelist]
        assert filtered == ["model-a", "model-b"]

    def test_blacklist_empty_after_filtering_unknown(self):
        """Test filtering when blacklist only contains unknown models."""
        settings = GPTSettings(MODELS_BLACKLIST="unknown-model-1, unknown-model-2")

        all_models = ["model-a", "model-b", "model-c"]
        filtered = [m for m in all_models if m not in settings.models_blacklist]
        # All models pass because none are in the blacklist
        assert filtered == ["model-a", "model-b", "model-c"]

    def test_combined_whitelist_and_blacklist_with_unknown(self):
        """Test combined whitelist and blacklist with unknown models."""
        settings = GPTSettings(
            MODELS_WHITELIST="gpt-4o, gpt-4o-mini, known-model",
            MODELS_BLACKLIST="unknown-model, excluded-model",
        )

        all_models = ["gpt-4o", "gpt-4o-mini", "known-model", "excluded-model", "unlisted-model"]
        # Whitelist is applied first, blacklist is ignored when whitelist exists
        filtered = [m for m in all_models if m in settings.models_whitelist]
        assert filtered == ["gpt-4o", "gpt-4o-mini", "known-model"]
