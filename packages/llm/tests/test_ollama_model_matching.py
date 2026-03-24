"""Tests for Ollama model name matching logic.

Regression test for a bug where 'nomic-embed-text' incorrectly matched
'nomic-embed-text-v2-moe:latest' due to a greedy startswith() prefix check.
The fix requires exact name or name-with-tag matching only.
"""

from dataknobs_llm.llm.providers.ollama import _find_matching_models


class TestFindMatchingModels:
    """Test the _find_matching_models helper function."""

    def test_exact_match(self):
        """Exact model name returns that model."""
        result = _find_matching_models("llama2:latest", ["llama2:latest", "mistral:latest"])
        assert result == ["llama2:latest"]

    def test_base_name_matches_tagged_variant(self):
        """Base name without tag matches the tagged version."""
        result = _find_matching_models("llama2", ["llama2:latest", "mistral:latest"])
        assert result == ["llama2:latest"]

    def test_base_name_does_not_match_different_model(self):
        """Base name must not match a model that merely shares a prefix.

        This is the specific bug: 'nomic-embed-text' was incorrectly matching
        'nomic-embed-text-v2-moe:latest' via startswith().
        """
        available = ["nomic-embed-text-v2-moe:latest", "mistral:latest"]
        result = _find_matching_models("nomic-embed-text", available)
        assert result == []

    def test_base_name_matches_correct_model_among_similar(self):
        """When both the correct and a prefix-similar model exist, only the correct one matches."""
        available = [
            "nomic-embed-text-v2-moe:latest",
            "nomic-embed-text:latest",
            "mistral:latest",
        ]
        result = _find_matching_models("nomic-embed-text", available)
        assert result == ["nomic-embed-text:latest"]

    def test_no_models_available(self):
        """Empty available list returns empty."""
        result = _find_matching_models("llama2", [])
        assert result == []

    def test_no_match_at_all(self):
        """Model not available in any form returns empty."""
        result = _find_matching_models("llama2", ["mistral:latest", "gemma:latest"])
        assert result == []

    def test_multiple_tag_variants(self):
        """Multiple tagged variants of the same base name all match."""
        available = ["llama2:latest", "llama2:7b", "llama2:13b", "mistral:latest"]
        result = _find_matching_models("llama2", available)
        assert result == ["llama2:latest", "llama2:7b", "llama2:13b"]

    def test_tagged_config_exact_match(self):
        """Configured model with explicit tag matches exactly."""
        available = ["llama2:7b", "llama2:13b", "llama2:latest"]
        result = _find_matching_models("llama2:7b", available)
        assert result == ["llama2:7b"]

    def test_tagged_config_falls_back_to_base_name(self):
        """Configured model with missing tag falls back to base name matches."""
        available = ["llama2:latest", "llama2:13b"]
        result = _find_matching_models("llama2:7b", available)
        assert result == ["llama2:latest", "llama2:13b"]
