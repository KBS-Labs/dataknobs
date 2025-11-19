"""Tests for configuration inheritance utilities."""

import json
import os
from pathlib import Path

import pytest
import yaml

from dataknobs_config import (
    InheritableConfigLoader,
    InheritanceError,
    deep_merge,
    load_config_with_inheritance,
    substitute_env_vars,
)


class TestDeepMerge:
    """Test deep_merge utility function."""

    def test_simple_merge(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        # Original dicts should be unchanged
        assert base == {"a": 1, "b": 2}
        assert override == {"b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merging nested dictionaries."""
        base = {
            "a": 1,
            "nested": {"x": 10, "y": 20},
        }
        override = {
            "nested": {"y": 25, "z": 30},
        }
        result = deep_merge(base, override)

        assert result == {
            "a": 1,
            "nested": {"x": 10, "y": 25, "z": 30},
        }

    def test_deeply_nested_merge(self):
        """Test deeply nested merge."""
        base = {
            "level1": {
                "level2": {
                    "level3": {"a": 1, "b": 2},
                },
            },
        }
        override = {
            "level1": {
                "level2": {
                    "level3": {"b": 3, "c": 4},
                },
            },
        }
        result = deep_merge(base, override)

        assert result["level1"]["level2"]["level3"] == {"a": 1, "b": 3, "c": 4}

    def test_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)

        assert result["items"] == [4, 5]

    def test_type_override(self):
        """Test that different types override completely."""
        base = {"value": {"nested": True}}
        override = {"value": "string now"}
        result = deep_merge(base, override)

        assert result["value"] == "string now"

    def test_empty_dicts(self):
        """Test merging with empty dicts."""
        assert deep_merge({}, {"a": 1}) == {"a": 1}
        assert deep_merge({"a": 1}, {}) == {"a": 1}
        assert deep_merge({}, {}) == {}


class TestSubstituteEnvVars:
    """Test environment variable substitution."""

    def test_simple_substitution(self, monkeypatch):
        """Test simple env var substitution."""
        monkeypatch.setenv("TEST_VAR", "hello")

        result = substitute_env_vars({"key": "${TEST_VAR}"})
        assert result["key"] == "hello"

    def test_default_value(self, monkeypatch):
        """Test default value when env var not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)

        result = substitute_env_vars({"key": "${MISSING_VAR:default}"})
        assert result["key"] == "default"

    def test_required_var_missing(self, monkeypatch):
        """Test error when required var is missing."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)

        with pytest.raises(ValueError, match="Required environment variable not set"):
            substitute_env_vars({"key": "${REQUIRED_VAR}"})

    def test_nested_substitution(self, monkeypatch):
        """Test substitution in nested structure."""
        monkeypatch.setenv("NESTED_VAR", "nested_value")

        data = {
            "level1": {
                "level2": "${NESTED_VAR}",
            },
        }
        result = substitute_env_vars(data)
        assert result["level1"]["level2"] == "nested_value"

    def test_list_substitution(self, monkeypatch):
        """Test substitution in lists."""
        monkeypatch.setenv("LIST_VAR", "list_value")

        data = {"items": ["${LIST_VAR}", "static"]}
        result = substitute_env_vars(data)
        assert result["items"] == ["list_value", "static"]

    def test_multiple_vars_in_string(self, monkeypatch):
        """Test multiple vars in same string."""
        monkeypatch.setenv("VAR1", "hello")
        monkeypatch.setenv("VAR2", "world")

        result = substitute_env_vars({"key": "${VAR1} ${VAR2}"})
        assert result["key"] == "hello world"

    def test_tilde_expansion(self, monkeypatch):
        """Test home directory tilde expansion."""
        monkeypatch.setenv("PATH_VAR", "~/test")

        result = substitute_env_vars({"path": "${PATH_VAR}"})
        assert "~" not in result["path"]
        assert result["path"].endswith("/test")

    def test_empty_default(self, monkeypatch):
        """Test empty string as default value."""
        monkeypatch.delenv("EMPTY_DEFAULT", raising=False)

        result = substitute_env_vars({"key": "${EMPTY_DEFAULT:}"})
        assert result["key"] == ""

    def test_non_string_values_unchanged(self):
        """Test that non-string values are unchanged."""
        data = {
            "number": 42,
            "boolean": True,
            "null": None,
            "float": 3.14,
        }
        result = substitute_env_vars(data)
        assert result == data


class TestInheritableConfigLoader:
    """Test InheritableConfigLoader class."""

    @pytest.fixture
    def config_dir(self, tmp_path):
        """Create temporary config directory."""
        return tmp_path / "configs"

    @pytest.fixture
    def loader(self, config_dir):
        """Create loader with temp config directory."""
        config_dir.mkdir()
        return InheritableConfigLoader(config_dir)

    def test_load_simple_yaml(self, loader, config_dir):
        """Test loading simple YAML config."""
        config_file = config_dir / "simple.yaml"
        config_file.write_text("""
llm:
  provider: openai
  model: gpt-4
""")

        result = loader.load("simple")
        assert result["llm"]["provider"] == "openai"
        assert result["llm"]["model"] == "gpt-4"

    def test_load_json_config(self, loader, config_dir):
        """Test loading JSON config."""
        config_file = config_dir / "config.json"
        config_file.write_text('{"key": "value", "number": 42}')

        result = loader.load("config")
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_load_yml_extension(self, loader, config_dir):
        """Test loading .yml extension."""
        config_file = config_dir / "config.yml"
        config_file.write_text("key: value")

        result = loader.load("config")
        assert result["key"] == "value"

    def test_config_not_found(self, loader):
        """Test error when config not found."""
        with pytest.raises(InheritanceError, match="not found"):
            loader.load("nonexistent")

    def test_simple_inheritance(self, loader, config_dir):
        """Test simple single-level inheritance."""
        # Create base config
        base_file = config_dir / "base.yaml"
        base_file.write_text("""
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
""")

        # Create child config
        child_file = config_dir / "child.yaml"
        child_file.write_text("""
extends: base

llm:
  model: gpt-4-turbo
""")

        result = loader.load("child")

        # Should have base values
        assert result["llm"]["provider"] == "openai"
        assert result["llm"]["temperature"] == 0.7
        # Should have overridden value
        assert result["llm"]["model"] == "gpt-4-turbo"
        # extends field should be removed
        assert "extends" not in result

    def test_multi_level_inheritance(self, loader, config_dir):
        """Test multi-level inheritance chain."""
        # Create base
        (config_dir / "base.yaml").write_text("a: 1\nb: 2\nc: 3")

        # Create middle
        (config_dir / "middle.yaml").write_text("extends: base\nb: 20")

        # Create child
        (config_dir / "child.yaml").write_text("extends: middle\nc: 30")

        result = loader.load("child")
        assert result == {"a": 1, "b": 20, "c": 30}

    def test_circular_inheritance_detection(self, loader, config_dir):
        """Test circular inheritance is detected."""
        # Create circular reference
        (config_dir / "a.yaml").write_text("extends: b\nvalue: a")
        (config_dir / "b.yaml").write_text("extends: a\nvalue: b")

        with pytest.raises(InheritanceError, match="Circular inheritance"):
            loader.load("a")

    def test_caching(self, loader, config_dir):
        """Test configuration caching."""
        config_file = config_dir / "cached.yaml"
        config_file.write_text("key: original")

        # First load
        result1 = loader.load("cached")
        assert result1["key"] == "original"

        # Modify file
        config_file.write_text("key: modified")

        # Second load should return cached value
        result2 = loader.load("cached", use_cache=True)
        assert result2["key"] == "original"

        # Load without cache should get new value
        result3 = loader.load("cached", use_cache=False)
        assert result3["key"] == "modified"

    def test_clear_cache(self, loader, config_dir):
        """Test cache clearing."""
        (config_dir / "test.yaml").write_text("key: value")

        loader.load("test")
        assert "test" in loader._cache

        loader.clear_cache("test")
        assert "test" not in loader._cache

    def test_clear_all_cache(self, loader, config_dir):
        """Test clearing all cache."""
        (config_dir / "a.yaml").write_text("key: a")
        (config_dir / "b.yaml").write_text("key: b")

        loader.load("a")
        loader.load("b")
        assert len(loader._cache) == 2

        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_env_var_substitution(self, loader, config_dir, monkeypatch):
        """Test environment variable substitution."""
        monkeypatch.setenv("TEST_VALUE", "from_env")

        config_file = config_dir / "env.yaml"
        config_file.write_text("key: ${TEST_VALUE}")

        result = loader.load("env")
        assert result["key"] == "from_env"

    def test_disable_env_substitution(self, loader, config_dir, monkeypatch):
        """Test disabling env var substitution."""
        monkeypatch.setenv("TEST_VALUE", "from_env")

        config_file = config_dir / "noenv.yaml"
        config_file.write_text("key: ${TEST_VALUE}")

        result = loader.load("noenv", substitute_vars=False)
        assert result["key"] == "${TEST_VALUE}"

    def test_list_available(self, loader, config_dir):
        """Test listing available configurations."""
        (config_dir / "a.yaml").write_text("key: a")
        (config_dir / "b.json").write_text('{"key": "b"}')
        (config_dir / "c.yml").write_text("key: c")

        available = loader.list_available()
        assert "a" in available
        assert "b" in available
        assert "c" in available

    def test_list_available_empty_dir(self, tmp_path):
        """Test listing with empty or missing directory."""
        loader = InheritableConfigLoader(tmp_path / "nonexistent")
        assert loader.list_available() == []

    def test_validate_valid_config(self, loader, config_dir):
        """Test validating a valid config."""
        (config_dir / "valid.yaml").write_text("key: value")

        is_valid, error = loader.validate("valid")
        assert is_valid is True
        assert error is None

    def test_validate_invalid_config(self, loader, config_dir):
        """Test validating an invalid config."""
        (config_dir / "invalid.yaml").write_text("- not a dict")

        is_valid, error = loader.validate("invalid")
        assert is_valid is False
        assert error is not None

    def test_validate_missing_config(self, loader):
        """Test validating a missing config."""
        is_valid, error = loader.validate("missing")
        assert is_valid is False
        assert "not found" in error

    def test_load_from_file(self, loader, tmp_path):
        """Test loading from absolute file path."""
        # Create config outside config_dir
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        config_file = other_dir / "external.yaml"
        config_file.write_text("key: external")

        result = loader.load_from_file(config_file)
        assert result["key"] == "external"

    def test_load_from_file_with_inheritance(self, loader, tmp_path):
        """Test load_from_file resolves inheritance relative to file."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        # Create base in other_dir
        (other_dir / "base.yaml").write_text("base_key: base_value")

        # Create child in other_dir
        child_file = other_dir / "child.yaml"
        child_file.write_text("extends: base\nchild_key: child_value")

        result = loader.load_from_file(child_file)
        assert result["base_key"] == "base_value"
        assert result["child_key"] == "child_value"

    def test_load_from_file_not_found(self, loader, tmp_path):
        """Test load_from_file with missing file."""
        with pytest.raises(InheritanceError, match="not found"):
            loader.load_from_file(tmp_path / "missing.yaml")

    def test_invalid_yaml(self, loader, config_dir):
        """Test error on invalid YAML."""
        (config_dir / "invalid.yaml").write_text("key: [unclosed")

        with pytest.raises(InheritanceError, match="Failed to parse"):
            loader.load("invalid")

    def test_invalid_json(self, loader, config_dir):
        """Test error on invalid JSON."""
        (config_dir / "invalid.json").write_text('{"key": invalid}')

        with pytest.raises(InheritanceError, match="Failed to parse"):
            loader.load("invalid")

    def test_non_dict_config(self, loader, config_dir):
        """Test error when config is not a dict."""
        (config_dir / "list.yaml").write_text("- item1\n- item2")

        with pytest.raises(InheritanceError, match="must contain a dictionary"):
            loader.load("list")

    def test_default_config_dir(self):
        """Test default config directory."""
        loader = InheritableConfigLoader()
        assert loader.config_dir == Path("./configs")

    def test_inheritance_adds_new_fields(self, loader, config_dir):
        """Test that inheritance adds new fields from child."""
        (config_dir / "base.yaml").write_text("existing: value")
        (config_dir / "child.yaml").write_text("extends: base\nnew_field: new_value")

        result = loader.load("child")
        assert result["existing"] == "value"
        assert result["new_field"] == "new_value"


class TestLoadConfigWithInheritance:
    """Test the convenience function."""

    def test_load_config_with_inheritance(self, tmp_path):
        """Test the convenience function."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")

        result = load_config_with_inheritance(config_file)
        assert result["key"] == "value"

    def test_load_with_inheritance_chain(self, tmp_path):
        """Test convenience function with inheritance."""
        (tmp_path / "base.yaml").write_text("a: 1")
        child_file = tmp_path / "child.yaml"
        child_file.write_text("extends: base\nb: 2")

        result = load_config_with_inheritance(child_file)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_load_without_substitution(self, tmp_path, monkeypatch):
        """Test convenience function without env substitution."""
        monkeypatch.setenv("VAR", "value")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: ${VAR}")

        result = load_config_with_inheritance(config_file, substitute_vars=False)
        assert result["key"] == "${VAR}"
