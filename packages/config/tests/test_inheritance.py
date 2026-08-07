"""Tests for configuration inheritance utilities."""

import warnings
from pathlib import Path

import pytest
import yaml

from dataknobs_common import CallableResolver, MappingResolver
from dataknobs_config import (
    InheritableConfigLoader,
    InheritanceError,
    RequiredEnvVarError,
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

    def test_url_with_double_slash_preserved(self, monkeypatch):
        """URLs with :// must not have double-slash collapsed.

        Path(result).expanduser() normalizes path separators, turning
        postgresql://host:5432/db into postgresql:/host:5432/db.
        os.path.expanduser() only expands ~ and leaves URLs intact.
        """
        monkeypatch.setenv("DB_URL", "postgresql://host:5432/db")

        result = substitute_env_vars({"dsn": "${DB_URL}"})
        assert result["dsn"] == "postgresql://host:5432/db"

    def test_url_env_var_substitution_preserves_scheme(self, monkeypatch):
        """Various URL schemes must survive env var substitution."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("HTTP_URL", "https://api.example.com/v1")

        result = substitute_env_vars({
            "redis": "${REDIS_URL}",
            "api": "${HTTP_URL}",
        })
        assert result["redis"] == "redis://localhost:6379/0"
        assert result["api"] == "https://api.example.com/v1"

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

    def test_dict_key_substitution(self, monkeypatch):
        """Dict keys containing ${VAR} must be substituted."""
        monkeypatch.setenv("TOKEN", "secret-abc-123")

        result = substitute_env_vars({"${TOKEN}": {"role": "admin"}})
        assert "secret-abc-123" in result
        assert "${TOKEN}" not in result
        assert result["secret-abc-123"]["role"] == "admin"

    def test_dict_key_substitution_with_default(self, monkeypatch):
        """Dict keys with default syntax ${VAR:default} must be substituted."""
        monkeypatch.delenv("MISSING_KEY_VAR", raising=False)

        result = substitute_env_vars({"${MISSING_KEY_VAR:fallback}": "value"})
        assert "fallback" in result
        assert result["fallback"] == "value"

    def test_dict_key_non_string_unchanged(self):
        """Non-string dict keys (ints, etc.) pass through unchanged."""
        data = {42: "int-key", True: "bool-key"}
        result = substitute_env_vars(data)
        assert result[42] == "int-key"
        assert result[True] == "bool-key"

    def test_dict_key_substitution_nested(self, monkeypatch):
        """Dict keys at multiple nesting levels must be substituted."""
        monkeypatch.setenv("OUTER_KEY", "outer")
        monkeypatch.setenv("INNER_KEY", "inner")

        data = {"${OUTER_KEY}": {"${INNER_KEY}": "deep_value"}}
        result = substitute_env_vars(data)
        assert result["outer"]["inner"] == "deep_value"

    def test_dict_key_tilde_expansion(self):
        """Tilde expansion via os.path.expanduser applies to dict keys."""
        result = substitute_env_vars({"~/configs": "value"})
        key = next(iter(result))
        assert "~" not in key
        assert key.endswith("/configs")

    # --- bash-superset syntax + new option flags ---

    def test_dash_default_syntax(self, monkeypatch):
        """Bash-style ${VAR:-default} works as alias for ${VAR:default}."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = substitute_env_vars({"k": "${MISSING_VAR:-fallback}"})
        assert result == {"k": "fallback"}

    def test_question_mark_error_syntax(self, monkeypatch):
        """Bash-style ${VAR:?msg} raises with the custom message when unset."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(
            ValueError, match="Required environment variable not set: DB password is required"
        ):
            substitute_env_vars({"k": "${MISSING_VAR:?DB password is required}"})

    def test_question_mark_uses_var_name_when_msg_empty(self, monkeypatch):
        """${VAR:?} (empty message) raises using the variable name."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(
            ValueError, match="Required environment variable not set: MISSING_VAR"
        ):
            substitute_env_vars({"k": "${MISSING_VAR:?}"})

    def test_question_mark_returns_value_when_set(self, monkeypatch):
        """${VAR:?msg} returns the env value when the variable is set."""
        monkeypatch.setenv("PRESENT_VAR", "hello")
        result = substitute_env_vars({"k": "${PRESENT_VAR:?must be set}"})
        assert result == {"k": "hello"}

    def test_type_coerce_int(self, monkeypatch):
        """type_coerce=True returns int when the entire string is a numeric ${VAR}."""
        monkeypatch.setenv("PORT", "5432")
        result = substitute_env_vars({"port": "${PORT}"}, type_coerce=True)
        assert result == {"port": 5432}

    def test_type_coerce_bool(self, monkeypatch):
        """type_coerce=True returns bool for the unambiguous bool words.

        Only ``true``/``false``/``yes``/``no`` (case-insensitive) coerce
        to bool. ``"0"`` / ``"1"`` are tested separately as int to lock
        in that they are NOT treated as booleans (see
        ``test_type_coerce_zero_one_are_int``).
        """
        monkeypatch.setenv("ENABLED", "true")
        monkeypatch.setenv("DISABLED", "false")
        monkeypatch.setenv("YES_VAR", "Yes")
        monkeypatch.setenv("NO_VAR", "NO")
        result = substitute_env_vars(
            {
                "on": "${ENABLED}",
                "off": "${DISABLED}",
                "yes_v": "${YES_VAR}",
                "no_v": "${NO_VAR}",
            },
            type_coerce=True,
        )
        assert result == {"on": True, "off": False, "yes_v": True, "no_v": False}

    def test_type_coerce_zero_one_are_int(self, monkeypatch):
        """${VAR}=='0' and '1' coerce to int, not bool.

        Bash conflates ``"0"`` / ``"1"`` with falsey/truthy, but for
        config values like port / count / size, callers expect an
        integer. ``isinstance(result, bool)`` is False here even though
        ``bool`` is a subclass of ``int`` — the actual type must be
        ``int``.
        """
        monkeypatch.setenv("ZERO", "0")
        monkeypatch.setenv("ONE", "1")
        result = substitute_env_vars(
            {"z": "${ZERO}", "o": "${ONE}"}, type_coerce=True
        )
        assert result == {"z": 0, "o": 1}
        assert type(result["z"]) is int
        assert type(result["o"]) is int

    def test_type_coerce_float(self, monkeypatch):
        """type_coerce=True returns float for numeric strings with a decimal."""
        monkeypatch.setenv("THRESHOLD", "0.95")
        result = substitute_env_vars({"t": "${THRESHOLD}"}, type_coerce=True)
        assert result == {"t": 0.95}

    def test_type_coerce_only_whole_value(self, monkeypatch):
        """type_coerce=True does NOT coerce mixed-content strings."""
        monkeypatch.setenv("PORT", "5432")
        result = substitute_env_vars({"k": "port=${PORT}"}, type_coerce=True)
        assert result == {"k": "port=5432"}  # string, not int

    def test_type_coerce_default_value(self, monkeypatch):
        """type_coerce=True coerces values from defaults too."""
        monkeypatch.delenv("PORT", raising=False)
        result = substitute_env_vars({"port": "${PORT:5432}"}, type_coerce=True)
        assert result == {"port": 5432}

    def test_type_coerce_preserves_string_for_non_numeric(self, monkeypatch):
        """type_coerce=True returns the original string when no coercion applies."""
        monkeypatch.setenv("NAME", "hello")
        result = substitute_env_vars({"name": "${NAME}"}, type_coerce=True)
        assert result == {"name": "hello"}

    def test_expand_user_paths_off(self, monkeypatch):
        """expand_user_paths=False leaves ~ literals intact."""
        monkeypatch.setenv("P", "~/data")
        result = substitute_env_vars(
            {"path": "${P}"}, expand_user_paths=False
        )
        assert result == {"path": "~/data"}

    def test_expand_user_paths_fast_path_tilde(self, monkeypatch, tmp_path):
        """type_coerce=True + expand_user_paths=True (default) expand ~ on the fast path.

        ``_substitute_string`` has two branches: a fast path for whole-value
        ``${VAR}`` placeholders that goes straight from ``_resolve_match``
        through ``os.path.expanduser`` to ``_convert_type``, and a slow
        path that uses ``re.sub`` for mixed-content strings. This test
        exercises the fast path with a tilde-valued env var to confirm
        ``os.path.expanduser`` is applied before type coercion (rather
        than skipped when ``type_coerce`` is on).
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("P", "~/data")
        result = substitute_env_vars({"path": "${P}"}, type_coerce=True)
        assert result == {"path": str(tmp_path / "data")}

    def test_substitute_keys_off(self, monkeypatch):
        """substitute_keys=False leaves ${VAR}-shaped keys as literals."""
        monkeypatch.setenv("K", "resolved")
        result = substitute_env_vars({"${K}": "v"}, substitute_keys=False)
        assert result == {"${K}": "v"}

    def test_keys_never_type_coerced(self, monkeypatch):
        """Even with type_coerce=True, dict keys remain strings (no int keys)."""
        monkeypatch.setenv("PORT", "5432")
        result = substitute_env_vars({"${PORT}": "v"}, type_coerce=True)
        assert "5432" in result  # string key
        assert 5432 not in result  # never int key

    def test_dash_default_set_value_parity(self, monkeypatch):
        """${VAR:default} and ${VAR:-default} return the env value identically when set."""
        monkeypatch.setenv("VAR", "real")
        legacy = substitute_env_vars({"k": "${VAR:fallback}"})
        bash = substitute_env_vars({"k": "${VAR:-fallback}"})
        assert legacy == bash == {"k": "real"}

    def test_dash_default_literal_dash(self, monkeypatch):
        """${VAR:--} resolves to a literal '-' default when VAR is unset."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = substitute_env_vars({"k": "${MISSING_VAR:--}"})
        assert result == {"k": "-"}

    def test_dash_default_with_dash_prefix(self, monkeypatch):
        """${VAR:--default} resolves to literal '-default' when VAR is unset."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = substitute_env_vars({"k": "${MISSING_VAR:--flag}"})
        assert result == {"k": "-flag"}

    def test_question_mark_msg_starting_with_colon(self, monkeypatch):
        """${VAR:?:foo} uses ':foo' as the error message when VAR is unset."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(
            ValueError, match=r"Required environment variable not set: :foo"
        ):
            substitute_env_vars({"k": "${MISSING_VAR:?:foo}"})

    def test_required_env_var_error_is_public_and_introspectable(self, monkeypatch):
        """RequiredEnvVarError is exposed publicly with usable attributes.

        Callers should be able to ``except RequiredEnvVarError`` directly
        (rather than catching ``ValueError`` and inspecting message text)
        and read ``var_name`` / ``bash_form`` / ``explicit_message`` to
        decide how to handle the failure.
        """
        monkeypatch.delenv("MISSING_VAR", raising=False)

        # Bare ${VAR} — bash_form=False, no explicit message.
        with pytest.raises(RequiredEnvVarError) as bare:
            substitute_env_vars({"k": "${MISSING_VAR}"})
        assert bare.value.var_name == "MISSING_VAR"
        assert bare.value.bash_form is False
        assert bare.value.explicit_message is None

        # ${VAR:?msg} — bash_form=True, message preserved.
        with pytest.raises(RequiredEnvVarError) as bash:
            substitute_env_vars({"k": "${MISSING_VAR:?must be set}"})
        assert bash.value.var_name == "MISSING_VAR"
        assert bash.value.bash_form is True
        assert bash.value.explicit_message == "must be set"

        # ${VAR:?} — bash_form=True, empty message normalized to None.
        with pytest.raises(RequiredEnvVarError) as empty:
            substitute_env_vars({"k": "${MISSING_VAR:?}"})
        assert empty.value.bash_form is True
        assert empty.value.explicit_message is None

        # Subclass relationship: existing ``except ValueError`` keeps working.
        assert issubclass(RequiredEnvVarError, ValueError)


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
        config_file = config_dir / "test.yaml"
        config_file.write_text("key: value")

        loader.load("test")

        # Cached: the rewrite on disk is not visible.
        config_file.write_text("key: rewritten")
        assert loader.load("test")["key"] == "value"

        loader.clear_cache("test")
        assert loader.load("test")["key"] == "rewritten"

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

        with pytest.raises(InheritanceError, match="Expected a dict at the root"):
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


# A value whose own *content* looks like a template. Substitution is idempotent
# for ordinary values and not for these, so a config holding one is the only
# thing that can tell "expanded once" apart from "expanded twice".
SECRET_WITH_VAR_SYNTAX = "p${GUARD_INNER}ss"
INNER_VALUE = "INJECTED"


class TestCacheSubstitutionProvenance:
    """The cache must not serve an entry expanded a different number of times.

    ``load()`` resolves ``extends:`` by recursing with ``substitute_vars=False``,
    so the same config can be produced in two forms. Storing both under one key
    makes the result of ``load(name)`` depend on what was loaded before it --
    in both directions.
    """

    @pytest.fixture
    def config_dir(self, tmp_path):
        """Create temporary config directory."""
        d = tmp_path / "configs"
        d.mkdir()
        return d

    @pytest.fixture(autouse=True)
    def _guard_env(self, monkeypatch):
        monkeypatch.setenv("GUARD_PW", SECRET_WITH_VAR_SYNTAX)
        monkeypatch.setenv("GUARD_INNER", INNER_VALUE)

    @pytest.fixture
    def parent_and_child(self, config_dir):
        """A child that extends a parent holding a `${`-containing secret."""
        (config_dir / "parent.yaml").write_text("secret: ${GUARD_PW}\nowner: parent")
        (config_dir / "child.yaml").write_text("extends: parent\nowner: child")
        return config_dir

    def _loader(self, config_dir):
        return InheritableConfigLoader(config_dir)

    def test_parent_loaded_after_child_is_substituted(self, parent_and_child):
        """Zero substitutions: the `extends:` recursion caches the raw parent.

        The recursion loads the parent with ``substitute_vars=False`` and caches
        it under the parent's name, so a later direct ``load("parent")`` -- which
        asked for substitution -- gets served the unexpanded entry.
        """
        loader = self._loader(parent_and_child)

        loader.load("child")
        parent = loader.load("parent")

        assert parent["secret"] == SECRET_WITH_VAR_SYNTAX

    def test_child_loaded_after_parent_substitutes_once(self, parent_and_child):
        """Two substitutions: an already-expanded parent is merged, then expanded.

        The reverse order. ``load("parent")`` caches the expanded parent; the
        child's recursion is served that entry despite asking for the raw form,
        and the merged result is expanded a second time -- so the secret's own
        content is read as a template.
        """
        loader = self._loader(parent_and_child)

        loader.load("parent")
        child = loader.load("child")

        assert child["secret"] == SECRET_WITH_VAR_SYNTAX
        assert child["secret"] != "pINJECTEDss", (
            "the secret's content was expanded as a template, disclosing "
            "GUARD_INNER into the value"
        )

    def test_load_order_does_not_change_result(self, parent_and_child):
        """Both orders equal the fresh-loader baseline.

        The two tests above are different symptoms of one defect; either alone
        would let a partial fix look complete. This is the property they are
        symptoms of.
        """
        baseline_child = self._loader(parent_and_child).load("child")
        baseline_parent = self._loader(parent_and_child).load("parent")

        child_first = self._loader(parent_and_child)
        child_first.load("child")
        parent_after_child = child_first.load("parent")

        parent_first = self._loader(parent_and_child)
        parent_first.load("parent")
        child_after_parent = parent_first.load("child")

        assert parent_after_child == baseline_parent
        assert child_after_parent == baseline_child

    def test_clear_cache_clears_both_substitution_variants(self, parent_and_child):
        """One `clear_cache(name)` clears the config, not one of its two forms.

        After a mixed-provenance load the parent is cached in both forms.
        Clearing only one would re-create the order dependence with an extra
        step, so both must re-read from disk after a single call.
        """
        loader = self._loader(parent_and_child)

        loader.load("child")  # caches the parent unsubstituted
        loader.load("parent")  # caches the parent substituted

        (parent_and_child / "parent.yaml").write_text("secret: ${GUARD_INNER}\nowner: rewritten")
        loader.clear_cache("parent")

        assert loader.load("parent", substitute_vars=False)["secret"] == "${GUARD_INNER}"
        assert loader.load("parent")["secret"] == INNER_VALUE


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


class TestCacheParticipationIsAllOrNothing:
    """``use_cache=False`` has to mean "not part of the cache", both ways.

    The cache write was unconditional, so a caller that asked to bypass the
    cache still populated it. Two callers do exactly that for a reason:
    ``validate`` is a dry run, and ``load_from_file`` temporarily rebinds
    ``config_dir`` to another directory. The second is the damaging one --
    the key holds no directory, so the entry it leaves behind answers later
    ``load()`` calls for a loader configured to read somewhere else.
    """

    def test_validate_does_not_warm_the_cache(self, tmp_path):
        (tmp_path / "svc.yaml").write_text(yaml.dump({"port": 1}))
        loader = InheritableConfigLoader(config_dir=tmp_path)

        loader.validate("svc")
        (tmp_path / "svc.yaml").write_text(yaml.dump({"port": 2}))

        assert loader.load("svc")["port"] == 2

    def test_load_from_file_does_not_answer_for_another_directory(
        self, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        configured = tmp_path / "configured"
        configured.mkdir()
        (elsewhere / "svc.yaml").write_text(yaml.dump({"origin": "elsewhere"}))
        (configured / "svc.yaml").write_text(yaml.dump({"origin": "configured"}))

        loader = InheritableConfigLoader(config_dir=configured)
        loader.load_from_file(elsewhere / "svc.yaml")

        assert loader.load("svc")["origin"] == "configured"

    def test_a_parent_pulled_in_by_load_from_file_is_not_cached_either(
        self, tmp_path
    ):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        configured = tmp_path / "configured"
        configured.mkdir()
        (elsewhere / "base.yaml").write_text(yaml.dump({"origin": "elsewhere"}))
        (elsewhere / "svc.yaml").write_text(
            yaml.dump({"extends": "base", "port": 1})
        )
        (configured / "base.yaml").write_text(
            yaml.dump({"origin": "configured"})
        )

        loader = InheritableConfigLoader(config_dir=configured)
        loader.load_from_file(elsewhere / "svc.yaml")

        # Read back in the form the recursion would have stored: it loads a
        # parent with substitute_vars=False, so asserting through the default
        # True would miss on the key alone and pass without the write gate.
        assert (
            loader.load("base", substitute_vars=False)["origin"] == "configured"
        )

    def test_a_bypassing_load_records_no_inheritance_edge(self, tmp_path):
        """Not participating in the cache means not recording edges for it.

        ``load_from_file`` rebinds ``config_dir``, and the edge would be filed
        under a bare parent name that means something else in the configured
        directory — so a later ``clear_cache`` there would evict a config that
        never inherited from it.
        """
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        configured = tmp_path / "configured"
        configured.mkdir()
        (elsewhere / "base.yaml").write_text(yaml.dump({"origin": "elsewhere"}))
        (elsewhere / "svc.yaml").write_text(
            yaml.dump({"extends": "base", "port": 1})
        )
        (configured / "base.yaml").write_text(yaml.dump({"origin": "conf"}))
        (configured / "svc.yaml").write_text(yaml.dump({"port": 9}))

        loader = InheritableConfigLoader(config_dir=configured)
        loader.load_from_file(elsewhere / "svc.yaml")
        loader.load("svc")
        loader.clear_cache("base")

        assert loader._cache.get(("svc", True)) is not None


class TestClearingAParentClearsWhatInheritedFromIt:
    """A cached child holds a copy of its parent's content, merged in.

    Clearing the parent alone leaves that copy answering, so the next read of
    the child returns content the parent no longer has -- which is the exact
    staleness ``clear_cache`` is called to resolve, surviving the call.
    """

    def test_clearing_a_parent_reloads_the_child(self, tmp_path):
        (tmp_path / "base.yaml").write_text(yaml.dump({"timeout": 30}))
        (tmp_path / "svc.yaml").write_text(
            yaml.dump({"extends": "base", "port": 8080})
        )
        loader = InheritableConfigLoader(config_dir=tmp_path)

        assert loader.load("svc")["timeout"] == 30

        (tmp_path / "base.yaml").write_text(yaml.dump({"timeout": 60}))
        loader.clear_cache("base")

        assert loader.load("svc")["timeout"] == 60

    def test_clearing_a_grandparent_reaches_the_grandchild(self, tmp_path):
        (tmp_path / "root.yaml").write_text(yaml.dump({"region": "us-east"}))
        (tmp_path / "mid.yaml").write_text(
            yaml.dump({"extends": "root", "tier": "std"})
        )
        (tmp_path / "leaf.yaml").write_text(
            yaml.dump({"extends": "mid", "name": "leaf"})
        )
        loader = InheritableConfigLoader(config_dir=tmp_path)

        assert loader.load("leaf")["region"] == "us-east"

        (tmp_path / "root.yaml").write_text(yaml.dump({"region": "eu-west"}))
        loader.clear_cache("root")

        assert loader.load("leaf")["region"] == "eu-west"

    def test_clearing_a_child_leaves_its_parent_cached(self, tmp_path):
        """Invalidation runs down the inheritance edges, not up them."""
        (tmp_path / "base.yaml").write_text(yaml.dump({"timeout": 30}))
        (tmp_path / "svc.yaml").write_text(
            yaml.dump({"extends": "base", "port": 8080})
        )
        loader = InheritableConfigLoader(config_dir=tmp_path)
        loader.load("svc")

        (tmp_path / "base.yaml").write_text(yaml.dump({"timeout": 60}))
        loader.clear_cache("svc")

        # The parent was not cleared, so its cached form still answers.
        assert loader.load("svc")["timeout"] == 30

    def test_a_cycle_in_the_edges_does_not_hang_the_clear(self, tmp_path):
        """Recorded edges are walked, so the walk needs its own guard."""
        (tmp_path / "a.yaml").write_text(yaml.dump({"v": 1}))
        (tmp_path / "b.yaml").write_text(yaml.dump({"extends": "a", "w": 2}))
        loader = InheritableConfigLoader(config_dir=tmp_path)
        loader.load("b")

        # Forge a cycle directly: reachable only if a config were edited to
        # extend its own descendant between loads.
        loader._dependents.setdefault("b", set()).add("a")

        loader.clear_cache("a")

        assert loader._cache == {}


class TestNameResolution:
    """A deployment governs how a config *name* maps to a location.

    The mapping has to reach ``extends:`` targets too. Those names are written
    inside config files, so a consumer who can only intercept the entry point
    cannot express a layout convention at all -- the recursion into parents
    happens entirely inside the loader.
    """

    @pytest.fixture
    def domains(self, tmp_path):
        """A `domains/` layout: parents named bare inside the child."""
        (tmp_path / "domains").mkdir()
        (tmp_path / "domains" / "parent.yaml").write_text(
            yaml.dump({"a": 1, "shared": {"x": 1}})
        )
        (tmp_path / "domains" / "child.yaml").write_text(
            yaml.dump({"extends": "parent", "b": 2})
        )
        return tmp_path

    def test_extends_resolves_through_resolver(self, domains):
        loader = InheritableConfigLoader(
            domains, resolver=CallableResolver(lambda n: f"domains/{n}")
        )

        assert loader.load("child") == {"a": 1, "shared": {"x": 1}, "b": 2}

    def test_without_a_resolver_the_same_layout_fails(self, domains):
        """What the seam exists for: the parent is unreachable without it."""
        loader = InheritableConfigLoader(domains)

        with pytest.raises(InheritanceError, match=r"parent\.yaml"):
            loader.load("domains/child")

    def test_resolve_name_default_is_identity(self, tmp_path):
        """No resolver: a flat layout behaves exactly as before."""
        (tmp_path / "flat.yaml").write_text(yaml.dump({"k": "v"}))
        loader = InheritableConfigLoader(tmp_path)

        assert loader.resolve_name("flat") == "flat"
        assert loader.load("flat") == {"k": "v"}

    def test_resolver_returning_none_falls_back_to_identity(self, tmp_path):
        """``None`` is the ResourceResolver contract for "no mapping"."""
        (tmp_path / "known.yaml").write_text(yaml.dump({"k": 1}))
        (tmp_path / "unmapped.yaml").write_text(yaml.dump({"k": 2}))
        loader = InheritableConfigLoader(
            tmp_path, resolver=MappingResolver({"known": "known"})
        )

        assert loader.resolve_name("unmapped") == "unmapped"
        assert loader.load("unmapped") == {"k": 2}

    def test_mapping_resolver_covers_an_alias(self, domains):
        """A shipped resolver, no consumer class: an alias to a real file."""
        loader = InheritableConfigLoader(
            domains,
            resolver=MappingResolver(
                {"tutor": "domains/child", "parent": "domains/parent"}
            ),
        )

        assert loader.load("tutor") == {"a": 1, "shared": {"x": 1}, "b": 2}

    def test_subclass_resolve_name_override(self, domains):
        """The second access mode: override the public method."""

        class DomainAware(InheritableConfigLoader):
            def resolve_name(self, name):
                return f"domains/{name}"

        assert DomainAware(domains).load("child") == {
            "a": 1,
            "shared": {"x": 1},
            "b": 2,
        }

    def test_an_override_replaces_the_injected_resolver(self, domains):
        """The two modes are alternatives; using both is not additive.

        An override replaces the default implementation, so the resolver is
        never consulted. The outcome would otherwise be silent -- the loader
        reads a file the caller did not configure and says nothing -- which
        is why construction warns.
        """

        class DomainAware(InheritableConfigLoader):
            def resolve_name(self, name):
                return f"domains/{name}"

        with pytest.warns(UserWarning, match="alternatives, not layers"):
            loader = DomainAware(
                domains, resolver=MappingResolver({"child": "somewhere-else"})
            )

        assert loader.resolve_name("child") == "domains/child"

    def test_an_override_delegating_to_super_applies_both(self, tmp_path):
        """The other half of the same trap, and the noisier one."""

        class Stacked(InheritableConfigLoader):
            def resolve_name(self, name):
                return f"domains/{super().resolve_name(name)}"

        with pytest.warns(UserWarning, match="alternatives, not layers"):
            loader = Stacked(
                tmp_path, resolver=CallableResolver(lambda n: f"domains/{n}")
            )

        assert loader.resolve_name("child") == "domains/domains/child"


class TestResolvedNameIsTheOneIdentity:
    """Everything the loader keys on a name keys on the *resolved* one.

    Two spellings of one config that shared a cache but not a cycle set -- or
    a cache but not an invalidation edge -- would be two configs to one
    structure and one config to another. That incoherence is worse than the
    duplication it would replace.
    """

    @pytest.fixture
    def aliased(self, tmp_path):
        """`c` and `child` both name domains/child.yaml."""
        (tmp_path / "domains").mkdir()
        (tmp_path / "domains" / "parent.yaml").write_text(yaml.dump({"t": 30}))
        (tmp_path / "domains" / "child.yaml").write_text(
            yaml.dump({"extends": "parent", "b": 2})
        )
        return tmp_path

    def _loader(self, root):
        return InheritableConfigLoader(
            root,
            resolver=MappingResolver(
                {
                    "c": "domains/child",
                    "child": "domains/child",
                    "parent": "domains/parent",
                }
            ),
        )

    def test_cache_keyed_on_resolved_name(self, aliased):
        loader = self._loader(aliased)

        loader.load("c")
        loader.load("child")

        # Both spellings, one entry -- asserting on the name half only, since
        # the other half is the substitution mode.
        names = {key[0] for key in loader._cache}
        assert names == {"domains/child", "domains/parent"}

    def test_two_spellings_return_the_same_object(self, aliased):
        loader = self._loader(aliased)

        assert loader.load("c") is loader.load("child")

    def test_clear_cache_resolves_name(self, aliased):
        """``clear_cache("c")`` clears what ``load("child")`` stored."""
        loader = self._loader(aliased)
        loader.load("child")

        (aliased / "domains" / "child.yaml").write_text(
            yaml.dump({"extends": "parent", "b": 99})
        )
        loader.clear_cache("c")

        assert loader.load("child")["b"] == 99

    def test_inheritance_edges_use_resolved_names(self, aliased):
        """Clearing a parent must still reach a child cached under an alias.

        The edges are walked against the cache's own keys, so recording them
        under the names the consumer happened to write would make the walk
        compute names the cache cannot match -- and the child would keep
        answering with the parent's old content.
        """
        loader = self._loader(aliased)
        assert loader.load("c")["t"] == 30

        (aliased / "domains" / "parent.yaml").write_text(yaml.dump({"t": 60}))
        loader.clear_cache("parent")

        assert loader.load("c")["t"] == 60

    def test_cycle_detection_uses_resolved_name(self, tmp_path):
        """Two spellings of one config are one node in the cycle graph.

        A cycle is caught either way -- ``extends:`` is single-valued, so a
        repeated *config* eventually repeats a *spelling* too. What the
        resolved keying buys is catching it at the first repeat: a raw cycle
        set walks the whole cycle again before it recognizes one, so the
        config is read twice, and a longer alias chain is re-read in full.
        """
        (tmp_path / "domains").mkdir()
        (tmp_path / "domains" / "a.yaml").write_text(
            yaml.dump({"extends": "alias-of-a", "v": 1})
        )

        class Counting(InheritableConfigLoader):
            reads = 0

            def _load_file(self, name):
                self.reads += 1
                return super()._load_file(name)

        loader = Counting(
            tmp_path,
            resolver=MappingResolver(
                {"a": "domains/a", "alias-of-a": "domains/a"}
            ),
        )

        with pytest.raises(InheritanceError, match="domains/a"):
            loader.load("a")

        assert loader.reads == 1


class TestLoadFromFileIgnoresResolution:
    """``load_from_file`` rebinds ``config_dir``; a mapping defined against
    the configured directory cannot be correct against that one.

    Suppression covers the whole subtree, not just the entry file -- the
    rebinding applies to every ``extends:`` target below it too.
    """

    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "domains").mkdir()
        (tmp_path / "domains" / "parent.yaml").write_text(
            yaml.dump({"a": 1, "shared": {"x": 1}})
        )
        (tmp_path / "domains" / "child.yaml").write_text(
            yaml.dump({"extends": "parent", "b": 2})
        )
        return tmp_path

    def _prefixing(self, root):
        return InheritableConfigLoader(
            root, resolver=CallableResolver(lambda n: f"domains/{n}")
        )

    def test_load_from_file_ignores_resolver(self, tree):
        loader = self._prefixing(tree)

        assert loader.load_from_file(tree / "domains" / "child.yaml") == {
            "a": 1,
            "shared": {"x": 1},
            "b": 2,
        }

    def test_load_from_file_extends_ignores_resolver(self, tmp_path):
        """The subtree, not just the entry file.

        The parent here exists ONLY beside the child, so a resolver applied
        one level down would look for it under a nested `domains/` that does
        not exist.
        """
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "base.yaml").write_text(yaml.dump({"t": 30}))
        (tmp_path / "elsewhere" / "svc.yaml").write_text(
            yaml.dump({"extends": "base", "port": 8080})
        )
        loader = self._prefixing(tmp_path)

        result = loader.load_from_file(tmp_path / "elsewhere" / "svc.yaml")

        assert result == {"t": 30, "port": 8080}

    def test_a_subclass_override_cannot_defeat_the_suppression(self, tree):
        """The flag is read where resolution is invoked, so the override is
        not called at all -- a suppression a subclass can accidentally break
        is not a suppression.
        """

        class DomainAware(InheritableConfigLoader):
            def resolve_name(self, name):
                return f"domains/{name}"

        loader = DomainAware(tree)

        assert loader.load_from_file(tree / "domains" / "child.yaml")["b"] == 2

    def test_resolution_is_restored_after_load_from_file(self, tree):
        loader = self._prefixing(tree)
        loader.load_from_file(tree / "domains" / "child.yaml")

        assert loader.load("child")["b"] == 2

    def test_a_missing_file_is_rejected_before_anything_is_rebound(self, tree):
        """The early exit, which is not a test of the ``finally``.

        This raise happens above the swap, so neither ``config_dir`` nor the
        suppression flag has been touched when it fires. Restoration is not
        exercised here at all -- see the next test, which fails after the
        swap, for that.
        """
        loader = self._prefixing(tree)

        with pytest.raises(InheritanceError, match="not found"):
            loader.load_from_file(tree / "domains" / "missing.yaml")

        assert loader.load("child")["b"] == 2

    def test_resolution_is_restored_after_a_failure(self, tree):
        """A failure *after* the swap still restores both fields.

        The entry file exists, so the swap happens; its ``extends:`` target
        does not, so the load raises with the flag set and ``config_dir``
        rebound. A leaked flag is the worse of the two: it disables the
        layout convention for every later load on this loader, and surfaces
        as file-not-found on an unresolved path, a long way from the cause.
        """
        (tree / "domains" / "orphan.yaml").write_text(
            yaml.dump({"extends": "no-such-parent", "c": 3})
        )
        loader = self._prefixing(tree)

        with pytest.raises(InheritanceError, match="no-such-parent"):
            loader.load_from_file(tree / "domains" / "orphan.yaml")

        assert loader._bypass_resolution is False
        assert loader.config_dir == tree
        assert loader.load("child")["b"] == 2


class TestEnumerationIsTheOtherHalfOfTheMapping:
    """``resolve_name`` is one-way, so enumeration cannot be derived from it.

    A resolver answers "where does this name live". Nothing runs it backwards
    to recover the names from the locations, so a deployment that governs the
    mapping has to govern enumeration too -- which is what
    ``available_names`` is for.
    """

    @pytest.fixture
    def nested(self, tmp_path):
        (tmp_path / "domains").mkdir()
        (tmp_path / "domains" / "parent.yaml").write_text(yaml.dump({"a": 1}))
        (tmp_path / "domains" / "child.yaml").write_text(
            yaml.dump({"extends": "parent", "b": 2})
        )
        return tmp_path

    def test_the_default_is_the_stems_under_config_dir(self, tmp_path):
        (tmp_path / "one.yaml").write_text(yaml.dump({"a": 1}))
        (tmp_path / "two.json").write_text('{"b": 2}')

        assert InheritableConfigLoader(tmp_path).available_names() == ["one", "two"]

    def test_list_available_delegates(self, tmp_path):
        """The old entry point routes through the new seam.

        An override has to take effect for every caller, not only the ones
        that learned the new name.
        """
        (tmp_path / "real.yaml").write_text(yaml.dump({"a": 1}))

        class Fixed(InheritableConfigLoader):
            def available_names(self):
                return ["declared"]

        assert Fixed(tmp_path).list_available() == ["declared"]

    def test_the_default_reports_nothing_under_a_resolver(self, nested):
        """The gap the seam exists to close, pinned so it stays deliberate.

        Every config is a directory down, so the top-level glob finds none of
        them and the natural enumerate-then-load loop runs zero times. It is
        the default being wrong for this layout, not a failure to load.
        """
        loader = InheritableConfigLoader(
            nested, resolver=CallableResolver(lambda n: f"domains/{n}")
        )

        assert loader.load("child")["b"] == 2
        assert loader.available_names() == []

    def test_an_override_makes_every_reported_name_loadable(self, nested):
        """The property that matters: what it lists, `load` accepts."""

        class DomainLoader(InheritableConfigLoader):
            def resolve_name(self, name: str) -> str:
                return f"domains/{name}"

            def available_names(self) -> list[str]:
                return sorted(
                    path.stem
                    for path in (self.config_dir / "domains").glob("*.yaml")
                    if path.is_file()
                )

        loader = DomainLoader(nested)

        assert loader.available_names() == ["child", "parent"]
        for name in loader.available_names():
            assert loader.load(name)


class TestBothModesAtOnceIsReported:
    """An override *replaces* ``resolve_name``, so a loader given both modes
    ignores the injected resolver -- silently, and it then reads a different
    file than the caller configured. Warn rather than raise: overriding to
    normalize or log *and* delegating to ``super()`` is a legitimate use of
    both, and raising would break it.
    """

    def test_override_plus_resolver_warns(self, tmp_path):
        class Overriding(InheritableConfigLoader):
            def resolve_name(self, name: str) -> str:
                return f"domains/{name}"

        with pytest.warns(UserWarning, match="alternatives, not layers"):
            Overriding(tmp_path, resolver=MappingResolver({"a": "b"}))

    def test_an_override_alone_is_silent(self, tmp_path):
        class Overriding(InheritableConfigLoader):
            def resolve_name(self, name: str) -> str:
                return f"domains/{name}"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Overriding(tmp_path)

    def test_a_resolver_alone_is_silent(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            InheritableConfigLoader(tmp_path, resolver=MappingResolver({"a": "b"}))

    def test_a_subclass_that_does_not_touch_resolve_name_is_silent(self, tmp_path):
        """Subclassing is not the trigger -- overriding the method is."""

        class Unrelated(InheritableConfigLoader):
            pass

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Unrelated(tmp_path, resolver=MappingResolver({"a": "b"}))
