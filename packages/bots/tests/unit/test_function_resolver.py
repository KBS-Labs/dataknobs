"""Tests for function_resolver module (enhancement 2h).

Tests the resolve_function and resolve_functions utilities for loading
callables from module paths with improved error messages.
"""

import os.path

import pytest

from dataknobs_bots.reasoning.function_resolver import (
    resolve_function,
    resolve_functions,
)


class TestResolveFunctionColonFormat:
    """Tests for resolve_function with colon format."""

    def test_resolve_colon_format(self) -> None:
        """Colon format resolves correctly."""
        func = resolve_function("os.path:join")
        assert func is os.path.join

    def test_resolve_nested_module(self) -> None:
        """Resolves functions from nested modules."""
        import json

        func = resolve_function("json:dumps")
        assert func is json.dumps

    def test_resolve_with_underscores(self) -> None:
        """Resolves functions with underscores in names."""
        func = resolve_function("os.path:splitext")
        assert func is os.path.splitext


class TestResolveFunctionDotFormat:
    """Tests for resolve_function with dot format."""

    def test_resolve_dot_format(self) -> None:
        """Dot format resolves correctly (last segment = function)."""
        func = resolve_function("os.path.join")
        assert func is os.path.join

    def test_resolve_dot_format_nested(self) -> None:
        """Dot format works with deeply nested modules."""
        import json

        func = resolve_function("json.dumps")
        assert func is json.dumps


class TestResolveFunctionErrors:
    """Tests for resolve_function error handling."""

    def test_empty_reference_error(self) -> None:
        """Empty reference gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_function("")

        error_msg = str(exc_info.value)
        assert "Empty function reference" in error_msg
        assert "module.path:function_name" in error_msg

    def test_whitespace_only_reference_error(self) -> None:
        """Whitespace-only reference gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_function("   ")

        assert "Empty function reference" in str(exc_info.value)

    def test_no_separator_error(self) -> None:
        """Reference without separator gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_function("just_a_name")

        error_msg = str(exc_info.value)
        assert "just_a_name" in error_msg  # Shows actual reference
        assert "module.path" in error_msg  # Shows expected format

    def test_invalid_colon_format_empty_function(self) -> None:
        """Malformed colon format (empty function) gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_function("module:")

        error_msg = str(exc_info.value)
        assert "module:" in error_msg

    def test_invalid_colon_format_empty_module(self) -> None:
        """Malformed colon format (empty module) gives helpful error."""
        with pytest.raises(ValueError) as exc_info:
            resolve_function(":function")

        error_msg = str(exc_info.value)
        assert ":function" in error_msg

    def test_module_not_found_error(self) -> None:
        """Missing module gives helpful error with context."""
        with pytest.raises(ImportError) as exc_info:
            resolve_function("nonexistent_module_xyz_12345:func")

        error_msg = str(exc_info.value)
        assert "nonexistent_module_xyz_12345" in error_msg
        assert "Cannot import module" in error_msg

    def test_function_not_found_error(self) -> None:
        """Missing function lists available functions."""
        with pytest.raises(AttributeError) as exc_info:
            resolve_function("os.path:nonexistent_func_xyz_12345")

        error_msg = str(exc_info.value)
        assert "nonexistent_func_xyz_12345" in error_msg
        assert "not found" in error_msg
        assert "Available functions:" in error_msg
        # Should list some real os.path functions
        assert "join" in error_msg or "exists" in error_msg

    def test_not_callable_error(self) -> None:
        """Non-callable attribute gives helpful error."""
        # os.path.sep is a string, not callable
        with pytest.raises(ValueError) as exc_info:
            resolve_function("os.path:sep")

        error_msg = str(exc_info.value)
        assert "not callable" in error_msg
        assert "str" in error_msg  # Shows actual type

    def test_whitespace_handling(self) -> None:
        """Whitespace in reference is stripped."""
        func = resolve_function("  os.path:join  ")
        assert func is os.path.join


class TestResolveFunctions:
    """Tests for resolve_functions utility."""

    def test_resolve_string_references(self) -> None:
        """Resolves string references to callables."""
        refs = {
            "join": "os.path:join",
            "exists": "os.path:exists",
        }
        resolved = resolve_functions(refs)

        assert resolved["join"] is os.path.join
        assert resolved["exists"] is os.path.exists

    def test_passthrough_callables(self) -> None:
        """Callable values are passed through unchanged."""

        def my_func() -> None:
            pass

        refs = {
            "my_func": my_func,
            "join": "os.path:join",
        }
        resolved = resolve_functions(refs)

        assert resolved["my_func"] is my_func
        assert resolved["join"] is os.path.join

    def test_mixed_formats(self) -> None:
        """Both colon and dot formats work in same dict."""
        refs = {
            "colon": "os.path:join",
            "dot": "os.path.exists",
        }
        resolved = resolve_functions(refs)

        assert resolved["colon"] is os.path.join
        assert resolved["dot"] is os.path.exists

    def test_invalid_type_error(self) -> None:
        """Invalid reference type raises ValueError."""
        refs = {
            "bad": 123,  # type: ignore[dict-item]
        }

        with pytest.raises(ValueError) as exc_info:
            resolve_functions(refs)

        error_msg = str(exc_info.value)
        assert "bad" in error_msg
        assert "int" in error_msg

    def test_propagates_resolution_errors(self) -> None:
        """Resolution errors are propagated."""
        refs = {
            "bad": "nonexistent.module:func",
        }

        with pytest.raises(ImportError):
            resolve_functions(refs)


class TestWizardLoaderIntegration:
    """Integration tests with WizardConfigLoader."""

    def test_wizard_loader_accepts_colon_format(self) -> None:
        """WizardConfigLoader accepts colon-separated function references."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test",
            "stages": [{"name": "test", "is_start": True, "is_end": True}],
        }

        loader = WizardConfigLoader()
        # Pass custom_functions with colon format
        result = loader.load_from_dict(
            config, custom_functions={"path_joiner": "os.path:join"}
        )

        assert result is not None

    def test_wizard_loader_accepts_dot_format(self) -> None:
        """WizardConfigLoader accepts dot-separated function references."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test",
            "stages": [{"name": "test", "is_start": True, "is_end": True}],
        }

        loader = WizardConfigLoader()
        # Pass custom_functions with dot format
        result = loader.load_from_dict(
            config, custom_functions={"path_joiner": "os.path.join"}
        )

        assert result is not None

    def test_wizard_loader_invalid_function_helpful_error(self) -> None:
        """Invalid function reference gives helpful error."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test",
            "stages": [{"name": "test", "is_start": True, "is_end": True}],
        }

        loader = WizardConfigLoader()

        with pytest.raises(ImportError) as exc_info:
            loader.load_from_dict(
                config, custom_functions={"bad_func": "nonexistent.module:fake_function"}
            )

        # Error should mention the problematic module
        assert "nonexistent" in str(exc_info.value).lower()
