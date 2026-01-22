"""Tests for function_resolver module."""

import pytest

from dataknobs_bots.reasoning.function_resolver import (
    resolve_function,
    resolve_functions,
)


class TestResolveFunction:
    """Tests for resolve_function."""

    def test_resolve_builtin_function(self) -> None:
        """Test resolving a function from a standard library module."""
        # Resolve os.path.exists
        func = resolve_function("os.path:exists")
        assert func is not None
        assert callable(func)

    def test_resolve_builtin_class(self) -> None:
        """Test resolving a class from standard library."""
        # Resolve pathlib.Path
        cls = resolve_function("pathlib:Path")
        assert cls is not None
        assert callable(cls)

    def test_resolve_dataknobs_function(self) -> None:
        """Test resolving a function from dataknobs."""
        # Resolve a known function from the same package
        func = resolve_function(
            "dataknobs_bots.reasoning.function_resolver:resolve_function"
        )
        assert func is not None
        assert func is resolve_function

    def test_resolve_empty_string(self) -> None:
        """Test that empty string returns None."""
        result = resolve_function("")
        assert result is None

    def test_resolve_none(self) -> None:
        """Test that None-ish values return None."""
        # Empty string
        assert resolve_function("") is None

    def test_resolve_invalid_format_no_colon(self) -> None:
        """Test that invalid format without colon returns None."""
        result = resolve_function("os.path.exists")
        assert result is None

    def test_resolve_nonexistent_module(self) -> None:
        """Test that nonexistent module returns None."""
        result = resolve_function("nonexistent_module_xyz:some_func")
        assert result is None

    def test_resolve_nonexistent_function(self) -> None:
        """Test that nonexistent function in valid module returns None."""
        result = resolve_function("os.path:nonexistent_function_xyz")
        assert result is None

    def test_resolve_nested_module(self) -> None:
        """Test resolving from deeply nested module."""
        func = resolve_function("dataknobs_bots.reasoning.wizard_loader:WizardConfigLoader")
        assert func is not None

    def test_resolve_with_multiple_colons_uses_last(self) -> None:
        """Test that multiple colons splits on the last one."""
        # This is an edge case - module:submodule:func would try to import
        # "module:submodule" which would fail
        result = resolve_function("invalid:module:format")
        assert result is None


class TestResolveFunctions:
    """Tests for resolve_functions."""

    def test_resolve_empty_dict(self) -> None:
        """Test resolving empty dict returns empty dict."""
        result = resolve_functions({})
        assert result == {}

    def test_resolve_all_strings(self) -> None:
        """Test resolving dict with all string references."""
        refs = {
            "path_exists": "os.path:exists",
            "path_join": "os.path:join",
        }

        result = resolve_functions(refs)

        assert "path_exists" in result
        assert "path_join" in result
        assert callable(result["path_exists"])
        assert callable(result["path_join"])

    def test_resolve_all_callables(self) -> None:
        """Test resolving dict with all callable values."""

        def my_func() -> str:
            return "hello"

        def my_other_func() -> int:
            return 42

        refs = {
            "func1": my_func,
            "func2": my_other_func,
        }

        result = resolve_functions(refs)

        assert result["func1"] is my_func
        assert result["func2"] is my_other_func

    def test_resolve_mixed_strings_and_callables(self) -> None:
        """Test resolving dict with mixed string refs and callables."""

        def my_callable() -> None:
            pass

        refs = {
            "builtin": "os.path:exists",
            "custom": my_callable,
        }

        result = resolve_functions(refs)

        assert "builtin" in result
        assert callable(result["builtin"])
        assert result["custom"] is my_callable

    def test_skips_unresolvable_strings(self) -> None:
        """Test that unresolvable strings are skipped."""
        refs = {
            "valid": "os.path:exists",
            "invalid": "nonexistent_module:func",
        }

        result = resolve_functions(refs)

        assert "valid" in result
        assert "invalid" not in result

    def test_skips_invalid_types(self) -> None:
        """Test that invalid types are skipped."""
        refs: dict = {
            "valid": "os.path:exists",
            "number": 42,
            "list": ["not", "callable"],
        }

        result = resolve_functions(refs)

        assert "valid" in result
        assert "number" not in result
        assert "list" not in result

    def test_preserves_callable_identity(self) -> None:
        """Test that callables are not modified."""

        def original_func(x: int) -> int:
            return x * 2

        refs = {"func": original_func}
        result = resolve_functions(refs)

        # Should be the exact same function object
        assert result["func"] is original_func
        assert result["func"](5) == 10


class TestIntegrationWithWizardLoader:
    """Integration tests with WizardConfigLoader."""

    def test_loader_accepts_string_functions(self) -> None:
        """Test that WizardConfigLoader accepts string function references."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        # Define a simple callable for testing
        transform_called = []

        def test_transform(data: dict, context: object = None) -> dict:
            transform_called.append(True)
            return data

        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Test",
                }
            ],
        }

        loader = WizardConfigLoader()

        # Test with string reference (uses a known function)
        wizard_fsm = loader.load_from_dict(
            config,
            custom_functions={
                # Use a known function that can be resolved
                "exists": "os.path:exists",
                # Also include a real callable
                "transform": test_transform,
            },
        )

        assert wizard_fsm is not None

    def test_loader_handles_unresolvable_gracefully(self) -> None:
        """Test that loader handles unresolvable functions gracefully."""
        from dataknobs_bots.reasoning.wizard_loader import WizardConfigLoader

        config = {
            "name": "test-wizard",
            "stages": [
                {
                    "name": "start",
                    "is_start": True,
                    "is_end": True,
                    "prompt": "Test",
                }
            ],
        }

        loader = WizardConfigLoader()

        # Should not raise, just skip the unresolvable function
        wizard_fsm = loader.load_from_dict(
            config,
            custom_functions={
                "invalid": "nonexistent_module:nonexistent_func",
            },
        )

        assert wizard_fsm is not None
