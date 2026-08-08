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
from dataknobs_common.exceptions import DottedPathError, DottedPathReason


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
    """Error handling for ``resolve_function``.

    Every case here used to assert one of three stdlib exception types —
    ``ValueError`` for a malformed reference, ``ImportError`` for a missing
    module, ``AttributeError`` for a missing function. All three are
    ``DottedPathError`` now, with the distinction moved to ``reason``, so a
    caller can catch one type and still branch on which fault it was.

    That mattered because three types meant three ``except`` clauses at every
    call site, and a site that listed two of them turned the third into a
    crash. The wizard's own hook loader caught all three; the task-injection
    loader caught bare ``Exception`` to be sure of covering them.
    """

    @pytest.mark.parametrize(
        ("ref", "reason", "expected_text"),
        [
            ("", DottedPathReason.MALFORMED, "Expected a dotted path"),
            ("   ", DottedPathReason.MALFORMED, "Expected a dotted path"),
            ("just_a_name", DottedPathReason.MALFORMED, "just_a_name"),
            ("module:", DottedPathReason.MALFORMED, "module:"),
            (":function", DottedPathReason.MALFORMED, ":function"),
            (
                "nonexistent_module_xyz_12345:func",
                DottedPathReason.MODULE_NOT_FOUND,
                "nonexistent_module_xyz_12345",
            ),
            (
                "os.path:nonexistent_func_xyz_12345",
                DottedPathReason.ATTRIBUTE_NOT_FOUND,
                "nonexistent_func_xyz_12345",
            ),
            ("os.path:sep", DottedPathReason.NOT_CALLABLE, "not callable"),
        ],
        ids=["empty", "whitespace", "no-separator", "empty-function",
             "empty-module", "missing-module", "missing-function",
             "not-callable"],
    )
    def test_every_fault_is_one_type_with_a_distinct_reason(
        self, ref: str, reason: DottedPathReason, expected_text: str
    ) -> None:
        with pytest.raises(DottedPathError) as exc_info:
            resolve_function(ref)

        assert exc_info.value.reason is reason
        assert expected_text in str(exc_info.value)

    def test_the_error_still_names_the_reference(self) -> None:
        """Carried on the exception now, not only interpolated into the text."""
        with pytest.raises(DottedPathError) as exc_info:
            resolve_function("nonexistent_module_xyz_12345:func")

        assert exc_info.value.ref == "nonexistent_module_xyz_12345:func"

    def test_a_missing_function_still_suggests_what_the_module_has(self) -> None:
        """The one thing worth keeping from the implementation this replaced.

        Its ``AttributeError`` enumerated the module's public callables, which
        is the most useful part of a missing-attribute message and the reason
        this resolver's error was better than the other eight. It moved into
        the shared primitive rather than being lost with the function.
        """
        with pytest.raises(DottedPathError) as exc_info:
            resolve_function("os.path:nonexistent_func_xyz_12345")

        error_msg = str(exc_info.value)
        assert "Available:" in error_msg
        assert "join" in error_msg or "exists" in error_msg

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

        with pytest.raises(DottedPathError):
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

        with pytest.raises(DottedPathError) as exc_info:
            loader.load_from_dict(
                config, custom_functions={"bad_func": "nonexistent.module:fake_function"}
            )

        # Error should mention the problematic module
        assert "nonexistent" in str(exc_info.value).lower()
