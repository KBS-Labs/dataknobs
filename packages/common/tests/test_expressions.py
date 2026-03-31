"""Tests for the safe expression evaluation engine.

Covers core functionality, security restrictions, and edge cases.
"""

from typing import Any

import pytest

from dataknobs_common.expressions import (
    ExpressionResult,
    SAFE_BUILTINS,
    YAML_ALIASES,
    safe_eval,
    safe_eval_value,
)


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


class TestCoreFunctionality:
    """Verify basic expression evaluation."""

    def test_simple_expression(self) -> None:
        result = safe_eval("1 + 2")
        assert result == ExpressionResult(value=3, success=True)

    def test_string_expression(self) -> None:
        result = safe_eval("'hello'.upper()")
        assert result.value == "HELLO"
        assert result.success is True

    def test_scope_variables(self) -> None:
        result = safe_eval("x + y", scope={"x": 1, "y": 2})
        assert result.value == 3

    def test_return_prefix_auto(self) -> None:
        result = safe_eval("42")
        assert result.value == 42

    def test_return_prefix_existing(self) -> None:
        result = safe_eval("return 42")
        assert result.value == 42

    def test_yaml_aliases(self) -> None:
        assert safe_eval("true").value is True
        assert safe_eval("false").value is False
        assert safe_eval("null").value is None
        assert safe_eval("none").value is None

    def test_safe_builtins_available(self) -> None:
        result = safe_eval("len([1, 2, 3])")
        assert result.value == 3

    def test_safe_builtins_types(self) -> None:
        assert safe_eval("int('42')").value == 42
        assert safe_eval("float('3.14')").value == 3.14
        assert safe_eval("str(42)").value == "42"
        assert safe_eval("bool(1)").value is True

    def test_safe_builtins_collections(self) -> None:
        assert safe_eval("sorted([3, 1, 2])").value == [1, 2, 3]
        assert safe_eval("min(5, 3, 8)").value == 3
        assert safe_eval("max(5, 3, 8)").value == 8
        assert safe_eval("abs(-7)").value == 7

    def test_coerce_bool_true(self) -> None:
        result = safe_eval("42", coerce_bool=True)
        assert result.value is True

    def test_coerce_bool_false(self) -> None:
        result = safe_eval("0", coerce_bool=True)
        assert result.value is False

    def test_coerce_bool_none(self) -> None:
        result = safe_eval("None", coerce_bool=True)
        assert result.value is False

    def test_default_on_failure(self) -> None:
        result = safe_eval("1/0", default=-1)
        assert result.value == -1
        assert result.success is False
        assert result.error is not None

    def test_default_bool_on_failure(self) -> None:
        result = safe_eval("1/0", coerce_bool=True, default=False)
        assert result.value is False
        assert result.success is False

    def test_safe_eval_value_convenience(self) -> None:
        assert safe_eval_value("1 + 2") == 3

    def test_safe_eval_value_with_default(self) -> None:
        assert safe_eval_value("1/0", default=-1) == -1

    def test_isinstance_available(self) -> None:
        result = safe_eval(
            "isinstance(value, list)",
            scope={"value": [1, 2]},
        )
        assert result.value is True

    def test_list_comprehension(self) -> None:
        result = safe_eval(
            "[x.lower() for x in value]",
            scope={"value": ["A", "B", "C"]},
        )
        assert result.value == ["a", "b", "c"]

    def test_dict_get(self) -> None:
        result = safe_eval(
            "{'easy': 30, 'hard': 120}.get(value, 60)",
            scope={"value": "hard"},
        )
        assert result.value == 120

    def test_ternary_expression(self) -> None:
        result = safe_eval(
            "'yes' if value > 5 else 'no'",
            scope={"value": 10},
        )
        assert result.value == "yes"

    def test_none_result(self) -> None:
        result = safe_eval("None")
        assert result == ExpressionResult(value=None, success=True)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    """Verify dangerous operations are blocked."""

    def test_no_import(self) -> None:
        result = safe_eval("__import__('os')")
        assert result.success is False
        assert "NameError" in (result.error or "") or "name" in (result.error or "")

    def test_no_open(self) -> None:
        result = safe_eval("open('/etc/passwd')")
        assert result.success is False

    def test_no_exec(self) -> None:
        result = safe_eval("exec('x=1')")
        assert result.success is False

    def test_no_eval(self) -> None:
        result = safe_eval("eval('1+1')")
        assert result.success is False

    def test_no_getattr(self) -> None:
        result = safe_eval("getattr([], '__class__')")
        assert result.success is False

    def test_no_globals(self) -> None:
        result = safe_eval("globals()")
        assert result.success is False

    def test_no_locals(self) -> None:
        result = safe_eval("locals()")
        assert result.success is False

    def test_no_compile(self) -> None:
        result = safe_eval("compile('1+1', '<>', 'eval')")
        assert result.success is False

    def test_no_mro_traversal_class(self) -> None:
        """Block MRO traversal via __class__."""
        result = safe_eval("().__class__")
        assert result.success is False
        assert "dunder attribute" in (result.error or "")

    def test_no_mro_traversal_bases(self) -> None:
        """Block MRO traversal via __bases__."""
        result = safe_eval("().__class__.__bases__")
        assert result.success is False

    def test_no_mro_traversal_subclasses(self) -> None:
        """Block full MRO chain to __subclasses__."""
        result = safe_eval(
            "().__class__.__bases__[0].__subclasses__()"
        )
        assert result.success is False

    def test_no_mro_popen_attack(self) -> None:
        """Block the classic Popen sandbox escape."""
        result = safe_eval(
            "[c for c in ().__class__.__bases__[0].__subclasses__() "
            "if c.__name__ == 'Popen']"
        )
        assert result.success is False

    def test_no_dunder_name(self) -> None:
        """Block dunder names as standalone variables."""
        result = safe_eval("__builtins__")
        assert result.success is False

    def test_scope_does_not_leak(self) -> None:
        """Scope variables don't leak into the caller's namespace."""
        safe_eval("x", scope={"x": 1})
        # If x leaked, this would not raise; but it should not exist
        with pytest.raises(NameError):
            _ = x  # type: ignore[name-defined]  # noqa: F821

    def test_restrict_builtins_false(self) -> None:
        """With restrict_builtins=False, full Python builtins available."""
        result = safe_eval("len([1])", restrict_builtins=False)
        assert result.value == 1

    def test_restrict_builtins_false_skips_ast_check(self) -> None:
        """When restrict_builtins=False, dunder access is allowed (trusted code)."""
        result = safe_eval("().__class__.__name__", restrict_builtins=False)
        assert result.value == "tuple"

    def test_no_multiline_expressions(self) -> None:
        """Block multiline expressions to prevent module-scope injection."""
        result = safe_eval("1 + 2\nimport os")
        assert result.success is False
        assert "Multiline" in (result.error or "")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify edge case handling."""

    def test_empty_code(self) -> None:
        result = safe_eval("")
        assert result.success is False

    def test_whitespace_only(self) -> None:
        result = safe_eval("   ")
        assert result.success is False

    def test_scope_overrides_yaml_aliases(self) -> None:
        result = safe_eval("true", scope={"true": 42})
        assert result.value == 42

    def test_scope_overrides_builtins_key(self) -> None:
        """Scope-provided values take precedence over YAML aliases."""
        result = safe_eval("none", scope={"none": "custom"})
        assert result.value == "custom"

    def test_default_value_none(self) -> None:
        result = safe_eval("1/0")
        assert result.value is None  # default is None

    def test_result_dataclass_frozen(self) -> None:
        result = ExpressionResult(value=42, success=True)
        with pytest.raises(AttributeError):
            result.value = 99  # type: ignore[misc]

    def test_has_helper_pattern(self) -> None:
        """Common pattern: has() helper as a scope callable."""
        data: dict[str, Any] = {"name": "test", "count": 0}
        has = lambda key: data.get(key) is not None  # noqa: E731
        result = safe_eval(
            "has('name') and has('count')",
            scope={"data": data, "has": has},
            coerce_bool=True,
        )
        assert result.value is True

    def test_has_helper_missing_key(self) -> None:
        data: dict[str, Any] = {"name": "test"}
        has = lambda key: data.get(key) is not None  # noqa: E731
        result = safe_eval(
            "has('missing')",
            scope={"data": data, "has": has},
            coerce_bool=True,
        )
        assert result.value is False
