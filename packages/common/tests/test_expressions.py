"""Tests for the safe expression evaluation engine.

Covers core functionality, security restrictions, and edge cases.
"""

from typing import Any

import pytest

from dataknobs_common.expressions import (
    ExpressionResult,
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

    def test_any_over_iterable(self) -> None:
        """``any()`` is in the safe builtin allowlist."""
        result = safe_eval(
            "any(data.get(k) for k in ['a', 'b', 'c'])",
            scope={"data": {"a": False, "b": True, "c": False}},
        )
        assert result.value is True

    def test_any_returns_false_when_all_falsy(self) -> None:
        result = safe_eval(
            "any(data.get(k) for k in ['a', 'b'])",
            scope={"data": {"a": False}},  # 'b' missing → None
        )
        assert result.value is False

    def test_all_over_iterable(self) -> None:
        """``all()`` is in the safe builtin allowlist."""
        result = safe_eval(
            "all(data.get(k) for k in ['a', 'b'])",
            scope={"data": {"a": True, "b": True}},
        )
        assert result.value is True

    def test_all_returns_false_when_any_falsy(self) -> None:
        result = safe_eval(
            "all(data.get(k) for k in ['a', 'b'])",
            scope={"data": {"a": True, "b": False}},
        )
        assert result.value is False

    def test_sum_over_iterable(self) -> None:
        """``sum()`` aggregates numeric iterables."""
        result = safe_eval(
            "sum(data.get(k, 0) for k in ['a', 'b', 'c'])",
            scope={"data": {"a": 10, "b": 20, "c": 30}},
        )
        assert result.value == 60

    def test_sum_counting_pattern(self) -> None:
        """Common derivation pattern: count truthy entries."""
        result = safe_eval(
            "sum(1 for x in items if x > 5)",
            scope={"items": [1, 6, 3, 10, 2, 8]},
        )
        assert result.value == 3

    def test_reversed_iterator(self) -> None:
        """``reversed()`` returns a reverse iterator."""
        result = safe_eval(
            "list(reversed(items))",
            scope={"items": [1, 2, 3]},
        )
        assert result.value == [3, 2, 1]

    def test_frozenset_membership(self) -> None:
        """``frozenset()`` constructs an immutable set for membership."""
        result = safe_eval(
            "value in frozenset(['accept', 'decline'])",
            scope={"value": "accept"},
        )
        assert result.value is True


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

    def test_no_format_spec_attribute_escape(self) -> None:
        """Block the format-spec attribute escape: ``{N.attr}`` syntax
        in ``str.format()`` performs runtime attribute access that
        bypasses the AST-level dunder check.
        """
        result = safe_eval("'{0.__class__}'.format(())")
        assert result.success is False
        assert "format" in (result.error or "")

    def test_no_format_spec_mro_chain(self) -> None:
        """The format-spec escape would chain to ``__subclasses__`` for
        a full sandbox escape — block at the ``.format()`` call.
        """
        result = safe_eval(
            "'{0.__class__.__bases__[0].__subclasses__()}'.format(())"
        )
        assert result.success is False

    def test_no_format_map_attribute_escape(self) -> None:
        """``.format_map()`` has the same format-spec vulnerability."""
        result = safe_eval(
            "'{x.__class__}'.format_map({'x': ()})"
        )
        assert result.success is False

    def test_no_format_via_walrus_aliased_method(self) -> None:
        """Aliasing ``.format`` through a walrus expression must still
        produce an ``ast.Attribute(attr='format')`` node, so the
        block fires. Verifies the AST walk doesn't depend on the
        method being the immediate call target.
        """
        result = safe_eval(
            "(f := ''.format)('{0.__class__}', ())",
        )
        assert result.success is False
        assert "format" in (result.error or "")

    def test_no_format_via_walrus_aliased_method_map(self) -> None:
        """Same for ``.format_map`` aliased via walrus."""
        result = safe_eval(
            "(f := ''.format_map)({'x': ()})",
        )
        assert result.success is False
        assert "format_map" in (result.error or "")

    def test_format_block_is_method_name_scoped(self) -> None:
        """The ``.format()`` block must match the method name on ANY
        attribute access — there is no carve-out for arbitrary
        objects that happen to have a ``.format()`` method. Pin this
        as a deliberate trade-off: config-authored expressions that
        need string formatting use f-strings (which go through normal
        AST validation), not ``.format()``.
        """
        # An object with its own .format() method is still blocked.
        class _Custom:
            def format(self) -> str:  # noqa: D401 - test fixture
                return "formatted"

        result = safe_eval("obj.format()", scope={"obj": _Custom()})
        assert result.success is False
        assert "format" in (result.error or "")

    def test_format_allowed_when_restrict_builtins_false(self) -> None:
        """``restrict_builtins=False`` is the trusted-code escape hatch:
        the AST check (including the ``.format()`` block) is skipped,
        and the expression runs with full Python builtins. This is the
        documented behavior — pin it so an over-zealous tightening of
        the AST validator doesn't silently break trusted-code call
        sites.
        """
        result = safe_eval(
            "'value is {}'.format(42)",
            restrict_builtins=False,
        )
        assert result.success is True
        assert result.value == "value is 42"

    def test_fstring_dunder_still_blocked(self) -> None:
        """f-strings substitutions go through normal AST validation,
        so dunder access is blocked the same way as bare attribute access.
        """
        result = safe_eval(
            "f'{x.__class__}'",
            scope={"x": ()},
        )
        assert result.success is False
        assert "dunder" in (result.error or "")

    def test_legitimate_fstring_still_works(self) -> None:
        """f-string formatting of in-scope values without dunder access
        is the safe replacement for ``.format()``.
        """
        result = safe_eval(
            "f'value is {x}'",
            scope={"x": 42},
        )
        assert result.success is True
        assert result.value == "value is 42"

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
