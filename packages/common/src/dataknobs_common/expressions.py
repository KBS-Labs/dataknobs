"""Safe expression evaluation engine.

Evaluates Python expression strings with restricted globals.
Used by wizard conditions, derivation expressions, and any other
context requiring safe config-authored code evaluation.

The engine wraps expressions in a function body, executes with
controlled globals (``__builtins__`` restricted to a safe allowlist),
and validates the AST to block dunder attribute access (preventing
MRO traversal attacks like ``().__class__.__bases__[0].__subclasses__()``).

Example::

    from dataknobs_common.expressions import safe_eval, safe_eval_value

    # Simple expression with scope variables
    result = safe_eval("x + y", scope={"x": 1, "y": 2})
    assert result.value == 3

    # Condition evaluation with bool coercion
    ok = safe_eval_value(
        "data.get('count', 0) > 5",
        scope={"data": {"count": 10}},
        coerce_bool=True,
    )
    assert ok is True

    # Expression returning native type
    val = safe_eval_value(
        "{'easy': 30, 'hard': 120}.get(value, 60)",
        scope={"value": "hard"},
    )
    assert val == 120
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Safe builtins allowlist ──

SAFE_BUILTINS: dict[str, Any] = {
    # Type constructors
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    # Collection/numeric functions
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "isinstance": isinstance,
    "enumerate": enumerate,
    "range": range,
    "zip": zip,
    # Constants
    "True": True,
    "False": False,
    "None": None,
}
"""Builtins allowlist shared by all expression contexts.

Explicitly excludes: ``exec``, ``eval``, ``__import__``, ``open``,
``getattr``, ``setattr``, ``delattr``, ``globals``, ``locals``,
``compile``, ``breakpoint``, ``__builtins__`` passthrough.
"""


# ── YAML/JSON literal aliases ──

YAML_ALIASES: dict[str, Any] = {
    "true": True,
    "false": False,
    "null": None,
    "none": None,
}
"""Common aliases for YAML/JSON boolean and null literals.

Included in expression scope so that config-authored expressions
can use ``true``/``false``/``null`` (YAML style) alongside Python's
``True``/``False``/``None``.

Note: scope variables with the same name override these aliases
(scope is applied after YAML_ALIASES).
"""


@dataclass(frozen=True)
class ExpressionResult:
    """Result of a safe expression evaluation.

    Attributes:
        value: The evaluated result (native Python type).
        success: Whether evaluation succeeded.
        error: Exception message if evaluation failed.
    """

    value: Any = None
    success: bool = True
    error: str | None = None


def _validate_ast(code: str) -> str | None:
    """Check the expression AST for unsafe attribute access.

    Blocks dunder attribute access (``__class__``, ``__bases__``,
    ``__subclasses__``, etc.) which can be used for MRO traversal
    to escape the restricted builtins sandbox.

    Also blocks dunder names used as standalone variables.

    Returns:
        Error message if unsafe access detected, ``None`` if safe.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block dunder attribute access: obj.__class__, obj.__bases__, etc.
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return (
                    f"Access to dunder attribute '{node.attr}' is not "
                    f"allowed in safe expressions"
                )
        # Block dunder names as variables: __builtins__, __import__, etc.
        if isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                return (
                    f"Access to dunder name '{node.id}' is not "
                    f"allowed in safe expressions"
                )

    return None


def safe_eval(
    code: str,
    scope: dict[str, Any] | None = None,
    *,
    coerce_bool: bool = False,
    restrict_builtins: bool = True,
    default: Any = None,
) -> ExpressionResult:
    """Evaluate a Python expression string safely.

    Wraps the expression in a function body, executes with restricted
    globals, and returns the native result.  This is the shared core
    used by wizard conditions and derivation expressions.

    Security model (when ``restrict_builtins=True``):

    1. ``__builtins__`` restricted to ``SAFE_BUILTINS`` (blocks
       ``exec``, ``eval``, ``__import__``, ``open``, etc.)
    2. AST validation blocks dunder attribute access (prevents
       MRO traversal via ``__class__.__bases__.__subclasses__``)

    Args:
        code: Python expression string.  If it does not start with
            ``return``, one is prepended automatically.
        scope: Variables available in the expression.  Merged on top
            of ``SAFE_BUILTINS`` and ``YAML_ALIASES``.  Callers
            provide context-specific variables here (e.g., ``data``,
            ``value``, ``has()``, ``bank``).  Scope variables with
            the same name as YAML aliases override the alias.
        coerce_bool: If True, coerce the result to ``bool`` (for
            condition evaluation).  If False, return native type.
        restrict_builtins: If True (default), set ``__builtins__``
            to ``SAFE_BUILTINS`` and validate AST for unsafe access,
            blocking ``exec``, ``eval``, ``__import__``, ``open``,
            and MRO traversal.  If False, use Python's default
            builtins and skip AST validation (for trusted code only).
        default: Value to return on evaluation failure.  Defaults
            to ``None``.  For condition evaluation, callers typically
            pass ``default=False``.

    Returns:
        ExpressionResult with the evaluated value and success status.
    """
    try:
        stripped = code.strip()
        if not stripped:
            return ExpressionResult(
                value=default,
                success=False,
                error="Empty expression",
            )

        if not stripped.startswith("return"):
            stripped = f"return {stripped}"

        # AST validation: block dunder access when builtins restricted
        if restrict_builtins:
            exec_code = f"def _fn():\n    {stripped}\n_result = _fn()"
            ast_error = _validate_ast(exec_code)
            if ast_error:
                return ExpressionResult(
                    value=default,
                    success=False,
                    error=ast_error,
                )

        global_vars: dict[str, Any] = {}
        if restrict_builtins:
            global_vars["__builtins__"] = SAFE_BUILTINS
        global_vars.update(YAML_ALIASES)
        if scope:
            global_vars.update(scope)

        local_vars: dict[str, Any] = {}
        exec_code = f"def _fn():\n    {stripped}\n_result = _fn()"
        exec(exec_code, global_vars, local_vars)  # nosec B102

        result = local_vars.get("_result", default)
        if coerce_bool:
            result = bool(result)

        return ExpressionResult(value=result, success=True)

    except Exception as e:
        return ExpressionResult(
            value=default,
            success=False,
            error=str(e),
        )


def safe_eval_value(
    code: str,
    scope: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper returning just the value.

    Same as ``safe_eval(...).value``.  Suitable for call sites that
    only need the result and handle errors via the default value.
    """
    return safe_eval(code, scope, **kwargs).value
