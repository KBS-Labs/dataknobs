"""Safe expression evaluation engine.

Evaluates Python expression strings with restricted globals.
Used by wizard conditions, derivation expressions, and any other
context requiring safe config-authored code evaluation.

The engine wraps expressions in a function body, executes with
controlled globals (``__builtins__`` restricted to a safe allowlist),
and validates the AST to block dunder attribute access (preventing
MRO traversal attacks like ``().__class__.__bases__[0].__subclasses__()``).

Example::

    from dataknobs_common.expressions import (
        safe_eval,
        safe_eval_validate,
        safe_eval_value,
    )

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

    # Pre-check an expression without evaluating it
    assert safe_eval_validate("data.get('a')") is None
    assert safe_eval_validate("().__class__") is not None
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Mapping
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
    "frozenset": frozenset,
    # Collection/numeric functions
    "len": len,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sum": sum,
    "any": any,
    "all": all,
    "sorted": sorted,
    "reversed": reversed,
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

    Also blocks dunder names used as standalone variables, and any
    ``.format()`` / ``.format_map()`` call — the format-spec
    mini-language performs runtime attribute access via ``{N.attr}``
    syntax that bypasses AST-level dunder checks (e.g.
    ``'{0.__class__}'.format(())`` reaches the tuple class). f-strings
    are safe because their substitutions go through normal AST nodes.

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
                    f"Access to dunder attribute '{node.attr}' is not allowed in safe expressions"
                )
            # Block .format() / .format_map() — format-spec attribute
            # access ({N.attr}) bypasses the AST dunder check.
            if node.attr in ("format", "format_map"):
                return (
                    f"Call to '.{node.attr}()' is not allowed in safe "
                    f"expressions (format-spec attribute access "
                    f"bypasses the dunder check). Use an f-string "
                    f"instead — its substitutions go through normal "
                    f"AST validation."
                )
        # Block dunder names as variables: __builtins__, __import__, etc.
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            return f"Access to dunder name '{node.id}' is not allowed in safe expressions"

    return None


#: Matches an expression that already *is* a ``return`` statement.  The
#: word boundary is what separates ``return x`` from ``returned_value`` —
#: a plain prefix test treats the latter as a statement, leaves it
#: unwrapped, and silently evaluates it to ``None``.
_RETURN_STATEMENT = re.compile(r"return\b")


def _to_exec_code(stripped: str) -> str:
    """Wrap a stripped, single-line expression in the function body.

    ``return`` is prepended unless the expression is already a return
    statement.  Shared by :func:`safe_eval` and :func:`safe_eval_validate`
    so the two cannot disagree about what will be parsed.
    """
    if not _RETURN_STATEMENT.match(stripped):
        stripped = f"return {stripped}"
    return f"def _fn():\n    {stripped}\n_result = _fn()"


def safe_eval_validate(
    expression: str,
    *,
    restrict_builtins: bool = True,
) -> str | None:
    """Report why :func:`safe_eval` would refuse an expression.

    Runs the same static pass :func:`safe_eval` runs before it evaluates
    anything, without evaluating anything.  Use it to pre-check a
    config-authored or generated expression while the author is still in
    the build loop, rather than discovering the refusal as a stalled
    condition at run time.

    The contract is definitional: this returns *the reason ``safe_eval``
    would refuse this expression before evaluating it*, or ``None`` if
    ``safe_eval`` would proceed to evaluation.  When the static rules are
    tightened, both answers change together.

    The static pass rejects, in order:

    1. an empty expression;
    2. a multiline expression;
    3. a syntax error (``restrict_builtins=True`` only);
    4. dunder attribute access (``restrict_builtins=True`` only);
    5. a ``.format()`` / ``.format_map()`` call (``restrict_builtins=True``
       only);
    6. a dunder name (``restrict_builtins=True`` only).

    ``None`` is not a safety review, and not a promise the expression will
    succeed.  An expression that reads a missing key is accepted here and
    raises at evaluation — that is the point of the distinction: it is
    "not satisfied yet", not "will not run".  In particular this does not
    check that the expression is free of side effects: ``safe_eval``
    blocks assignment but permits mutation by method call, so
    ``data.update(...)`` is reported as acceptable and will take effect.
    Nor does it check that names resolve.

    Args:
        expression: Python expression string, as it would be passed to
            :func:`safe_eval`.
        restrict_builtins: Must match the value :func:`safe_eval` will be
            called with.  Rules 3-6 live behind the AST pass, which the
            unrestricted path skips.

    Returns:
        The refusal reason, or ``None`` if ``safe_eval`` would evaluate.
        Never raises — an input :func:`safe_eval` would reject outright,
        such as a non-string, is reported as a reason like any other.

    Example::

        reason = safe_eval_validate("data.get('a').__class__")
        if reason is not None:
            raise ValueError(f"unusable condition: {reason}")
    """
    try:
        stripped = expression.strip()
        if not stripped:
            return "Empty expression"

        # Reject multiline expressions — config-authored expressions
        # should be single-line.  Multiline strings could inject
        # module-scope code past the function wrapper.
        if "\n" in stripped:
            return "Multiline expressions are not allowed"

        # The remaining rules are the AST pass, which the unrestricted
        # path does not run.
        if not restrict_builtins:
            return None

        return _validate_ast(_to_exec_code(stripped))

    except Exception as e:
        # safe_eval wraps its whole body in the same guard and degrades a
        # bad input to a failed result rather than raising, so this must
        # too: a pre-check that crashes where evaluation would not is
        # worse than no pre-check.  Reachable from config — an unquoted
        # YAML scalar (``condition: true``) arrives as a bool, which has
        # no .strip(), and a lone surrogate escapes ast.parse as a
        # UnicodeEncodeError rather than a SyntaxError.
        return str(e)


def safe_eval(
    code: str,
    scope: Mapping[str, Any] | None = None,
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
       MRO traversal via ``__class__.__bases__.__subclasses__``),
       dunder names, and ``.format()`` / ``.format_map()`` calls

    The checks that run before evaluation are also available on their
    own — see :func:`safe_eval_validate`, which reports why an
    expression would be refused without evaluating it.

    Args:
        code: Python expression string.  Unless it is already a
            ``return`` statement, ``return`` is prepended
            automatically.  The test is on the ``return`` *token*, so
            an expression starting with a name such as
            ``returned_value`` is treated as an expression.
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
        # The static pass, in callable form.  Consumers can run the same
        # check ahead of time via safe_eval_validate().
        reason = safe_eval_validate(code, restrict_builtins=restrict_builtins)
        if reason is not None:
            return ExpressionResult(
                value=default,
                success=False,
                error=reason,
            )

        if not restrict_builtins:
            logger.warning(
                "safe_eval called with restrict_builtins=False — "
                "full Python builtins available (trusted code only)"
            )

        global_vars: dict[str, Any] = {}
        if restrict_builtins:
            global_vars["__builtins__"] = SAFE_BUILTINS
        global_vars.update(YAML_ALIASES)
        if scope:
            global_vars.update(scope)

        local_vars: dict[str, Any] = {}
        exec_code = _to_exec_code(code.strip())
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
    scope: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper returning just the value.

    Same as ``safe_eval(...).value``.  Suitable for call sites that
    only need the result and handle errors via the default value.
    """
    return safe_eval(code, scope, **kwargs).value
