"""Every ContextAwareTool answers an omitted required argument.

``ContextAwareTool.execute`` forwards the model's arguments into
``execute_with_context``. A subclass declaring a parameter ``required``
in its schema but not defaulting it in its signature turns a routine
omission into a ``TypeError`` raised by the call itself, and the model
gets a Python binding message where it needed something to retry from.

Nine tools in this package carried that shape. The base class now
checks the declared-required set before forwarding, so the defect is
fixed for all of them at once; these tests are the guard that keeps a
tenth from reintroducing it.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import pytest

from dataknobs_llm.tools.context_aware import ContextAwareTool

SRC_ROOTS = (
    pathlib.Path(__file__).resolve().parents[3] / "src",
    pathlib.Path(__file__).resolve().parents[4] / "llm" / "src",
)


def _context_aware_tool_classes() -> list[type[ContextAwareTool]]:
    """Import every concrete ContextAwareTool subclass in bots and llm."""
    import importlib

    found: list[type[ContextAwareTool]] = []
    for root in SRC_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text())
            names = [
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and "ContextAwareTool" in [ast.unparse(b) for b in node.bases]
            ]
            if not names:
                continue
            module_name = ".".join(path.relative_to(root).with_suffix("").parts)
            module = importlib.import_module(module_name)
            for name in names:
                cls = getattr(module, name)
                if not inspect.isabstract(cls):
                    found.append(cls)
    return found


TOOL_CLASSES = _context_aware_tool_classes()


def test_the_sweep_found_the_tools() -> None:
    """Guard the guard: an import or glob regression must not pass silently."""
    assert len(TOOL_CLASSES) >= 25


@pytest.mark.parametrize("cls", TOOL_CLASSES, ids=lambda c: c.__name__)
def test_declared_required_parameters_are_defaulted(cls: type[ContextAwareTool]) -> None:
    """A declared-required parameter must still have a signature default.

    Without one the override is incompatible with the base signature --
    which is what the type checker reports as ``[override]`` -- and the
    base class cannot reach its own guard, because binding the call
    fails before the body runs.
    """
    signature = inspect.signature(cls.execute_with_context)
    required_without_default = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        and name not in ("self", "context")
    ]
    assert required_without_default == [], (
        f"{cls.__name__}.execute_with_context has non-defaulted parameter(s) "
        f"{required_without_default}; an LLM omitting one raises TypeError "
        f"before the base class can report it"
    )


@pytest.mark.parametrize("cls", TOOL_CLASSES, ids=lambda c: c.__name__)
def test_schema_required_names_are_real_parameters(cls: type[ContextAwareTool]) -> None:
    """The guard reads ``schema["required"]``, so the names must be real.

    A name declared required but absent from the signature would be
    reported missing forever, since nothing can supply it.
    """
    schema = _schema_of(cls)
    if schema is None:
        pytest.skip(f"{cls.__name__}.schema needs an instance to evaluate")
    required = schema.get("required") or []
    parameters = inspect.signature(cls.execute_with_context).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    for name in required:
        assert name in parameters or accepts_kwargs, (
            f"{cls.__name__}.schema declares {name!r} required but "
            f"execute_with_context cannot receive it"
        )
        assert name in (schema.get("properties") or {}), (
            f"{cls.__name__}.schema declares {name!r} required but does not "
            f"describe it under 'properties'"
        )


def _schema_of(cls: type[ContextAwareTool]) -> dict[str, Any] | None:
    """Return the class's declared schema, read from its source.

    The property is read statically rather than called: several schemas
    are literals on a class whose constructor needs live dependencies,
    and a test that had to build those would be testing them instead.
    Anything not a literal returns None and the caller skips.
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):
        return None
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name != "schema":
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                try:
                    value = ast.literal_eval(statement.value)
                except ValueError:
                    return None
                return value if isinstance(value, dict) else None
    return None
