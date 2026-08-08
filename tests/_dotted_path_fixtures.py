"""Import targets for the dotted-path agreement guard.

Every entry point in ``test_dotted_path_agreement.py`` resolves against this
module, so the four cases (``:`` path, ``.`` path, missing module, missing
attribute) mean the same thing for all of them.

A real importable module rather than a ``tmp_path`` one written per test: the
thing under test *is* ``importlib.import_module``, and a fixture that has to
manipulate ``sys.path`` to be importable tests the manipulation as much as the
resolver. This module is reachable as
``tests._dotted_path_fixtures`` from the repo root, the same way
``tests._workspace`` already is.

Nothing here is imported by production code, and nothing here should acquire a
dependency that makes importing it expensive — several tests import it as a
side effect of resolving a path through it, which is the point.
"""

from __future__ import annotations

from typing import Any

from dataknobs_bots.middleware.base import Middleware
from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceMetrics,
)
from dataknobs_llm.tools import Tool
from dataknobs_xization.chunking.base import Chunker

# ── Instantiation counter ─────────────────────────────────────────────
#
# `resolve_class` returns the class and lets the caller instantiate, so that
# validate-before-instantiate is the only order it can express. That property
# is invisible to a test that only checks the return value — a resolver that
# instantiated, checked, threw the instance away and returned the class would
# pass every behavioral assertion. The counter is what makes it observable.


class _InstantiationCounter:
    """A count that lives in an attribute rather than a module global.

    Rebinding a module-level ``int`` would work, but the name is exported,
    and a rebound global is invisible to anyone who imported the *name*
    instead of the module: ``from tests._dotted_path_fixtures import
    instantiations`` would freeze at whatever the value was on import and
    never move again, so a reader asserting on it would assert on a
    constant. An object with a field cannot be read that way by accident.
    """

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def bump(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


instantiations = _InstantiationCounter()


def reset_instantiations() -> None:
    """Zero the counter. Call in a fixture, not inline — see the guard."""
    instantiations.reset()


class _Counted:
    """Base that records every construction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        instantiations.bump()


# ── Resolvable targets ────────────────────────────────────────────────


def resolvable_function(*args: Any, **kwargs: Any) -> str:
    """A plain module-level callable. The target for the callable resolvers."""
    return "resolved"


async def resolvable_async_function(*args: Any, **kwargs: Any) -> str:
    """An async callable — `callable()` is true for these too."""
    return "resolved"


not_callable = 42
"""A module attribute that resolves but is not callable."""


class ConformingMergeFilter(_Counted):
    """Satisfies ``wizard_types.MergeFilter`` (a method-only Protocol)."""

    def filter(self, *args: Any, **kwargs: Any) -> Any:
        return None


class ConformingFieldTransform(_Counted):
    """Satisfies ``wizard_derivations.FieldTransform``."""

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        return None


class BareClass(_Counted):
    """Conforms to nothing. Resolves fine; fails every shape check.

    Counted like the others deliberately: the assertion that a wrong-shape
    target is rejected *without being constructed* needs this class to be
    able to report that it was constructed.
    """


# ── Targets for the nominal-base entry points ─────────────────────────
#
# `MergeFilter` and `FieldTransform` above are runtime-checkable Protocols, so
# conforming to them costs nothing and this module stays dependency-free for
# them. `Middleware`, `Tool` and `Chunker` are real base classes checked with
# `issubclass`, so their targets must inherit — which is the only reason this
# module imports from `bots`, `llm` and `xization` at all. `IResourceProvider`
# is a runtime-checkable Protocol, but its five members are easiest to satisfy
# by subclassing `fsm`'s own base, so that import is here for convenience
# rather than necessity.
#
# Worth every one of them: leaving these rows out would exempt the entry
# points with the most consumer-visible config keys (`middleware:`, tool
# `class:`, `chunker:`, resource `class:`) from the guard, which is the
# population the guard exists for.


class ConformingMiddleware(_Counted, Middleware):
    """Subclasses the bot-turn ``Middleware`` base."""


class ConformingTool(_Counted, Tool):
    """Subclasses the LLM ``Tool`` ABC, implementing both abstract members."""

    name = "fixture_tool"

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return None


class ConformingChunker(_Counted, Chunker):
    """Subclasses the ``xization`` ``Chunker`` ABC, implementing ``chunk``."""

    def chunk(self, content: str, **kwargs: Any) -> list[Any]:
        return []


class ConformingResourceProvider(_Counted, BaseResourceProvider):
    """Satisfies ``fsm``'s ``IResourceProvider`` runtime-checkable Protocol."""

    def __init__(self, name: str = "fixture", **kwargs: Any) -> None:
        BaseResourceProvider.__init__(self, name)
        _Counted.__init__(self)

    def acquire(self, **kwargs: Any) -> Any:
        return object()

    def release(self, resource: Any) -> None:
        return None

    def validate(self, resource: Any) -> bool:
        return True

    def health_check(self) -> ResourceHealth:
        return ResourceHealth.HEALTHY

    def get_metrics(self) -> ResourceMetrics:
        return self.metrics


__all__ = [
    "BareClass",
    "ConformingChunker",
    "ConformingFieldTransform",
    "ConformingMergeFilter",
    "ConformingMiddleware",
    "ConformingResourceProvider",
    "ConformingTool",
    "instantiations",
    "not_callable",
    "reset_instantiations",
    "resolvable_async_function",
    "resolvable_function",
]
