"""Resource classes a ``custom`` resource config can point at.

``_create_resource`` turns a dotted ``class:`` from configuration into a live
resource provider, so testing it needs targets on both sides of the shape
check: one that satisfies :class:`IResourceProvider` and one that does not.

Both count their constructions. Asserting only that a wrong-shape target
*raised* would pass against an implementation that built the object, looked at
it, threw it away and raised — which is exactly the pattern the canonical
resolver exists to make unexpressible. The counter is what tells the two apart.

A module rather than classes defined inside the test: the thing under test
resolves a dotted path through ``importlib``, and a class defined in a test
function has no importable path to name.
"""

from __future__ import annotations

from typing import Any

from dataknobs_fsm.resources.base import (
    BaseResourceProvider,
    ResourceHealth,
    ResourceMetrics,
)


class _InstantiationCounter:
    """A count in an attribute rather than a module global.

    A rebound module-level ``int`` is invisible to anyone who imported the
    *name* rather than the module, so an assertion on it would silently read a
    frozen copy. An object with a field cannot be read that way by accident.
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
    """Zero the counter. Call from a fixture, not inline."""
    instantiations.reset()


class ConformingResource(BaseResourceProvider):
    """A provider of the shape ``ResourceManager.register_provider`` declares."""

    def __init__(self, name: str = "fixture", **kwargs: Any) -> None:
        super().__init__(name)
        instantiations.bump()

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


class StrictSignatureResource(BaseResourceProvider):
    """A provider that declares its parameters instead of absorbing them.

    ``ConformingResource`` takes ``**kwargs``, which is the shape that hides
    the builder passing a key that is not a constructor argument: the stray
    keyword lands in the catch-all and nothing complains. Most real providers
    are written this way — ``FileSystemResource`` and ``HTTPServiceResource``
    both name their parameters — so a fixture that only ever absorbs cannot
    speak for them.
    """

    def __init__(self, name: str = "fixture", param1: str | None = None) -> None:
        super().__init__(name)
        self.param1 = param1
        instantiations.bump()

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


class NoInitResource:
    """Satisfies the protocol structurally and defines no ``__init__``.

    ``IResourceProvider`` is a runtime-checkable, method-only Protocol, so
    conforming to it needs the five methods and nothing else — a provider has
    no obligation to define a constructor. Such a class inherits
    ``object.__init__``, a slot wrapper, which is the shape that has no
    ``__code__`` attribute to read parameter names out of.

    Not a ``BaseResourceProvider`` subclass on purpose: inheriting a *Python*
    ``__init__`` supplies a ``__code__`` and would not exercise this at all.
    """

    def acquire(self, **kwargs: Any) -> Any:
        return object()

    def release(self, resource: Any) -> None:
        return None

    def validate(self, resource: Any) -> bool:
        return True

    def health_check(self) -> ResourceHealth:
        return ResourceHealth.HEALTHY

    def get_metrics(self) -> ResourceMetrics:
        return ResourceMetrics()


class NotAResource:
    """Resolves fine, satisfies nothing, and reports being constructed.

    Counted deliberately: the claim under test is that this class is rejected
    *without* being built, which requires it to be able to say that it was.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        instantiations.bump()


__all__ = [
    "ConformingResource",
    "NoInitResource",
    "NotAResource",
    "StrictSignatureResource",
    "instantiations",
    "reset_instantiations",
]
