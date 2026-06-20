"""Runtime-state projection Protocol and reference implementations.

A :class:`ScopeProjector` transforms a source value (typically a runtime
state-bearing object — a wizard state, a manager metadata mapping, a
sub-strategy outputs dict) into a ``Mapping[str, Any]`` suitable for use
as a safe-eval scope, a template-evaluation context, or any other
key-lookup-by-name surface.

The Protocol is intentionally minimal — one method, one return shape.
Implementations choose semantics:

- :class:`IdentityProjector` — pass-through (Mapping in, same Mapping
  out; non-Mapping → empty dict)
- :class:`ReadOnlyProjector` — wrap in a read-only view; writes raise
- :class:`WhitelistProjector` — filter to a declared key set
- :class:`ChainedProjector` — compose multiple projectors, later wins
- :class:`CallableProjector` — wrap a callable returning a Mapping
- :class:`CachedProjector` — LRU-memoize an inner projector

Distinct from a state-bridge ``project``:
    A *state-bridge* — a projector whose ``project`` writes a derived
    value into a host metadata key rather than returning a scope Mapping —
    shares the method name but has different value semantics: it returns
    whatever value the consumer wants stored under a key, not a
    ``Mapping[str, Any]`` scope. The two are deliberately NOT unified; a
    ``ScopeProjector`` always returns a Mapping. The shared method name is
    a known collision — code accepting one Protocol must not silently
    accept the other. (No such bridge ships today; this note records the
    boundary so a future bridge isn't mistaken for a scope projector.)

Composition with the callback substrate:
    Callback bodies registered on a ``CallbackRegistry`` frequently need
    scoped data; a per-callback ``ScopeProjector`` closure makes the
    projection explicit and testable. The projector is constructed at
    registration time; the callback applies it on every fire.

A known design seam: the three source-capturing impls
(:class:`ReadOnlyProjector`, :class:`WhitelistProjector`, and the
bots-layer ``JinjaInputsProjector``) capture their source at construction
and **ignore** the ``project(source)`` argument. Consumers needing a
per-call source select via :class:`CallableProjector` (or construct a
fresh projector). A future ``StatefulProjector`` honoring ``source``, or
a narrowing of the Protocol signature, is left open.

The Jinja-expression-evaluating reference impl lives in
``dataknobs_bots.prompts.scope`` to avoid forcing an optional ``jinja2``
dependency onto ``dataknobs_common``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CachedProjector",
    "CallableProjector",
    "ChainedProjector",
    "IdentityProjector",
    "ReadOnlyProjector",
    "ScopeProjector",
    "WhitelistProjector",
]


@runtime_checkable
class ScopeProjector(Protocol):
    """Projects a source value into a ``Mapping[str, Any]``.

    Implementations MUST:

    - return a ``Mapping[str, Any]`` (never ``None`` — return an empty
      Mapping for the "no projection" case)
    - be deterministic (same source → equal Mapping across calls)
    - not mutate the source value

    The returned Mapping MAY be a view into the source (e.g.
    :class:`ReadOnlyProjector` returns ``MappingProxyType(source)``) or a
    fresh dict (e.g. :class:`WhitelistProjector` returns a filtered copy).
    Implementations document their choice in their docstring.
    """

    def project(self, source: Any) -> Mapping[str, Any]: ...


# --------------------------------------------------------------------- #
# Reference implementations — generic
# --------------------------------------------------------------------- #


class IdentityProjector:
    """Pass-through projector.

    If ``source`` is a Mapping, return it as-is. Otherwise return an empty
    Mapping. The backwards-compatible default for consumers that haven't
    opted into a more specific projection.

    Note: returning ``source`` as-is means downstream consumers may
    accidentally mutate the source through the returned Mapping. For
    safe-eval scopes where the source must be protected, use
    :class:`ReadOnlyProjector` instead.
    """

    def project(self, source: Any) -> Mapping[str, Any]:
        if isinstance(source, Mapping):
            return source
        return {}


class ReadOnlyProjector:
    """Read-only view projector.

    Wraps a source Mapping in a :class:`~types.MappingProxyType` view.
    Reads return the live value (no copy); writes through the returned
    view raise ``TypeError``.

    Use case: safe-eval scopes where the eval context must not mutate the
    wrapped source. A consumer evaluating a condition against a scope
    built with a ``ReadOnlyProjector`` over a metadata mapping cannot
    accidentally introduce a mutation via the eval side.

    Construct once per projection target; the projector itself is cheap
    (no allocations beyond the proxy). The view is **live** — mutations to
    the wrapped source are visible through the proxy (read-only at the
    boundary, not a point-in-time snapshot).
    """

    def __init__(self, source: Mapping[str, Any]) -> None:
        self._source = source

    def project(self, source: Any) -> Mapping[str, Any]:
        # The `source` argument is accepted for Protocol conformance but
        # ignored — this projector wraps the source captured at
        # construction. Consumers wanting to project a different source
        # per call use ``CallableProjector`` instead.
        return MappingProxyType(self._source)


class WhitelistProjector:
    """Whitelist projector — returns only declared keys.

    Construct with a frozenset of allowed key names. :meth:`project`
    returns a fresh dict containing only declared keys present in the
    source; absent declared keys are omitted (no defaults injected).

    Use case: security-bounded scope projection. A surface that exposes
    only ``session_id`` to a transition condition constructs
    ``WhitelistProjector(metadata, frozenset({"session_id"}))``.
    Conditions referring to other keys raise ``NameError`` at eval time
    (the natural safe-eval failure mode).

    The returned dict is fresh on every :meth:`project` call — mutations
    through the returned dict do not propagate to the source.
    """

    def __init__(
        self,
        source: Mapping[str, Any],
        allowed_keys: frozenset[str],
    ) -> None:
        self._source = source
        self._allowed_keys = allowed_keys

    def project(self, source: Any) -> Mapping[str, Any]:
        # The `source` argument is accepted for Protocol conformance;
        # this projector reads from the source captured at construction.
        return {
            k: self._source[k]
            for k in self._allowed_keys
            if k in self._source
        }


class ChainedProjector:
    """Composes projectors left-to-right.

    Applies each inner projector to the source, then merges the results
    into a fresh dict. Later projectors win on key collision.

    Use case: composition. A renderer combining default template params
    with declarative ``inputs:`` derived variables constructs
    ``ChainedProjector(default_params_projector, inputs_projector)``. The
    default params land first; the declarative inputs may override.

    Mutating the returned dict does not propagate to any inner projector
    or source.

    Caveat — source-capturing inners: this composer passes the per-call
    ``source`` to every inner, but the source-capturing impls
    (:class:`ReadOnlyProjector`, :class:`WhitelistProjector`, and the
    bots-layer ``JinjaInputsProjector``) ignore it and project the source
    they captured at construction. Composing one of those means the
    ``source`` passed to :meth:`project` is silently dropped for that
    inner — the captured source wins with no error. This is the in-tree
    pattern (the renderer constructs each capturing inner with the exact
    source it wants), but composing a capturing inner expecting it to see
    a *different* per-call source will silently misbehave.
    """

    def __init__(self, *projectors: ScopeProjector) -> None:
        if not projectors:
            raise ValueError(
                "ChainedProjector requires at least one inner projector",
            )
        self._projectors = projectors

    def project(self, source: Any) -> Mapping[str, Any]:
        merged: dict[str, Any] = {}
        for projector in self._projectors:
            merged.update(projector.project(source))
        return merged


class CallableProjector:
    """Wraps a callable returning a Mapping.

    Escape hatch for arbitrary projection logic where none of the
    declarative impls fit. The callable receives the source and returns a
    Mapping.

    Use case: ad-hoc per-consumer projection. A consumer wanting to
    project ``state.data`` plus a derived field writes
    ``CallableProjector(lambda s: {**s, "computed_total": sum(...)})``.
    Unlike the source-capturing impls, this projector honors the per-call
    ``source`` argument.
    """

    def __init__(
        self,
        fn: Callable[[Any], Mapping[str, Any]],
    ) -> None:
        self._fn = fn

    def project(self, source: Any) -> Mapping[str, Any]:
        result = self._fn(source)
        if not isinstance(result, Mapping):
            raise TypeError(
                f"CallableProjector function returned "
                f"{type(result).__name__}; ScopeProjector requires a "
                f"Mapping return.",
            )
        return result


class CachedProjector:
    """LRU-memoizing wrapper over any :class:`ScopeProjector`.

    Mirrors the shipped :class:`~dataknobs_common.resolver.CachedResolver`.
    Caches ``inner.project(source)`` keyed by the source. Use when a
    projection is expensive (e.g. a Jinja-expression projector evaluated
    per turn) and the source is hashable + stable across calls.

    The source must be hashable to serve as a cache key; non-hashable
    sources (e.g. a dict) raise ``TypeError``. The source-capturing
    in-tree projectors that ignore ``source`` are the common cached
    target, so the typical key is a sentinel/constant.

    Caveat — source-capturing inners: the cache keys on the per-call
    ``source``, but a source-capturing inner ignores ``source`` and always
    projects its captured source. Caching such an inner under *varying*
    keys is incoherent — distinct keys map to entries that all return the
    captured projection, wasting cache slots. Cache a source-capturing
    inner under a single constant/sentinel key (the in-tree pattern), and
    reserve varying keys for source-honoring inners
    (:class:`CallableProjector`).
    """

    def __init__(self, inner: ScopeProjector, max_size: int = 128) -> None:
        self._inner = inner
        # ``lru_cache`` wraps the inner's *bound* method stored on this
        # instance, so the cache is per-CachedProjector — no cross-instance
        # leak and no unbounded growth beyond ``max_size``.
        self._cached = lru_cache(maxsize=max_size)(self._inner.project)

    def project(self, source: Any) -> Mapping[str, Any]:
        return self._cached(source)
