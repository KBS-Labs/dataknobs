"""Declarative capability advertisement.

Classes that participate in capability advertisement declare which
optional features they support so consumers can query the support set
before invoking methods that may not be available. This replaces the
implicit "call the method, get NotImplementedError" pattern with an
explicit "query supports(), get a bool" pattern.

The :class:`Capability` enum declares stable identifiers for
cross-cutting optional features. Members are ``str``-typed for stable
serialization and JSON-friendly logging.

Backends advertise capabilities via the ``SUPPORTED_CAPABILITIES``
``ClassVar`` (with :class:`CapabilityMixin`) or via an
:meth:`instance_capabilities` override (with
:class:`DynamicCapabilityMixin` for config-dependent support).

Consumer-side guards use :func:`require_capability` for pre-call
validation or ``host.supports(capability)`` for conditional branches.

Consumer-defined capabilities are supported via raw-string capability
values: pass a string to :meth:`supports` or :func:`require_capability`
instead of a :class:`Capability` enum member. Implementations comparing
against ``SUPPORTED_CAPABILITIES`` accept both forms.
"""

from __future__ import annotations

from enum import Enum
from typing import (
    Any,
    ClassVar,
    Protocol,
    Union,
    runtime_checkable,
)


class Capability(str, Enum):
    """Stable identifiers for cross-cutting optional features.

    Members are ``str``-typed so values serialize cleanly and remain
    stable across enum-method-call semantics changes. Consumer-defined
    capabilities use raw strings rather than Enum extension.

    Members cluster into capability families. Family membership is
    informational only — the enum is flat; :meth:`supports` /
    :func:`require_capability` do not honor family hierarchies.
    """

    # ---- Tenancy ----
    TENANT_SCOPED_LOCKS = "tenant_scoped_locks"
    TENANT_SCOPED_STATE = "tenant_scoped_state"
    PER_TENANT_RATE_LIMITS = "per_tenant_rate_limits"

    # ---- Observability ----
    EVENT_BUS_EMISSION = "event_bus_emission"
    CALLBACK_REGISTRY = "callback_registry"
    METRICS_EMISSION = "metrics_emission"

    # ---- Consistency ----
    SNAPSHOT_ISOLATION = "snapshot_isolation"
    TRANSACTIONAL_METADATA = "transactional_metadata"
    STREAMING_READS = "streaming_reads"
    STREAMING_WRITES = "streaming_writes"

    # ---- Composition ----
    KEY_PATTERN_FILTERING = "key_pattern_filtering"
    CHANGE_SUBSCRIPTION = "change_subscription"


CapabilityLike = Union[Capability, str]


@runtime_checkable
class CapabilityContract(Protocol):
    """Declarative advertise of supported optional features.

    Implementations expose :meth:`supported_capabilities` (classmethod —
    invariant support, queryable without an instance) and
    :meth:`instance_capabilities` (instance method — may depend on
    construction args, useful when the same class supports different
    feature sets based on config).

    :meth:`supports` and :func:`require_capability` are convenience
    helpers; consumers may also query :meth:`instance_capabilities`
    directly.
    """

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]]

    @classmethod
    def supported_capabilities(cls) -> frozenset[Capability]: ...

    def instance_capabilities(self) -> frozenset[Capability]: ...

    def supports(self, capability: CapabilityLike) -> bool: ...


class CapabilityMixin:
    """Default capability-contract implementation reading from a
    ``ClassVar``.

    Subclasses declare:

        class MyBackend(CapabilityMixin, ...):
            SUPPORTED_CAPABILITIES = frozenset({
                Capability.STREAMING_READS,
                Capability.KEY_PATTERN_FILTERING,
            })

    Instance capabilities default to the class-level set. Subclasses
    needing config-dependent support override
    :meth:`instance_capabilities` or inherit from
    :class:`DynamicCapabilityMixin` instead.
    """

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset()

    @classmethod
    def supported_capabilities(cls) -> frozenset[Capability]:
        return cls.SUPPORTED_CAPABILITIES

    def instance_capabilities(self) -> frozenset[Capability]:
        return type(self).SUPPORTED_CAPABILITIES

    def supports(self, capability: CapabilityLike) -> bool:
        normalized = _normalize_capability(capability)
        return normalized in {
            _normalize_capability(c) for c in self.instance_capabilities()
        }


class DynamicCapabilityMixin(CapabilityMixin):
    """Capability-contract implementation with config-driven instance
    capabilities.

    Subclasses override :meth:`_compute_instance_capabilities` to build
    the capability set from ``__init__`` state (e.g. "EVENT_BUS_EMISSION
    only if an event bus was configured").

    The default :meth:`instance_capabilities` caches the computed set on
    first call. Subclasses needing dynamic recomputation should call
    :meth:`_invalidate_capability_cache` after state changes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._capability_cache: frozenset[Capability] | None = None

    def _compute_instance_capabilities(self) -> frozenset[Capability]:
        """Override to compute capabilities from instance state.

        Default returns the ``ClassVar`` ``SUPPORTED_CAPABILITIES``
        (identical to :class:`CapabilityMixin`). Override to add
        config-dependent capabilities.
        """
        return type(self).SUPPORTED_CAPABILITIES

    def instance_capabilities(self) -> frozenset[Capability]:
        if self._capability_cache is None:
            self._capability_cache = self._compute_instance_capabilities()
        return self._capability_cache

    def _invalidate_capability_cache(self) -> None:
        """Force recomputation on next :meth:`instance_capabilities` call.

        Useful when instance state mutates after construction in a way
        that changes supported capabilities (rare).
        """
        self._capability_cache = None


class CapabilityNotSupportedError(Exception):
    """Raised when a required capability is not supported by the host.

    Attributes:
        capability: The capability that was required.
        host: The object that lacks the capability.
    """

    def __init__(self, capability: CapabilityLike, host: Any) -> None:
        self.capability = capability
        self.host = host
        cap_str = (
            capability.value if isinstance(capability, Capability) else str(capability)
        )
        super().__init__(f"{type(host).__name__} does not support {cap_str!r}")


def require_capability(host: Any, capability: CapabilityLike) -> None:
    """Pre-call guard. Raises :class:`CapabilityNotSupportedError` when
    the host does not support the capability.

    Usage::

        require_capability(backend, Capability.TENANT_SCOPED_STATE)
        await backend.set_ingestion_status(ctx, status)

    Accepts both :class:`Capability` enum members and raw strings (for
    consumer-defined capabilities not part of the dataknobs enum).
    """
    supports = getattr(host, "supports", None)
    if supports is None or not supports(capability):
        raise CapabilityNotSupportedError(capability, host)


def _normalize_capability(capability: CapabilityLike) -> str:
    """Coerce :class:`Capability` member or raw string to the underlying
    ``str`` identifier for set comparison.
    """
    if isinstance(capability, Capability):
        return capability.value
    return str(capability)


__all__ = [
    "Capability",
    "CapabilityContract",
    "CapabilityLike",
    "CapabilityMixin",
    "CapabilityNotSupportedError",
    "DynamicCapabilityMixin",
    "require_capability",
]
