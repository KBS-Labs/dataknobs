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

This surface is distinct from :class:`dataknobs_llm.ModelCapability`,
which advertises LLM-specific model features (chat, function calling,
streaming, vision, etc.) rather than cross-cutting infrastructure
capabilities. They cover disjoint concept spaces and are not bridged.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Protocol,
    Union,
    runtime_checkable,
)

from dataknobs_common.exceptions import OperationError


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
    TENANT_SCOPED_CHUNKS = "tenant_scoped_chunks"
    TENANT_SCOPED_LOCKS = "tenant_scoped_locks"
    TENANT_SCOPED_STATE = "tenant_scoped_state"
    PER_TENANT_RATE_LIMITS = "per_tenant_rate_limits"

    # ---- Observability ----
    EVENT_BUS_EMISSION = "event_bus_emission"
    CALLBACK_REGISTRY = "callback_registry"
    METRICS_EMISSION = "metrics_emission"
    INGEST_EVENT_PUBLICATION = "ingest_event_publication"
    BACKEND_STATE_OBSERVABILITY = "backend_state_observability"
    EXECUTION_TRACKING = "execution_tracking"

    # ---- Consistency ----
    SNAPSHOT_ISOLATION = "snapshot_isolation"
    TRANSACTIONAL_METADATA = "transactional_metadata"
    STREAMING_READS = "streaming_reads"
    STREAMING_WRITES = "streaming_writes"

    # ---- Composition ----
    KEY_PATTERN_FILTERING = "key_pattern_filtering"
    CHANGE_SUBSCRIPTION = "change_subscription"

    # ---- Scope projection ----
    SCOPE_PROJECTOR_READ_ONLY = "scope_projector_read_only"


CAPABILITY_FAMILIES: Mapping[str, frozenset[Capability]] = MappingProxyType({
    "tenancy": frozenset({
        Capability.TENANT_SCOPED_CHUNKS,
        Capability.TENANT_SCOPED_LOCKS,
        Capability.TENANT_SCOPED_STATE,
        Capability.PER_TENANT_RATE_LIMITS,
    }),
    "observability": frozenset({
        Capability.EVENT_BUS_EMISSION,
        Capability.CALLBACK_REGISTRY,
        Capability.METRICS_EMISSION,
        Capability.INGEST_EVENT_PUBLICATION,
        Capability.BACKEND_STATE_OBSERVABILITY,
        Capability.EXECUTION_TRACKING,
    }),
    "consistency": frozenset({
        Capability.SNAPSHOT_ISOLATION,
        Capability.TRANSACTIONAL_METADATA,
        Capability.STREAMING_READS,
        Capability.STREAMING_WRITES,
    }),
    "composition": frozenset({
        Capability.KEY_PATTERN_FILTERING,
        Capability.CHANGE_SUBSCRIPTION,
    }),
    "scope_projection": frozenset({
        Capability.SCOPE_PROJECTOR_READ_ONLY,
    }),
})
"""Family → capability-member mapping. Family membership is
informational only — :meth:`CapabilityMixin.supports` does not honor
family hierarchies. Useful for consumers wanting "all tenancy
capabilities" without hand-rolling the set. Immutable via
:class:`~types.MappingProxyType`.
"""


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

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]]

    @classmethod
    def supported_capabilities(cls) -> frozenset[CapabilityLike]: ...

    def instance_capabilities(self) -> frozenset[CapabilityLike]: ...

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

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = frozenset()

    @classmethod
    def supported_capabilities(cls) -> frozenset[CapabilityLike]:
        return cls.SUPPORTED_CAPABILITIES

    def instance_capabilities(self) -> frozenset[CapabilityLike]:
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

    The mixin owns a single ``_capability_cache`` field. The first call
    to :meth:`instance_capabilities` populates it from
    :meth:`_compute_instance_capabilities`; subsequent calls hit the
    cache. Subclasses needing dynamic recomputation call
    :meth:`_invalidate_capability_cache` after state changes.

    Subclass ``__init__`` requirements:

    - The mixin does NOT forward ``__init__`` args through cooperative
      multiple inheritance (the previous ``*args, **kwargs`` shape
      collided with ``object.__init__`` when adopters' MRO terminated
      there). Subclasses are responsible for their own
      ``super().__init__(...)`` chain.
    - Subclasses MUST call :meth:`_init_capability_cache` exactly once
      in their ``__init__`` (typically as the LAST line) to initialize
      the cache field. Forgetting this raises ``AttributeError`` on the
      first :meth:`instance_capabilities` call.
    - List :class:`DynamicCapabilityMixin` FIRST among bases so the
      mixin's resolution wins consistently when the subclass also
      inherits from a config-consumer or another mixin.

    Example::

        class MyBackend(DynamicCapabilityMixin, OtherBase):
            SUPPORTED_CAPABILITIES = frozenset({Capability.STREAMING_READS})

            def __init__(self, *, event_bus=None) -> None:
                super().__init__()  # configures OtherBase
                self._event_bus = event_bus
                self._init_capability_cache()

            def _compute_instance_capabilities(self):
                caps = self.SUPPORTED_CAPABILITIES
                if self._event_bus is not None:
                    caps = caps | {Capability.EVENT_BUS_EMISSION}
                return caps

    Cache initialization is not thread-safe — two threads observing a
    fresh instance may both compute the set. The result is idempotent
    (the computed set is invariant for a given instance state) so no
    data corruption occurs, but redundant computation is possible in
    multi-threaded sync contexts. Async (single-loop) contexts are
    unaffected.
    """

    def _init_capability_cache(self) -> None:
        """Initialize the capability cache field.

        MUST be called from each subclass's ``__init__``. The cache is
        explicit because the mixin cannot reliably participate in
        cooperative MI without dictating the subclass ``__init__`` shape.
        """
        self._capability_cache: frozenset[CapabilityLike] | None = None

    def _compute_instance_capabilities(self) -> frozenset[CapabilityLike]:
        """Override to compute capabilities from instance state.

        Default returns the ``ClassVar`` ``SUPPORTED_CAPABILITIES``
        (identical to :class:`CapabilityMixin`). Override to add
        config-dependent capabilities.
        """
        return type(self).SUPPORTED_CAPABILITIES

    def instance_capabilities(self) -> frozenset[CapabilityLike]:
        if self._capability_cache is None:
            self._capability_cache = self._compute_instance_capabilities()
        return self._capability_cache

    def _invalidate_capability_cache(self) -> None:
        """Force recomputation on next :meth:`instance_capabilities` call.

        Useful when instance state mutates after construction in a way
        that changes supported capabilities (rare).
        """
        self._capability_cache = None


class CapabilityNotSupportedError(OperationError):
    """Raised when a required capability is not supported by the host.

    Part of the :class:`~dataknobs_common.exceptions.DataknobsError`
    hierarchy via :class:`~dataknobs_common.exceptions.OperationError`
    so consumers catching ``DataknobsError`` see this exception as a
    member of the unified error surface. The offending capability
    identifier and host class name are recorded on
    :attr:`~dataknobs_common.exceptions.DataknobsError.context` for
    structured logging.

    Attributes:
        capability: The capability that was required.
        host: The object that lacks the capability.
    """

    def __init__(self, capability: CapabilityLike, host: Any) -> None:
        cap_str = _normalize_capability(capability)
        host_class = type(host).__name__
        super().__init__(
            f"{host_class} does not support {cap_str!r}",
            context={"capability": cap_str, "host": host_class},
        )
        self.capability = capability
        self.host = host


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
    "CAPABILITY_FAMILIES",
    "Capability",
    "CapabilityContract",
    "CapabilityLike",
    "CapabilityMixin",
    "CapabilityNotSupportedError",
    "DynamicCapabilityMixin",
    "require_capability",
]
