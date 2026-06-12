"""Behavior tests for CapabilityMixin and DynamicCapabilityMixin."""

from __future__ import annotations

from typing import ClassVar

from dataknobs_common.capabilities import (
    Capability,
    CapabilityMixin,
    DynamicCapabilityMixin,
)


# ---- CapabilityMixin (ClassVar-only) ----


class _StaticBackend(CapabilityMixin):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STREAMING_READS,
        Capability.KEY_PATTERN_FILTERING,
    })


def test_static_backend_supports_declared_capability() -> None:
    backend = _StaticBackend()
    assert backend.supports(Capability.STREAMING_READS)
    assert backend.supports(Capability.KEY_PATTERN_FILTERING)


def test_static_backend_does_not_support_undeclared_capability() -> None:
    backend = _StaticBackend()
    assert not backend.supports(Capability.SNAPSHOT_ISOLATION)
    assert not backend.supports(Capability.TENANT_SCOPED_LOCKS)


def test_static_backend_classmethod_returns_full_set() -> None:
    assert _StaticBackend.supported_capabilities() == frozenset({
        Capability.STREAMING_READS,
        Capability.KEY_PATTERN_FILTERING,
    })


def test_static_backend_instance_capabilities_matches_classvar() -> None:
    backend = _StaticBackend()
    assert backend.instance_capabilities() == _StaticBackend.SUPPORTED_CAPABILITIES


def test_supports_accepts_raw_string_capability() -> None:
    """Consumer-defined capabilities expressed as raw strings work.

    The ``CapabilityLike`` annotation on ``SUPPORTED_CAPABILITIES``
    accepts both ``Capability`` enum members and bare strings, so no
    ``# type: ignore`` is needed.
    """

    class _CustomBackend(CapabilityMixin):
        SUPPORTED_CAPABILITIES: ClassVar[frozenset[str]] = frozenset({
            "custom_feature_x",  # consumer-defined; not in the enum
        })

    backend = _CustomBackend()
    assert backend.supports("custom_feature_x")
    assert not backend.supports("custom_feature_y")


def test_default_mixin_has_empty_capabilities() -> None:
    """A class that doesn't declare SUPPORTED_CAPABILITIES inherits empty."""

    class _NoDeclarations(CapabilityMixin):
        pass

    backend = _NoDeclarations()
    assert backend.supported_capabilities() == frozenset()
    assert not backend.supports(Capability.STREAMING_READS)


# ---- DynamicCapabilityMixin (computed) ----


class _DynamicBackend(DynamicCapabilityMixin):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STREAMING_READS,
    })

    def __init__(self, *, event_bus_configured: bool = False) -> None:
        self._event_bus_configured = event_bus_configured
        self._init_capability_cache()

    def _compute_instance_capabilities(self) -> frozenset[Capability]:
        caps = self.SUPPORTED_CAPABILITIES
        if self._event_bus_configured:
            caps = caps | {
                Capability.EVENT_BUS_EMISSION,
                Capability.CHANGE_SUBSCRIPTION,
            }
        return caps


def test_dynamic_backend_no_event_bus_no_emission_capability() -> None:
    backend = _DynamicBackend(event_bus_configured=False)
    assert backend.supports(Capability.STREAMING_READS)
    assert not backend.supports(Capability.EVENT_BUS_EMISSION)
    assert not backend.supports(Capability.CHANGE_SUBSCRIPTION)


def test_dynamic_backend_with_event_bus_gains_emission_capability() -> None:
    backend = _DynamicBackend(event_bus_configured=True)
    assert backend.supports(Capability.STREAMING_READS)
    assert backend.supports(Capability.EVENT_BUS_EMISSION)
    assert backend.supports(Capability.CHANGE_SUBSCRIPTION)


def test_dynamic_backend_classmethod_returns_base_set_only() -> None:
    """Classmethod query doesn't see instance-dependent capabilities."""
    assert _DynamicBackend.supported_capabilities() == frozenset({
        Capability.STREAMING_READS,
    })


def test_dynamic_backend_cache_persists_across_calls() -> None:
    backend = _DynamicBackend(event_bus_configured=True)
    first = backend.instance_capabilities()
    second = backend.instance_capabilities()
    assert first is second  # same object — cache hit


def test_dynamic_backend_cache_invalidation() -> None:
    backend = _DynamicBackend(event_bus_configured=True)
    _ = backend.instance_capabilities()
    backend._event_bus_configured = False
    backend._invalidate_capability_cache()
    assert not backend.supports(Capability.EVENT_BUS_EMISSION)


# ---- Composite intersection ----


def test_composite_intersection_pattern() -> None:
    """Consumers compose multiple capability-bearing hosts by intersecting.

    Documented usage: a composite over multiple backends supports only
    the intersection — the composite is only as capable as the weakest
    member.
    """
    backend_a = _StaticBackend()  # STREAMING_READS, KEY_PATTERN_FILTERING
    backend_b = _DynamicBackend(event_bus_configured=True)
    # backend_b: STREAMING_READS, EVENT_BUS_EMISSION, CHANGE_SUBSCRIPTION

    composite = backend_a.instance_capabilities() & backend_b.instance_capabilities()
    assert composite == frozenset({Capability.STREAMING_READS})


# ---- DynamicCapabilityMixin with another base (MI shape) ----


class _OtherBase:
    """A non-mixin base with a positional-arg ``__init__``.

    Models the common adopter shape: a backend already inherits from
    a base class that takes positional args. The capability mixin must
    not collide with that base's ``__init__`` signature.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _MultiBaseBackend(DynamicCapabilityMixin, _OtherBase):
    """Adopter that pairs the mixin with another base.

    The mixin is listed FIRST among bases per the documented contract.
    The subclass owns its own ``super().__init__(...)`` chain — the
    mixin no longer forwards ``*args, **kwargs``, so adopters cannot
    accidentally route kwargs to ``object.__init__``.
    """

    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STREAMING_READS,
    })

    def __init__(self, name: str, *, enable_extra: bool = False) -> None:
        super().__init__(name)
        self._enable_extra = enable_extra
        self._init_capability_cache()

    def _compute_instance_capabilities(self) -> frozenset[Capability]:
        caps = self.SUPPORTED_CAPABILITIES
        if self._enable_extra:
            caps = caps | {Capability.EVENT_BUS_EMISSION}
        return caps


def test_multi_base_adopter_constructs_cleanly() -> None:
    """The mixin coexists with another positional-arg base.

    Previously the mixin's ``__init__(*args, **kwargs)`` forwarder would
    route a kwarg past ``_OtherBase`` to ``object.__init__`` and raise
    ``TypeError``. The current contract (subclass calls
    ``_init_capability_cache()`` explicitly) avoids that footgun.
    """
    backend = _MultiBaseBackend("acme", enable_extra=True)
    assert backend.name == "acme"
    assert backend.supports(Capability.STREAMING_READS)
    assert backend.supports(Capability.EVENT_BUS_EMISSION)


def test_multi_base_adopter_cache_works() -> None:
    """Cache initialization succeeds when the mixin is not the only base."""
    backend = _MultiBaseBackend("acme")
    first = backend.instance_capabilities()
    second = backend.instance_capabilities()
    assert first is second  # cached
