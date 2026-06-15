"""Registry tests for ``resolver_backends`` + ``partition_resolver_backends``.

Pins the public registry surface (built-in factory closures, error
shape, consumer-extensibility, async-path parity) for the two resolver
factory registries shipped from :mod:`dataknobs_common.resolver`.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from dataknobs_common.exceptions import OperationError
from dataknobs_common.registry import BackendRegistry, PluginRegistry
from dataknobs_common.resolver import (
    CachedResolver,
    CallablePartitionResolver,
    CallableResolver,
    CompositeResolver,
    DefaultingResolver,
    JoiningPartitionResolver,
    MappingResolver,
    MetadataKeyPartitionResolver,
    NullPartitionResolver,
    NullResolver,
    ResourceResolver,
    TemporalPartitionResolver,
    partition_resolver_backends,
    resolver_backends,
)


class TestResolverBackends:
    """Generic ``resolver_backends`` registry — one happy-path test per
    built-in factory plus the registry-level conformance / error-shape /
    consumer-extensibility pins.
    """

    def test_resolver_backends_is_plugin_registry(self) -> None:
        assert isinstance(resolver_backends, PluginRegistry)

    def test_resolver_backends_is_backend_registry(self) -> None:
        assert isinstance(resolver_backends, BackendRegistry)

    def test_resolver_backends_creates_mapping_resolver(self) -> None:
        resolver = resolver_backends.create(
            config={"backend": "mapping", "mapping": {"a": 1, "b": 2}}
        )
        assert isinstance(resolver, MappingResolver)
        assert resolver.resolve("a") == 1
        assert resolver.resolve("missing") is None

    def test_resolver_backends_creates_callable_resolver(self) -> None:
        def lookup(key: str) -> int | None:
            return len(key) if key else None

        resolver = resolver_backends.create(
            config={"backend": "callable", "fn": lookup}
        )
        assert isinstance(resolver, CallableResolver)
        assert resolver.resolve("hello") == 5

    def test_resolver_backends_creates_composite_resolver(self) -> None:
        first = MappingResolver(mapping={"a": 1})
        second = MappingResolver(mapping={"b": 2})
        resolver = resolver_backends.create(
            config={"backend": "composite", "resolvers": [first, second]}
        )
        assert isinstance(resolver, CompositeResolver)
        assert resolver.resolve("a") == 1
        assert resolver.resolve("b") == 2
        assert resolver.resolve("missing") is None

    def test_resolver_backends_creates_defaulting_resolver(self) -> None:
        inner = MappingResolver(mapping={"a": 1})
        resolver = resolver_backends.create(
            config={
                "backend": "defaulting",
                "inner": inner,
                "default": 99,
            }
        )
        assert isinstance(resolver, DefaultingResolver)
        assert resolver.resolve("a") == 1
        assert resolver.resolve("missing") == 99

    def test_resolver_backends_creates_cached_resolver(self) -> None:
        inner = MappingResolver(mapping={"a": 1})
        resolver = resolver_backends.create(
            config={"backend": "cached", "inner": inner, "max_size": 4}
        )
        assert isinstance(resolver, CachedResolver)
        assert resolver.resolve("a") == 1

    def test_resolver_backends_creates_null_resolver(self) -> None:
        resolver = resolver_backends.create(config={"backend": "null"})
        assert isinstance(resolver, NullResolver)
        assert resolver.resolve("any") is None

    def test_resolver_backends_default_backend_is_mapping(self) -> None:
        """``config_key_default="mapping"`` dispatches to the mapping
        factory when the ``backend`` key is unset — the default-path
        semantic that lets a ``{"mapping": {...}}`` config skip the
        discriminator entirely."""
        resolver = resolver_backends.create(config={"mapping": {"a": 1}})
        assert isinstance(resolver, MappingResolver)
        assert resolver.resolve("a") == 1

    def test_resolver_backends_unknown_backend_message(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolver_backends.create(config={"backend": "never-registered"})
        msg = str(excinfo.value)
        assert "Unknown resolver backend: never-registered" in msg
        assert "Available backends:" in msg
        # Every built-in is enumerated in the sorted message.
        for name in (
            "cached",
            "callable",
            "composite",
            "defaulting",
            "mapping",
            "null",
        ):
            assert name in msg

    def test_resolver_backends_consumer_register(self) -> None:
        """An out-of-tree consumer registers a custom ``ResourceResolver``
        backend through the registry surface and dispatches it via
        ``create(config={"backend": "..."})`` — the consumer-extensibility
        capability the consolidation surfaces."""

        class _LengthResolver:
            def resolve(self, key: str) -> int | None:
                return len(key) if key else None

        def _make_length_resolver(config: dict) -> _LengthResolver:
            return _LengthResolver()

        resolver_backends.register("length-test", _make_length_resolver)
        try:
            resolver = resolver_backends.create(
                config={"backend": "length-test"}
            )
            assert isinstance(resolver, ResourceResolver)
            assert resolver.resolve("hello") == 5
            assert resolver.resolve("") is None
        finally:
            resolver_backends.unregister("length-test")

    async def test_resolver_backends_async_path(self) -> None:
        """``create_async`` resolves a sync factory transparently — the
        await is a no-op on the non-awaitable result. Symmetry guard for
        consumers that use the async surface uniformly."""
        resolver = await resolver_backends.create_async(
            config={"backend": "mapping", "mapping": {"a": 1}}
        )
        assert isinstance(resolver, MappingResolver)
        assert resolver.resolve("a") == 1

    def test_resolver_backends_factory_error_wrapped_in_operation_error(
        self,
    ) -> None:
        """Factory-construction exceptions surface as ``OperationError``
        with the originating exception preserved on ``__cause__`` — the
        documented contract shared with ``create_event_bus`` /
        ``create_lock`` / ``create_rate_limiter`` (PR-B/C/D).

        Selecting the ``"mapping"`` backend without supplying the
        ``"mapping"`` key triggers a ``KeyError`` inside the factory
        closure; ``PluginRegistry.create`` MUST wrap it so consumers
        catch a single error type across the consolidation."""
        with pytest.raises(OperationError) as excinfo:
            resolver_backends.create(config={"backend": "mapping"})
        assert isinstance(excinfo.value.__cause__, KeyError)


class TestPartitionResolverBackends:
    """Partition resolver registry. Distinct namespace from
    ``resolver_backends`` per the per-input-shape split convention; no
    declared partition Protocol so ``validate_type=`` is intentionally
    not set.
    """

    def test_partition_resolver_backends_is_plugin_registry(self) -> None:
        assert isinstance(partition_resolver_backends, PluginRegistry)

    def test_partition_resolver_backends_is_backend_registry(self) -> None:
        assert isinstance(partition_resolver_backends, BackendRegistry)

    def test_partition_resolver_backends_creates_null(self) -> None:
        resolver = partition_resolver_backends.create(
            config={"backend": "null", "default": "global"}
        )
        assert isinstance(resolver, NullPartitionResolver)
        record = SimpleNamespace(metadata={})
        assert resolver.resolve(record) == "global"

    def test_partition_resolver_backends_creates_metadata_key(self) -> None:
        resolver = partition_resolver_backends.create(
            config={
                "backend": "metadata_key",
                "metadata_key": "tenant_id",
                "default": "shared",
            }
        )
        assert isinstance(resolver, MetadataKeyPartitionResolver)
        assert (
            resolver.resolve(SimpleNamespace(metadata={"tenant_id": "acme"}))
            == "acme"
        )
        assert resolver.resolve(SimpleNamespace(metadata={})) == "shared"

    def test_partition_resolver_backends_creates_temporal(self) -> None:
        resolver = partition_resolver_backends.create(
            config={
                "backend": "temporal",
                "timestamp_key": "ingested_at",
                "bucket": "quarter",
            }
        )
        assert isinstance(resolver, TemporalPartitionResolver)
        record = SimpleNamespace(
            metadata={"ingested_at": datetime(2026, 5, 1, 12, 0, 0)}
        )
        assert resolver.resolve(record) == "2026_q2"

    def test_partition_resolver_backends_creates_callable(self) -> None:
        def partition_fn(record: object) -> str | None:
            return "custom"

        resolver = partition_resolver_backends.create(
            config={"backend": "callable", "fn": partition_fn}
        )
        assert isinstance(resolver, CallablePartitionResolver)
        assert resolver.resolve(SimpleNamespace()) == "custom"

    def test_partition_resolver_backends_creates_joining(self) -> None:
        tenant = MetadataKeyPartitionResolver(metadata_key="tenant_id")
        temporal = TemporalPartitionResolver(
            timestamp_key="ingested_at", bucket="quarter"
        )
        resolver = partition_resolver_backends.create(
            config={
                "backend": "joining",
                "resolvers": [tenant, temporal],
                "sep": "::",
            }
        )
        assert isinstance(resolver, JoiningPartitionResolver)
        record = SimpleNamespace(
            metadata={
                "tenant_id": "acme",
                "ingested_at": datetime(2026, 5, 1, 12, 0, 0),
            }
        )
        assert resolver.resolve(record) == "acme::2026_q2"

    def test_partition_resolver_backends_unknown_backend_message(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            partition_resolver_backends.create(
                config={"backend": "never-registered"}
            )
        msg = str(excinfo.value)
        assert "Unknown partition resolver backend: never-registered" in msg
        assert "Available backends:" in msg
        for name in (
            "callable",
            "joining",
            "metadata_key",
            "null",
            "temporal",
        ):
            assert name in msg

    def test_partition_resolver_backends_consumer_register(self) -> None:
        """An out-of-tree consumer registers a custom partition resolver
        backend through the registry surface and dispatches it via
        ``create(config={"backend": "..."})`` — sibling pin to
        ``test_resolver_backends_consumer_register``. The partition
        registry has no ``validate_type=`` (record→str shape has no
        declared Protocol), so the conformance check is structural at
        use-time only."""

        class _TenantPartition:
            def resolve(self, record: object) -> str | None:
                return "tenant-test"

        def _make_tenant_partition(config: dict) -> _TenantPartition:
            return _TenantPartition()

        partition_resolver_backends.register(
            "tenant-test", _make_tenant_partition
        )
        try:
            resolver = partition_resolver_backends.create(
                config={"backend": "tenant-test"}
            )
            assert resolver.resolve(SimpleNamespace()) == "tenant-test"
        finally:
            partition_resolver_backends.unregister("tenant-test")

    async def test_partition_resolver_backends_async_path(self) -> None:
        """``create_async`` resolves a sync factory transparently — the
        await is a no-op on the non-awaitable result. Sibling symmetry
        guard to ``test_resolver_backends_async_path``; the async shim
        is a distinct code path through ``PluginRegistry.create_async``."""
        resolver = await partition_resolver_backends.create_async(
            config={"backend": "null", "default": "global"}
        )
        assert isinstance(resolver, NullPartitionResolver)
        assert resolver.resolve(SimpleNamespace()) == "global"

    async def test_partition_resolver_backends_async_unknown_backend_message(
        self,
    ) -> None:
        """Async shim surfaces the same unknown-backend ``ValueError``
        shape as the sync path. Sibling pin to
        ``test_partition_resolver_backends_unknown_backend_message`` —
        the async dispatch is a separate code path that must surface
        the registry's not-found message identically."""
        with pytest.raises(ValueError) as excinfo:
            await partition_resolver_backends.create_async(
                config={"backend": "still-never-registered"}
            )
        msg = str(excinfo.value)
        assert (
            "Unknown partition resolver backend: still-never-registered" in msg
        )
        # Prefix-only check (defensive vs. exact enumeration) — a
        # consumer-registered backend lingering from an upstream test
        # run shouldn't flip this pin.
        assert "Available backends:" in msg
