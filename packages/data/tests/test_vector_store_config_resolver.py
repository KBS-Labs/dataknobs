"""Tests for the ``vector_store`` config resolver.

The resolver registered by ``dataknobs-data`` into the shared
``config_registries`` lets ``StructuredConfig.validate`` check a raw
``vector_store`` section without constructing the store. These tests pin:

- The resolver is registered eagerly on import.
- It returns the same ``CONFIG_CLS`` the construction registry would use,
  for *every* registered backend (the no-drift guarantee).
- It returns ``None`` for an unknown backend.
- It defaults the discriminator to ``"memory"`` (the factory's own default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import pytest

import dataknobs_data.vector.stores  # noqa: F401 — eager resolver registration
from dataknobs_data.vector.stores import vector_backends
from dataknobs_data.vector.stores.config import MemoryVectorStoreConfig

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    StructuredConfig,
    config_registries,
)


@dataclass(frozen=True)
class _HoldsAVectorStore(StructuredConfig):
    """The smallest config that reaches the resolver through ``validate``.

    Any config declaring a ``vector_store`` polymorphic section has this
    shape; several ship in other packages, but a local one keeps this file
    a test of the resolver rather than of whichever config it borrowed.
    """

    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"vector_store": "vector_store"}

    vector_store: dict[str, Any] = field(default_factory=dict)


def _resolver():
    return config_registries.get("vector_store")


def test_resolver_registered_on_import() -> None:
    assert config_registries.has("vector_store")


def test_resolver_agrees_with_construction_registry_for_all_backends() -> None:
    # Drift guard: for every registered backend, the resolver must return
    # exactly the CONFIG_CLS the construction path reads off the store class,
    # so validation and construction can never resolve to different configs.
    #
    # Iterates `list_known_keys`, not `list_keys`: gating registration on the
    # driver removes an uninstalled backend from the creatable set, which is
    # exactly the set this guard was reading. It would have covered `faiss`
    # on a machine with faiss-cpu and quietly stopped covering it on one
    # without -- reporting a clean scan for a backend it was no longer
    # looking at, in the environment where the resolver is most likely to be
    # wrong about it.
    resolver = _resolver()
    keys = vector_backends.list_known_keys()
    assert keys, "expected at least the always-available memory backend"
    for key in keys:
        expected = getattr(vector_backends.get_factory(key), "CONFIG_CLS", None)
        resolved = resolver({"backend": key})
        if expected is None:
            # Known but not creatable here: no store class to read a schema
            # off. Skipping is right; reporting the name as unrecognised is
            # not.
            assert resolved is SKIP_VALIDATION, f"drift for backend {key!r}"
        else:
            assert resolved is expected, f"drift for backend {key!r}"


def test_resolver_defaults_to_memory() -> None:
    # No "backend" key => memory (the VectorStoreFactory default).
    assert _resolver()({}) is MemoryVectorStoreConfig


def test_resolver_returns_none_for_unknown_backend() -> None:
    result = _resolver()({"backend": "pgvektor"})
    assert result is None


class TestABackendThisMachineCannotBuild:
    """Validating a config must not depend on the local install set.

    Registration probes a backend's driver and declares the backend
    unavailable when it is absent, which is what makes ``available`` mean
    ``installed``. The resolver read the *creatable* set to answer a
    question that is not about creatability, so on a machine without
    ``faiss-cpu`` a perfectly good ``vector_store: {backend: faiss}``
    section -- the one this project's own migration guide prints --
    resolved to ``None`` and ``validate()`` reported it as matching no
    known variant, telling the reader to go and check their spelling.

    A static config is the same config everywhere. What differs is whether
    *this* machine could build it, which is a question ``create()`` answers
    at the point it matters, with a message naming the driver.
    """

    @staticmethod
    def _withdrawn(backend: str):
        """Declare a real backend unavailable, as a driverless machine would.

        Uses the shipped registry and the registration API rather than
        substituting a double, because the resolver reads that singleton
        and the bug is in how it reads it.
        """
        store_cls = vector_backends.get_factory(backend)
        assert store_cls is not None, f"{backend} is not installed in this env"
        metadata = vector_backends.get_metadata(backend)
        vector_backends.declare_unavailable(
            backend,
            metadata=metadata,
            reason="faiss is not installed. Install with: pip install faiss-cpu",
        )
        return store_cls, metadata

    @staticmethod
    def _restore(backend: str, store_cls, metadata) -> None:
        vector_backends.register(backend, store_cls, metadata=metadata, override=True)

    def test_its_config_section_is_not_reported_as_an_unknown_variant(self) -> None:
        store_cls, metadata = self._withdrawn("faiss")
        try:
            assert _resolver()({"backend": "faiss"}) is not None, (
                "a known backend whose driver is missing was reported as a typo"
            )
        finally:
            self._restore("faiss", store_cls, metadata)

    def test_its_section_is_skipped_rather_than_rejected(self) -> None:
        store_cls, metadata = self._withdrawn("faiss")
        try:
            assert _resolver()({"backend": "faiss"}) is SKIP_VALIDATION
        finally:
            self._restore("faiss", store_cls, metadata)

    def test_a_real_typo_is_still_reported(self) -> None:
        """The distinction only matters if the other half still works."""
        store_cls, metadata = self._withdrawn("faiss")
        try:
            assert _resolver()({"backend": "fiass"}) is None
        finally:
            self._restore("faiss", store_cls, metadata)

    def test_a_parent_config_holding_the_section_still_validates(self) -> None:
        """End to end, through ``validate`` -- the consumer-visible failure.

        The resolver is reached from a parent config that declares a
        ``vector_store`` polymorphic section, which is where the wrong
        answer became a ``ConfigurationError`` telling the reader to check
        a discriminator that was correct. Declared here rather than
        imported from a package that owns one, so this stays a test of the
        resolver.
        """
        store_cls, metadata = self._withdrawn("faiss")
        try:
            _HoldsAVectorStore(vector_store={"backend": "faiss"}).validate()
        finally:
            self._restore("faiss", store_cls, metadata)

    def test_a_parent_config_naming_a_typo_still_raises(self) -> None:
        store_cls, metadata = self._withdrawn("faiss")
        try:
            with pytest.raises(ConfigurationError):
                _HoldsAVectorStore(vector_store={"backend": "fiass"}).validate()
        finally:
            self._restore("faiss", store_cls, metadata)


def test_registered_backend_without_config_cls_is_skipped() -> None:
    # A backend registered as a bare callable (no CONFIG_CLS) is recognized
    # but has no typed schema to validate against. The resolver returns
    # SKIP_VALIDATION (not None), so validate() skips it rather than
    # false-positive-raising on a valid, constructible backend.
    def _untyped_factory(config: object = None, **_: object) -> object:
        raise NotImplementedError  # never built — resolver only reads the type

    vector_backends.register("untyped_test_backend", _untyped_factory, override=True)
    try:
        assert _resolver()({"backend": "untyped_test_backend"}) is SKIP_VALIDATION
    finally:
        vector_backends.unregister("untyped_test_backend")


def test_resolved_config_is_structured_config_subclass() -> None:
    cls = _resolver()({"backend": "memory"})
    assert isinstance(cls, type) and issubclass(cls, StructuredConfig)
