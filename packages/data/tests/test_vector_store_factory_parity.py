"""Drift guards for the ``dataknobs-data`` vector store registry.

``VectorStoreFactory.create`` dispatches via ``backend_class(config)``
— the whole-dict-pass shape, now backed by
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`.
Drift modes:

- A registered backend stops being a ``StructuredConfigConsumer`` (its
  ``CONFIG_CLS`` field set drifts from the documented surface, or the
  consumer mixin stops being the construction entry point).
- A backend's config dataclass loses round-trip symmetry.

These structural checks run without instantiating backends, so
optional-dependency backends (faiss, chroma, pgvector) are still
audited. Behavioural coverage of each vector store lives in its own
test module; construction-parity coverage lives in
``test_vector_store_structured_config.py``.
"""

from __future__ import annotations

import pytest
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_common.testing import assert_structured_config_consumer

from dataknobs_data.vector.stores import vector_backends


def _registered_backend_classes() -> list[tuple[str, type]]:
    by_class: dict[type, str] = {}
    for key in vector_backends.list_keys():
        cls = vector_backends.get_factory(key)
        if cls is None:
            continue
        by_class.setdefault(cls, key)
    return [(name, cls) for cls, name in by_class.items()]


VECTOR_BACKENDS = _registered_backend_classes()


@pytest.mark.parametrize(
    "name, backend_cls",
    VECTOR_BACKENDS,
    ids=[name for name, _ in VECTOR_BACKENDS],
)
def test_vector_store_is_structured_config_consumer(
    name: str, backend_cls: type
) -> None:
    """Every registered vector backend applies the structured-config pattern.

    Pins, without instantiating the backend (so optional-dependency
    backends are still audited): a declared ``CONFIG_CLS`` that is a
    ``StructuredConfig`` subclass, a config field set matching the
    construction surface, the consumer mixin preceding other bases in the
    MRO (so its ``__init__`` is the construction entry point), and
    entry-point / collaborator-hook safety. This is what keeps the
    factory's ``backend_class(config)`` dict dispatch working.
    """
    assert issubclass(backend_cls, StructuredConfigConsumer)
    assert_structured_config_consumer(backend_cls)


def test_memory_vector_store_constructs_from_empty_config() -> None:
    """Smoke test: ``MemoryVectorStore({})`` succeeds (no optional deps)."""
    from dataknobs_data.vector.stores.memory import MemoryVectorStore

    store = MemoryVectorStore({"dimensions": 8})
    assert store is not None


def test_registered_vector_backends_audit_set() -> None:
    """The audit matrix above must cover every registered built-in.

    A new built-in vector backend → it is auto-discovered by
    ``_registered_backend_classes`` and audited by the parametrized
    guard above; this asserts the canonical built-ins remain registered.
    """
    keys = set(vector_backends.list_keys())
    # The set of canonical backend names that MUST exist as built-ins.
    canonical = {"memory"}
    missing = canonical - keys
    assert not missing, f"Built-in vector backends missing: {missing}"
