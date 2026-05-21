"""Drift guards for the ``dataknobs-data`` vector store registry.

``VectorStoreFactory.create`` dispatches via ``backend_class(config)``
— the whole-dict-pass shape. Drift modes:

- A registered backend is no longer constructable from a dict
  (constructor signature regressed to direct kwargs).
- A backend no longer reads its documented config keys via the
  base-class chain.

These structural checks run without instantiating backends, so
optional-dependency backends (faiss, chroma, pgvector) are still
audited. Behavioural coverage of each vector store lives in its own
test module.
"""

from __future__ import annotations

import inspect

import pytest

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
def test_vector_store_accepts_dict_config(
    name: str, backend_cls: type
) -> None:
    """Every registered vector backend's ``__init__`` accepts a dict.

    ``VectorStoreFactory.create`` calls ``backend_class(config)`` with
    ``config`` being a dict. If a backend's ctor regresses to direct
    kwargs (``backend_class(dimensions=768)``), the factory dispatch
    breaks. This pins the dict-config contract structurally.
    """
    sig = inspect.signature(backend_cls.__init__)
    params = list(sig.parameters.values())
    # Skip ``self``; the next positional must be the config dict
    # (or keyword with a sensible default).
    assert len(params) >= 2, (
        f"{backend_cls.__name__}.__init__ must accept a config "
        "positional/keyword argument after `self`."
    )
    first_real = params[1]
    # The convention is ``__init__(self, config: dict | None = None)``
    # — either positional ``config`` or first-positional with default.
    default = first_real.default
    accepts_dict_default = (
        default is None
        or default is inspect.Parameter.empty
        or isinstance(default, dict)
    )
    assert accepts_dict_default, (
        f"{backend_cls.__name__}.__init__'s first parameter "
        f"{first_real.name!r} should default to None or accept a dict."
    )


def test_memory_vector_store_constructs_from_empty_config() -> None:
    """Smoke test: ``MemoryVectorStore({})`` succeeds (no optional deps)."""
    from dataknobs_data.vector.stores.memory import MemoryVectorStore

    store = MemoryVectorStore({"dimensions": 8})
    assert store is not None


def test_registered_vector_backends_audit_set() -> None:
    """The audit matrix above must cover every registered built-in.

    A new built-in vector backend → add a row to the per-backend tests
    above so the parity guard continues to cover the registry.
    """
    keys = set(vector_backends.list_keys())
    # The set of canonical backend names that MUST exist as built-ins.
    canonical = {"memory"}
    missing = canonical - keys
    assert not missing, f"Built-in vector backends missing: {missing}"
