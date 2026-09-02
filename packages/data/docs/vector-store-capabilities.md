# Vector Store Capabilities

Not every vector-store method exists on every backend. `save()` and
`load()` are meaningless for a store the service persists continuously;
`create_index()` has no analogue in a brute-force store. Before this,
reaching one of those methods through the `VectorStore` abstraction
meant an `isinstance` downcast — and a consumer that downcasts has
stopped being portable, which is what the abstraction was for.

The family implements `CapabilityContract` (from
`dataknobs_common.capabilities`), so the question has an answer:

```python
from dataknobs_common import Capability

if store.supports(Capability.VECTOR_PERSIST):
    await store.save()
else:
    # This store keeps its data somewhere save() would not reach.
    ...
```

## The matrix

<!-- capability-matrix:start -->

| Backend | `VECTOR_PERSIST` | `VECTOR_INDEX_TUNING` |
|---|---|---|
| `MemoryVectorStore` | with `persist_path` | — |
| `FaissVectorStore` | with `persist_path` | — |
| `ChromaVectorStore` | — | — |
| `PgVectorStore` | — | yes |

<!-- capability-matrix:end -->

`yes` is advertised on the class and holds for every instance.
`with persist_path` is advertised per instance, and only when one is
configured. `—` is never advertised.

The table is not maintained by hand:
`tests/vector/stores/test_capability_advertisement.py` rebuilds it from
the classes and from live instances, and fails if this file disagrees.
A backend added to the family without a row here fails that test, which
is the point — prose is what let four cross-backend divergences
accumulate under a promise nobody was checking.

## Why `VECTOR_PERSIST` is answered per instance

`save()` and `load()` return early when `persist_path` is unset. A
`MemoryVectorStore` built without one therefore persists exactly as much
as a server-backed store does: not at all. The methods are still
*present*, so the failure mode of getting this wrong is the bad kind —
a caller who checked by reading the class, or by `hasattr`, gets a
silent no-op rather than an error, and finds out when the data is gone.

So the capability is computed from instance state rather than declared
on the class:

```python
store = MemoryVectorStore({"dimensions": 768})
store.supports(Capability.VECTOR_PERSIST)          # False

store = MemoryVectorStore({"dimensions": 768, "persist_path": "/data/vectors.pkl"})
store.supports(Capability.VECTOR_PERSIST)          # True
```

`persist_path` lives on the shared `VectorStoreConfig`, so every backend
accepts the field. Accepting it is not the same as honouring it —
Chroma answers `False` however it is configured, because its data lives
in its own client's store and nothing `save()` could write would be the
row.

## Querying without an instance

Capabilities that do not depend on configuration are also readable from
the class, which is useful when choosing a backend before building one:

```python
from dataknobs_data.vector.stores.pgvector import PgVectorStore

Capability.VECTOR_INDEX_TUNING in PgVectorStore.supported_capabilities()  # True
```

`supported_capabilities()` is the **class-level** answer and deliberately
excludes anything instance-scoped, so it never returns a `True` a given
instance would contradict. For the full picture on a store you hold, use
`instance_capabilities()` or `supports()`.

## Adding a capability to a backend

`CapabilityMixin` does **not** auto-union across the MRO. A backend
declaring its own set must union the ABC's, or everything the ABC
declares silently drops out of that backend:

```python
class MyVectorStore(VectorStore):
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = (
        VectorStore.SUPPORTED_CAPABILITIES | {Capability.VECTOR_INDEX_TUNING}
    )
```

For a capability that depends on construction, override
`_compute_instance_capabilities()` instead — `VectorStore` inherits
`DynamicCapabilityMixin`, and
`PathPersistedCapabilityMixin` is the in-tree example.

Consumer-defined capabilities need no enum member: `supports()` accepts
a raw string, so an out-of-tree backend can advertise its own vocabulary
without a change here.
