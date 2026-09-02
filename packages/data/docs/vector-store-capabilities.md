# Vector Store Capabilities

Not every vector-store operation works on every backend. `save()` and
`load()` are meaningless for a store whose service persists
continuously; `create_index()` has no analogue in a brute-force store;
handing raw text to the store and letting *it* choose the embedding
model is one backend's bargain and not the family's.

Two things are needed to make that usable, and they only work together.
The family implements `CapabilityContract` (from
`dataknobs_common.capabilities`), so the question has an answer — and
the guarded methods live on `VectorStore` itself, so the answer can be
acted on without an `isinstance` downcast:

```python
from dataknobs_common import Capability

if store.supports(Capability.VECTOR_PERSIST):
    await store.save()
else:
    # This store keeps its data somewhere save() would not reach.
    ...
```

Skip the check and a backend that cannot do the thing raises
`CapabilityNotSupportedError`, following `AsyncDatabase.begin_transaction`:
the caller learns on the first call rather than on the first restore.

## The matrix

<!-- capability-matrix:start -->

| Backend | `VECTOR_PERSIST` | `VECTOR_INDEX_TUNING` | `VECTOR_DOCUMENT_API` |
|---|---|---|---|
| `MemoryVectorStore` | with `persist_path` | — | — |
| `FaissVectorStore` | with `persist_path` | — | — |
| `ChromaVectorStore` | — | — | yes |
| `PgVectorStore` | — | yes | — |

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

A `MemoryVectorStore` built without a `persist_path` persists exactly as
much as a server-backed store does: not at all. Since the methods are on
the ABC, `hasattr` answers `True` for every backend and settles nothing
— which is the point, but it means probing for the method instead of
asking for the capability tells you less than nothing.

If the store is held behind an untyped attribute — a backend a consumer
handed in, a plugin loaded by name — ask through
`supports_capability(store, ...)` from `dataknobs_common` instead of
`store.supports(...)`. It reads an object that does not implement the
contract at all as "cannot", which is the same reading the raising guard
gives it, rather than raising `AttributeError` on the check itself.

There are two ways a store can decline to persist and **they behave the
same**: no `VECTOR_PERSIST`, no snapshot, `CapabilityNotSupportedError`.

| | Why it declines |
|---|---|
| Chroma, pgvector | the rows live in a service; there is no snapshot to take |
| Memory, FAISS with no `persist_path` | the class can snapshot; this instance has nowhere to put one |

The second case used to return quietly, and that was the more dangerous
of the two — a request to persist answered with a successful-looking
call and nothing on disk to restore from. A caller that legitimately
does not know whether a store persists gates on the capability:

```python
if store.supports(Capability.VECTOR_PERSIST):
    await store.save()
```

which is what `MemoryVectorStore.initialize()` now does with
`persist_path` before calling `load()` — the shape
`FaissVectorStore.initialize()` already had.

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

A backend with its own configuration names it as a type parameter, and
`self.config` then has that type rather than the shared base:

```python
@dataclass(frozen=True)
class MyVectorStoreConfig(VectorStoreConfig):
    endpoint: str = ""


class MyVectorStore(VectorStore[MyVectorStoreConfig]):
    CONFIG_CLS: ClassVar[type[MyVectorStoreConfig]] = MyVectorStoreConfig

    def _setup(self) -> None:
        super()._setup()
        self.endpoint = self.config.endpoint     # typed, not `Any`
```

Leaving it unparameterized — `VectorStore`, as an annotation naming the
family rather than one backend — behaves exactly as before. The parameter
is what lets a backend read its own fields off `self.config`; without it
the checker sees only `VectorStoreConfig` and reports every backend field
as an attribute that does not exist. That is what pgvector, Chroma and
FAISS reported until they named theirs; `MemoryVectorStore` never did,
because its config adds no fields to read.

Consumer-defined capabilities need no enum member: `supports()` accepts
a raw string, so an out-of-tree backend can advertise its own vocabulary
without a change here.

## `VECTOR_DOCUMENT_API` is not the portable way to store text

This one is easy to misread as "only Chroma can store documents", and
that is not what it says. Every backend can:

```python
await store.bulk_embed_and_store(texts, embedder=my_embedder, ids=ids)
```

`bulk_embed_and_store` is on `VectorStore`, so it is the family's
portable text-to-vector path, and it is the one to reach for by default.

What `VECTOR_DOCUMENT_API` marks is a genuinely different bargain.
Chroma's `add_documents()` / `search_documents()` embed **server-side**,
using the embedding function its client was configured with. That is
convenient and it costs something specific: the model is the store's, so
the caller neither chooses it nor records which one produced a given
vector. A store whose rows were written that way cannot answer "are
these vectors stale against the current model?", because nothing wrote
down which model it was — whereas `bulk_embed_and_store` records the
model identity in each row's metadata.

So the capability is worth advertising in both directions. A consumer
that wants the convenience can find the one backend offering it; a
consumer that needs provenance can tell it is about to lose it.
