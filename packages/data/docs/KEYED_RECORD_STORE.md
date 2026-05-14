# Keyed Record Store

`AsyncKeyedRecordStore[T]` and `SyncKeyedRecordStore[T]` are generic
id-keyed record-store wrappers over `AsyncDatabase` / `SyncDatabase`
designed for registry / pointer-table use cases.

They encapsulate `Record` construction so the two-column shape (the
`data` column and the `metadata` column) is preserved **by
construction**: the serializer signature is
`(T) -> tuple[dict, dict]` — a tuple of `data` and `metadata`
dicts, not a `Record`.  The metadata channel is part of the function's
*type*, so a consumer cannot accidentally route metadata into the
data column or omit metadata entirely.

Use this instead of building `Record(...)` inline whenever a backend
is being used as an id-keyed key/value store with structured payloads.

## When to use

- You have a typed value `T` keyed by `str` (`bot_id`, `artifact_id`,
  `rubric_id`, `generator_id`, …).
- You want to persist the value with cross-cutting context
  (`tenant_id`, audit info, feature flags) in the backend's
  `metadata` column so it can be filtered independently of the
  structural payload.
- You want a uniform CRUD + list / count / stream / batch surface
  with typed-channel filters
  (`filter_data=` for the `data` column, `filter_metadata=` for the
  `metadata` column).

## Example

```python
from dataclasses import dataclass

from dataknobs_data import AsyncKeyedRecordStore, Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase


@dataclass
class Registration:
    bot_id: str
    config: dict
    status: str
    tenant_id: str | None = None


def to_columns(r: Registration) -> tuple[dict, dict]:
    """Split the value across the two backend columns.

    The (data, metadata) tuple signature is load-bearing: a future
    change to ``Registration`` that adds a new cross-cutting field
    can't accidentally drop it into the data column without a
    type-visible diff at this site.
    """
    data = {"bot_id": r.bot_id, "config": r.config, "status": r.status}
    metadata = {"tenant_id": r.tenant_id} if r.tenant_id else {}
    return data, metadata


def from_record(record: Record) -> Registration:
    return Registration(
        bot_id=record.get_value("bot_id"),
        config=record.get_value("config"),
        status=record.get_value("status"),
        tenant_id=record.metadata.get("tenant_id"),
    )


db = AsyncMemoryDatabase()
await db.connect()

store = AsyncKeyedRecordStore[Registration](
    db, serializer=to_columns, deserializer=from_record,
)

await store.put("bot-a", Registration("bot-a", {}, "active", "t1"))
await store.put("bot-b", Registration("bot-b", {}, "active", "t2"))

# Typed-channel filters — `filter_metadata` AND-combines with `filter_data`.
t1_regs = await store.list(filter_metadata={"tenant_id": "t1"})
assert [r.bot_id for r in t1_regs] == ["bot-a"]

# Pagination + sort push down to the database query.
from dataknobs_data import SortSpec, SortOrder

page = await store.list(
    sort=[SortSpec(field="bot_id", order=SortOrder.ASC)],
    limit=10,
    offset=0,
)

# Count routes through ``AsyncDatabase.count(query)`` so backends
# with pushdown counts benefit transparently.
n = await store.count(filter_metadata={"tenant_id": "t1"})
assert n == 1
```

## Surface

| Method | Purpose |
|--------|---------|
| `put(key, value)` | Insert or update the record for `key`. |
| `get(key)` | Return the value for `key`, or `None`. |
| `exists(key)` / `delete(key)` | Existence check / delete. |
| `put_batch` / `get_batch` / `delete_batch` | Batched variants. |
| `clear()` | Delete every record in the underlying database.  Returns count. |
| `list(*, filter_data=None, filter_metadata=None, sort=None, limit=None, offset=None, vector_query=None)` | List matching values; channels AND-combine. |
| `count(*, filter_data=None, filter_metadata=None)` | Count matching records.  Routes through `AsyncDatabase.count(query)`. |
| `stream(*, filter_data=None, filter_metadata=None, config=None)` | Async-iterator surface for large populations. |
| `search(query)` | Escape hatch — returns raw `Record` instances; use when you need `Query` / `ComplexQuery` / vector scores. |
| `.db` | Read-only handle to the underlying database for lifecycle / schema inspection. |

`SyncKeyedRecordStore[T]` mirrors the async surface for synchronous use.

## Consumers in the dataknobs stack

This abstraction is used by:

- `dataknobs_bots.registry.DataKnobsRegistryAdapter` — single record-
  construction site for `Registration`; the
  `(Registration) -> (data, metadata)` serializer ensures the
  metadata channel cannot be silently dropped on storage shape changes.
- `dataknobs_bots.artifacts.ArtifactRegistry`,
  `dataknobs_bots.rubrics.RubricRegistry`,
  `dataknobs_bots.generators.GeneratorRegistry` — same pattern; all
  three registries inherit the `filter_metadata=` / `sort=` /
  `limit=` / `offset=` surface via this store.  `GeneratorRegistry`
  historically fell into the exact defect class this store prevents:
  a local variable named `metadata` was passed positionally to
  `Record(metadata)`, but `Record`'s positional arg-0 is `data`, so
  schema fields were silently routed to the wrong column.  The
  `(T) -> (data, metadata)` serializer signature now makes that kind
  of routing error a type-visible diff at the single construction
  site.
- `dataknobs_fsm.storage.database.UnifiedDatabaseStorage.save_step` —
  composes `AsyncKeyedRecordStore[_StepRecord]`, which gives the
  step's new `metadata=` kwarg a typed channel from caller down to
  the underlying record's metadata column on day one (previously the
  method had no `metadata` parameter at all — see
  `packages/fsm/CHANGELOG.md`).  The serializer signature makes the
  metadata channel part of the function's type, so future additions
  to `_StepRecord` can't silently regress to the variable-shadow
  pattern that `GeneratorRegistry` fell into.

## Why the (data, metadata) tuple signature?

Routing through this store is a **structural prevention contract**.
Building a `Record` inline (`Record({...})` plus optional
`record.metadata = {...}`) makes the metadata channel a
soft convention; it's easy to forget the second statement, and the
caller doesn't see anything wrong until a downstream consumer tries
to filter by metadata and gets zero results.

Making the serializer return `tuple[dict, dict]` instead of
`Record` moves the contract into the type system.  A future change
to `T` that adds a cross-cutting field can't land in the wrong
column without a type-visible diff at the serializer site, and code
review catches the diff because it's load-bearing structure rather
than a forgettable line.
