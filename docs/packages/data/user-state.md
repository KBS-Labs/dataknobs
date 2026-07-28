# User State Coordinator

`UserStateStore` and `AsyncUserStateStore` coordinate a user's state **across
sessions**. They scope an injected database by `(namespace, tenant, user_id,
section)` so a single backend can hold isolated per-user state for many users
and tenants, with optimistic-concurrency writes, whole-user snapshot/erasure,
and delta events.

The coordinator ships **zero** domain sections — you declare every section your
application needs. It is backend-agnostic (any `dataknobs-data` backend) and
comes in symmetric sync and async variants.

## Concepts

State is organized into named **sections**. Each section has a **kind**:

| Kind | Records per user | Addressed by | Use for |
|------|------------------|--------------|---------|
| `document` | exactly one | a derived deterministic id | settings, a profile, a consent ledger |
| `collection` | many | backend-generated ids, read by filter | events, notes, interactions |

Every section also carries governance metadata — a `Sensitivity`
classification (`PUBLIC` / `INTERNAL` / `SENSITIVE`), an optional
`consent_scope`, a `retention_days` window, and a schema `version`.

## Configuration

```python
from dataknobs_data.user import (
    UserStateStoreConfig, UserStateSectionSpec, SectionKind, Sensitivity,
)

config = {
    "backend": "memory",          # any dataknobs-data backend key
    "namespace": "acme",          # isolates this store's records on a shared backend
    "sections": [
        {"name": "preferences", "kind": "document"},
        {"name": "profile", "kind": "document", "sensitivity": "sensitive"},
        {"name": "alerts", "kind": "collection"},
    ],
}
```

`UserStateStoreConfig` fields:

| Field | Default | Purpose |
|-------|---------|---------|
| `backend` | `"memory"` | Backing database backend, built only when no database is injected. |
| `namespace` | `"user_state"` | Logical namespace isolating this coordinator's records; feeds the derived document id and the default single-tenant context. |
| `sections` | `()` | The declared sections (`UserStateSectionSpec` list). |
| `enable_event_log` | `False` | Reserved flag for a persisted audit section (inert in the base coordinator). |

`UserStateSectionSpec` fields: `name`, `kind` (`SectionKind`), `schema`,
`sensitivity` (`Sensitivity`, default `INTERNAL`), `version` (default `1`),
`consent_scope`, `retention_days`. `consent_scope`, `retention_days`, and
`version` are stamped/carried but not enforced by the base coordinator; they are
reserved for governance enforcement.

## Constructing

Build from config — the async variant builds its own backing database:

```python
from dataknobs_data.user import AsyncUserStateStore, UserStateStore

store = await AsyncUserStateStore.from_config(config)   # async
store = UserStateStore.from_config(config)              # sync
```

Or inject a pre-built, shared database (the coordinator does **not** own it and
leaves it open on `close()`):

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase

db = AsyncMemoryDatabase()
store = AsyncUserStateStore.from_components(
    UserStateStoreConfig.from_dict(config), db=db,
)
```

Ownership follows the standard dataknobs convention: a config-built database is
owned and closed by `close()`; an injected one is caller-owned and left open, so
several coordinators can share one backend and each close independently.

## Document sections

```python
await store.put_document("user-42", "preferences", {"theme": "dark"})
record = await store.get_document("user-42", "preferences")
record.get_value("theme")           # "dark"
```

`get_document` returns `None` when the user has no record in that section.

## Collection sections

```python
record_id = await store.add_record("user-42", "alerts", {"text": "welcome"})
rows = await store.query("user-42", "alerts")               # only this user's records
```

`query` accepts an optional `dataknobs_data.Query` to add payload filters, sort,
or pagination — the user + section (+ bound tenant) scope is AND-composed
automatically:

```python
from dataknobs_data import Filter, Operator, Query

q = Query(filters=[Filter("level", Operator.EQ, "high")])
rows = await store.query("user-42", "alerts", q)
```

`update_record` and `delete_record` operate by record id and are
**scope-checked**: a record id belonging to another user cannot be updated
(which would re-tag it) or deleted through the coordinator.

## Optimistic concurrency (compare-and-set)

Reads expose a version token; pass it back as `expected_version` to make a write
conditional. A stale token raises `dataknobs_common.exceptions.ConcurrencyError`
instead of silently overwriting a concurrent change.

```python
token = await store.document_version("user-42", "preferences")
await store.put_document(
    "user-42", "preferences", {"theme": "light"}, expected_version=token,
)

# For collection records (scope-checked — an out-of-scope id returns None):
token = await store.record_version("user-42", "alerts", record_id)
await store.update_record(
    "user-42", "alerts", record_id, {"text": "seen"}, expected_version=token,
)
```

The coordinator advertises `Capability.CONDITIONAL_WRITE`.

## Whole-user operations

```python
view = await store.snapshot("user-42")                 # {section_name: data}
view = await store.snapshot("user-42", include_sensitive=True)
count = await store.clear("user-42")                   # right-to-erasure
```

`snapshot` returns a view keyed by section name — document sections map to their
payload dict (or `None`), collection sections to a list of payload dicts.
Coordinator-owned fields are stripped. `SENSITIVE` sections are **omitted** by
default; pass `include_sensitive=True` to include them. `clear` deletes every
record for the user across all sections and returns the count deleted.

## Tenant scoping

Inject a `BoundTenantContext` to isolate state per tenant on a shared backend.
Two coordinators bound to different tenants see disjoint state even for the same
`user_id`:

```python
from dataknobs_common.tenancy import BoundTenantContext

t1 = AsyncUserStateStore.from_components(
    cfg, db=db, tenant=BoundTenantContext("tenant-1", "acme"),
)
t2 = AsyncUserStateStore.from_components(
    cfg, db=db, tenant=BoundTenantContext("tenant-2", "acme"),
)
```

Reads AND-compose the bound tenant, with **explicit-filter-wins** semantics: an
admin passing an explicit `tenant_id` filter reads across tenants. The
coordinator advertises `Capability.TENANT_SCOPED_STATE`.

## Delta events

Every successful write fires a **metadata-only** event
(`user_state:section_written`) on an in-process callback registry — section
values are never emitted, so a `SENSITIVE` section's contents cannot leak into an
observer. The payload does carry the `user_id` for routing, so treat the event
stream with the same care as that identifier. Inject an `event_bus` to fan the
events out across replicas:

```python
from dataknobs_common.events import InMemoryEventBus

bus = InMemoryEventBus(); await bus.connect()
store = await AsyncUserStateStore.from_config(config, event_bus=bus)
```

Fan-out is non-load-bearing observability: a failing subscriber is isolated and
never aborts the write.

`EventBus` fan-out is an **async-variant capability**: `EventBus.publish` is a
coroutine, and the sync `fire` path cannot drive it safely from within a running
loop. The sync `UserStateStore` therefore **rejects an injected `event_bus` at
construction** — use `AsyncUserStateStore` for bus fan-out, or register sync
callbacks on `store._callbacks` directly for in-process observation.

## Opacity-safe user ids

The `user_id` is treated as fully opaque. Document ids are derived from a
length-delimited hash of the scope tuple and scoping is filter-based, so an id
containing `/`, `://`, or any other character is structurally safe — it is only
ever a hash input or a filter value, never split into a delimited key.

## Sync vs async

The two variants are behavioral mirrors sharing the same pure scoping helpers.
Every async method has a synchronous twin. Use the sync `UserStateStore` in
synchronous contexts and `AsyncUserStateStore` under an event loop. The one
capability that is async-only is `EventBus` fan-out (see [Delta
events](#delta-events)); the sync store supports in-process callbacks but
rejects an injected `event_bus`.

## Record identity is coordinator-owned

The coordinator owns each record's storage identity — document ids derive from
the scope tuple, collection ids are backend-generated. A section payload may not
carry a storage-identity key (`id`, `storage_id`, `_id`, `record_id`); one is
rejected with a `ValueError`. This keeps identity under the coordinator's
control and avoids a backend-dependent divergence (some backends key a
collection `create` off a payload `id`, others mint a fresh id). Rename such a
field (e.g. `alert_id`) in your payload.
