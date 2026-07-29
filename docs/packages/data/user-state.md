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
| `enable_event_log` | `False` | When set, appends a metadata-only record to a reserved `events` audit section after every write and scoped deletion; read it with `query_events` (see [Persisted event log](#persisted-event-log)). |
| `event_log_retention_days` | `None` | Retention window (days) for the reserved `events` audit section; a section-less `prune` ages it out. `None` = unbounded until `clear`. Positive only. |
| `prune_on_query` | `False` | When set, `query` prunes a windowed collection section's expired records for the queried user before returning (see [Lifecycle: retention pruning](#lifecycle-retention-pruning)). |
| `persist_migrations` | `False` | When set, a record upgraded on read by its section's migration chain is written back under a compare-and-set guard (see [Schema versioning and migration](#schema-versioning-and-migration)). |

`UserStateSectionSpec` fields: `name`, `kind` (`SectionKind`), `schema`,
`sensitivity` (`Sensitivity`, default `INTERNAL`), `version` (default `1`),
`consent_scope`, `retention_days`. A non-`None` `consent_scope` gates the
section behind a consent grant (see [Governance: consent-gated
access](#governance-consent-gated-access)). A `retention_days` window ages out
records in a **collection** section (see [Lifecycle: retention
pruning](#lifecycle-retention-pruning)) — it is rejected on a document section,
which holds one evolving record per user and never expires. `version` drives lazy on-read schema migration (see [Schema
versioning and migration](#schema-versioning-and-migration)); it must be a
positive integer (versions start at `1`), and a zero or negative version is
rejected at config-load time.

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

## Governance: consent-gated access

A section can be placed behind a **consent scope**. Declare a `consent_scope` on
the section and the coordinator refuses reads and writes to it until the user
grants that scope:

```python
config = {
    "namespace": "acme",
    "sections": [
        {"name": "preferences", "kind": "document"},
        {"name": "analytics", "kind": "collection",
         "consent_scope": "analytics_processing"},
    ],
}
store = await AsyncUserStateStore.from_config(config)

# Ungranted access is refused (fail-closed):
await store.add_record("user-42", "analytics", {"event": "click"})
# -> dataknobs_common.exceptions.ConsentRequiredError

# Grant the scope, then the section is accessible:
await store.grant_consent("user-42", "analytics_processing")
await store.add_record("user-42", "analytics", {"event": "click"})   # ok
```

- **Scope, not section, granularity.** A `consent_scope` is a named scope shared
  across sections; one `grant_consent(user, scope)` unlocks every section tagged
  with it.
- **Fail-closed.** A missing consent document, a missing scope grant, or a
  revoked scope all refuse access, raising
  `dataknobs_common.exceptions.ConsentRequiredError` on a direct `get_document`
  / `query` / write.
- **`snapshot` omits rather than raises.** An ungranted consent-scoped section is
  simply absent from `snapshot()` output (like a `SENSITIVE` section), so a
  whole-user view never fails on a partially-consented user.
- **Revocation is block-only.** `revoke_consent(user, scope)` refuses future
  access but leaves the stored data in place; a later `grant_consent` surfaces
  it again. Erasure remains the explicit `clear(user)`.
- **Erasure is never gated.** `clear(user)` removes consent-scoped data even
  while the scope is revoked — data minimization must always be possible.

```python
await store.has_consent("user-42", "analytics_processing")     # bool
await store.revoke_consent("user-42", "analytics_processing")  # block access
```

Grants are stored in a reserved, coordinator-managed `consent` document section,
so `consent` is a reserved section name — declaring your own section named
`consent` is a `ConfigurationError`. The reserved section is also unreachable
through the content API: `get_document` / `put_document` / `query` / `add_record`
(and the version accessors) on `"consent"` raise `ConfigurationError`, so a
caller cannot forge a grant or clobber the ledger by writing it directly —
grants flow only through `grant_consent` / `revoke_consent`. The consent helpers
are available only when at least one declared section carries a `consent_scope`.

## Lifecycle: retention pruning

A **collection** section can declare a `retention_days` window. Records whose
`_written_at` stamp is older than the window are removed by `prune`:

```python
config = {
    "namespace": "acme",
    "sections": [
        {"name": "activity", "kind": "collection", "retention_days": 30},
    ],
}
store = await AsyncUserStateStore.from_config(config)

removed = await store.prune("user-42", "activity")   # count deleted
removed = await store.prune("user-42")               # every windowed section
```

- **Explicit by default — the consumer schedules it.** DataKnobs is a library,
  not a daemon: `prune(user_id, section=None)` runs when you call it (a
  background job, a login hook, a cron). With `section=None` every collection
  section carrying a `retention_days` window is pruned; naming a document,
  unknown, or reserved section raises.
- **Opt-in lazy pruning.** Set `prune_on_query: true` and a `query` of a
  windowed section first prunes that user's expired records in it, so a read
  never returns aged-out data. Off by default (a read has no write side effect).
- **Collection sections only.** A document section holds one evolving record per
  user and never expires; a `retention_days` on a document section is a
  `ConfigurationError` at config load.
- **Positive windows only.** `retention_days` must be a positive number of days.
  A zero or negative window is a `ConfigurationError` at config load — it would
  mark live records as already expired and delete them on the next `prune`, so a
  mis-signed window is caught at the boundary rather than silently destroying
  data.
- **Not consent-gated.** Pruning is data minimization, so — like `clear` — it is
  never blocked by a consent scope.

Time is measured against an injected clock. By default the coordinator uses
wall-clock UTC; inject a `now` collaborator (a `Callable[[], datetime]`) to make
retention deterministic in tests or drive it from an external clock:

```python
from datetime import datetime, timezone

store = await AsyncUserStateStore.from_config(
    config, now=lambda: datetime.now(timezone.utc),
)
```

Prefer a **timezone-aware** clock (as above): `_written_at` stamps are recorded
with whatever awareness the clock returns, and expiry compares the stamp against
`now`. A record whose stamp cannot be confidently compared to `now` — a missing
or unparseable stamp, or an aware/naive timezone mismatch between the stamp and
the clock — is treated as *not* expired and left in place rather than crashing
the prune. Keeping the clock consistently aware avoids that edge entirely.

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

A "write" here is a create, update, or consent grant/revoke — the operations
that record new or changed state.

Deletions and erasure fire a **sibling** metadata-only event
(`user_state:section_deleted`) on the same registry, so a consumer can build a
deletion or erasure audit trail from the event stream. One event fires per
delete-method call, discriminated by an `op` field naming the method:

| `op` | fired by | payload |
|---|---|---|
| `"delete_record"` | a single scoped collection delete | `record_id`, `count == 1`, the `section` name |
| `"prune"` | a retention sweep (explicit or lazy `prune_on_query`) | `count` (records removed), the `section` name — or `None` for a section-less `prune`, which also carries a `sections` map |
| `"clear"` | the whole-user right-to-erasure primitive | `count` (total removed), `section == None` |

The delete payload is metadata-only *by construction*: every delete path removes
records by id and never reads a record's payload, so no section value is ever
available to emit — a `SENSITIVE` section is safe. `section == None` is the
signal for a whole-user / multi-section operation, so a consumer keys erasure
handling off `op == "clear"` rather than off `section`. A **section-less**
`prune` (`prune(user_id)`) can age out several windowed collections in one call;
its event adds a `sections` field — a `{section_name: removed_count}` map — so an
erasure-audit consumer gets the per-section split while `count` stays the total.
A single-section `prune` names its target in `section` and omits `sections`. An
event fires **only when data was actually removed** — a no-op delete, an empty
prune, and a clear of an empty user emit nothing.

```python
from dataknobs_common.events import InMemoryEventBus

bus = InMemoryEventBus(); await bus.connect()
store = await AsyncUserStateStore.from_config(config, event_bus=bus)
```

Both topics ride the same registry and fan-out. Fan-out is non-load-bearing
observability: a failing subscriber is isolated and never aborts the write or
delete.

`EventBus` fan-out is an **async-variant capability**: `EventBus.publish` is a
coroutine, and the sync `fire` path cannot drive it safely from within a running
loop. The sync `UserStateStore` therefore **rejects an injected `event_bus` at
construction** — use `AsyncUserStateStore` for bus fan-out, or register sync
callbacks on `store._callbacks` directly for in-process observation.

## Persisted event log

The delta events above are *ephemeral* — an in-process notification with no
storage. Set `enable_event_log` for a **persisted** per-user audit trail: the
coordinator registers a reserved `events` collection section and appends one
**metadata-only** record to it after every data write and scoped deletion. Read
it with `query_events`:

```python
config = UserStateStoreConfig.from_dict({
    "namespace": "acme",
    "enable_event_log": True,
    "event_log_retention_days": 90,   # optional; unbounded when omitted
    "sections": [{"name": "activity", "kind": "collection"}],
})
store = await AsyncUserStateStore.from_config(config)

await store.add_record("user-42", "activity", {"event": "login"})
events = await store.query_events("user-42")
# [Record(op="add_record", op_section="activity", op_record_id=..., _written_at=...)]
```

Each record carries the operation metadata under `op`-prefixed keys — never a
section value, so a `SENSITIVE` section's contents cannot leak into the log:

| field | meaning |
|---|---|
| `op` | the logged operation (`put_document` / `add_record` / `update_record` / `delete_record` / `prune`) |
| `op_section` | the section the operation targeted, or `None` for a section-less `prune` |
| `op_record_id` | the record id for a single-record operation |
| `op_count` | the number removed for a delete operation |
| `op_sections` | the `{section: removed_count}` split of a section-less `prune` |

The record's own `_written_at` scope stamp is the audit timestamp.

- **Read-only, coordinator-written.** The reserved `events` section is walled off
  from the generic content API (a `put_document` / `query` of `"events"` raises),
  so a consumer cannot forge or clobber audit entries; records are appended only
  by the coordinator's own write/delete paths and read only through
  `query_events`. A `query_events` on a store without `enable_event_log` raises
  `ConfigurationError`.
- **A refused write logs nothing.** A consent gate raises before the primary
  write, so a refused write never reaches the append — the log records completed
  operations only.
- **Erasure leaves no trace.** `clear` (right-to-erasure) deliberately appends
  **no** record: it erases the log along with the rest of the user's state, and
  re-materialising a `clear` record into the just-erased section would defeat the
  erasure. The *ephemeral* `user_state:section_deleted` event still fires for
  real-time audit.
- **Consent changes are ephemeral-only.** Consent grants / revocations fire the
  ephemeral write event but are not appended to the persisted log — it captures
  the consumer's data operations.
- **Retention.** Set `event_log_retention_days` to age the log out through the
  ordinary section-less `prune` sweep; without it the log grows until `clear`.
- **Best-effort, non-atomic.** The append is a second write after the primary one
  persists (mirroring the ephemeral fire); no cross-record transaction spans the
  two. A backend failure appending the audit entry is logged and swallowed, never
  propagated — the primary operation already succeeded, so raising would
  spuriously fail (and, on a retry, duplicate) it. The trade-off is that the log
  may miss an entry for an operation that did persist; `clear`-scoped erasure and
  the ephemeral delta event are unaffected.

The log is tenant-scoped like every other section — `query_events` under a bound
tenant returns only that tenant's records.

## Schema versioning and migration

Every section carries a schema `version` (default `1`), stamped onto each written
record as `_section_version`. When a section's payload shape evolves, bump its
`version` and register the pure per-version upgraders that rewrite an older
record's payload forward. A read that surfaces a record stamped behind the
section's current version applies the registered chain **in memory** before
returning it:

```python
from dataknobs_data import register_section_migrator

def _v1_to_v2(payload):
    out = dict(payload)
    out["theme"] = out.pop("color", "light")   # a renamed field
    return out

register_section_migrator("preferences", 1, _v1_to_v2)

config = UserStateStoreConfig.from_dict({
    "namespace": "acme",
    "sections": [{"name": "preferences", "kind": "document", "version": 2}],
})
store = await AsyncUserStateStore.from_config(config)

# A record written by an older (version 1) deployment is upgraded on read:
record = await store.get_document("user-42", "preferences")
# record.get_value("_section_version") == 2, payload in the version-2 shape
```

An upgrader is a pure `Callable[[Mapping], Mapping]`: it receives the record's
consumer payload — never the coordinator's scope stamps — and returns the next
version's payload. The boundary is symmetric: any reserved scope stamp the
upgrader returns (`_section_version`, `_written_at`, `tenant_id`, `user_id`,
`section`) is stripped from its output, so the coordinator's own re-stamp is the
sole authority on those fields — a buggy upgrader cannot leak or forge one.
Register one per step (`v1 -> v2`, `v2 -> v3`, …); the coordinator composes the
chain for whatever gap a given record has.

- **Lazy, in memory by default.** Migration runs on `get_document`, `query`, and
  `snapshot` (which reads through them). The stored record is left untouched
  unless `persist_migrations` is set, and the migrated record keeps its original
  `_written_at` stamp — a read never resets the retention clock.
- **Persist-on-read (opt-in).** Set `persist_migrations` to write the upgrade
  back once, under a compare-and-set guard. A concurrent write that advanced the
  record first wins the guard; the write-back is skipped and the in-memory
  upgrade is still returned (migrations are deterministic, so concurrent
  persists converge on the same content). The write-back is a representation
  upgrade — it emits no delta event and appends no audit record.
- **Rollback reads fail open.** A record stamped *newer* than the running spec (a
  rolled-back deployment reading records a newer one wrote) passes through
  un-migrated with a `WARNING` — the replica can read it but cannot down-convert
  it.
- **A missing step is a wiring bug.** If a record needs a version the registered
  chain cannot reach (no migrator for the section, or a gap between steps), the
  read raises `ConfigurationError` rather than returning a partially-upgraded
  record.

Migrators are registered in a process-global registry keyed by section name
(as with the other DataKnobs named registries), so a consumer wires each
section's chain once at import time. Registration is a non-atomic
read-modify-write of the registry, designed for import-time, single-threaded
wiring — it is **not** safe against concurrent registration of different steps
for the same section, so register each chain once at import, before any store
reads. The reserved `consent` and `events` sections are coordinator-owned and
never migrated: a migrator registered for one of those reserved names registers
fine but is **inert** — the on-read migration path never consults it.

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
