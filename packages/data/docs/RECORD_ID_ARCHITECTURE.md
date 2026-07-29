# Record ID Architecture

## Overview

The DataKnobs Data Package cleanly separates a user-defined identifier from the
system-assigned storage identifier, while giving the caller full control over
which value keys a record. This architecture keeps data integrity intact and
makes the storage key predictable across every backend.

## The Two-ID Concept

### 1. Storage ID (System ID)
- **Purpose**: The key a record is stored and addressed under
- **Source**: `record.id` when the caller supplies one (see the
  [Write-Keying Contract](#write-keying-contract)); a fresh UUID v4 is minted
  only when the record carries no id — a *falsy* id (`""`, or `0` before
  stringification) counts as none and is minted
- **Access**: Via `record.storage_id` property
- **Mutability**: Set once for a stored record; immutable thereafter

### 2. User ID (Data Field)
- **Purpose**: Application-specific identifier in the record's data
- **Location**: Stored as a field named `id` in the record's data
- **Format**: Any user-defined format (string, integer, etc.)
- **Access**: Via `record.get_user_id()` or `record.get_value("id")`
- **Mutability**: Can be changed by the application

## Write-Keying Contract

`create()` and `create_batch()` — **sync and async, on every backend** — derive
a record's storage id the same way:

> **The storage id is `record.id` (honor a caller-supplied id); a fresh UUID is
> minted only when the record carries no id. A colliding id fails closed with
> `DuplicateRecordError`. Use `upsert` to insert-or-overwrite.**

`record.id` resolves through a 5-step priority (see
[ID Priority Resolution](#id-priority-resolution)), so a business identifier
placed in the record's `id` (or `record_id`) data field **becomes the storage
key**:

```python
record = Record({"id": "user-123", "name": "Test"})
storage_id = await db.create(record)   # "user-123" — the caller's id is honored
```

A record with no resolvable id is minted a UUID (by the overridable
[`_generate_id()` hook](#minting-a-storage-id--the-_generate_id-hook) — override
it for a custom id scheme):

```python
storage_id = await db.create(Record({"name": "Test"}))  # e.g. "uuid-456"
```

A *falsy* id counts as "no id": the resolution is `record.id or <uuid>`, so an
empty or zero id (`Record({"id": ""})`, or a `0` id before stringification) is
treated as absent and a fresh UUID is minted rather than keying the record under
the falsy value. Supply a non-empty id when you need the caller value honored.

A second `create()` under the same id fails closed rather than overwriting:

```python
await db.create(Record({"id": "user-123", "name": "Test"}))
await db.create(Record({"id": "user-123", "name": "Other"}))  # DuplicateRecordError
await db.upsert(Record({"id": "user-123", "name": "Other"}))  # overwrites instead
```

This is the same contract for the single-record and batch forms, and for
insert-or-overwrite: a record keys identically whether written through
`create()`, `create_batch()`, `upsert()`, or `upsert_batch()` — including the
falsy-id rule, so `upsert(Record({"id": ""}))` mints rather than keying under
`""`, matching `create` and `upsert_batch`.

### Read + write coherence

Because `Filter("id", ...)` is **reserved to the storage key** on every backend
(see [Searching by a User-Defined Identifier](#searching-by-a-user-defined-identifier)),
honoring a caller-supplied `id` on write keeps read and write coherent: the id
you supply becomes the storage key **and** is what `Filter("id", ...)` matches.
If you want a business identifier that is *not* the storage key — a system UUID
for the key, with the identifier as pure business data — store it under a field
name **other than** `id` / `record_id` (see the recipe below).

### Security: validate a caller-supplied id you do not trust

Because `create()` honors `record.id` as the storage key on **every** backend, a
caller-supplied `id` chooses the record's key — including the S3 object key
(`{prefix}{id}.json`) and the file backend's JSON dict key. When the `id`
originates from untrusted input (a request payload, an uploaded document),
validate it at your boundary as you would any other external value (see the
project's input-validation-at-boundaries rule).

This is a **namespacing** concern, not an overwrite one: storage keys are flat,
so an `id` like `"../x"` is a literal key segment rather than a path traversal;
reads use the same key builder, so they stay symmetric; and the fail-closed
`create()` (S3's `If-None-Match: *`) prevents clobbering an existing record.
But an unvalidated `id` still lets caller input place a record outside your
intended key namespace, so treat it as boundary input. To keep the storage key
entirely under your control regardless of payload contents, set
`record.storage_id` explicitly (or store the untrusted identifier under a
non-`id` field name — see the recipe below).

## The ID Priority System

### Record Class Properties

```python
class Record:
    _storage_id: str | None  # System-assigned storage ID
    fields: dict             # User data (may include an "id" field)

    @property
    def storage_id(self) -> str | None:
        """Get the storage system ID (None until stored / assigned)."""
        return self._storage_id

    @property
    def id(self) -> str | None:
        """Get the record ID via the 5-step priority (see below)."""
        ...

    def get_user_id(self) -> str | None:
        """Explicitly get the user-defined 'id' data field (never the storage id)."""
        return self.get_value("id")

    def has_storage_id(self) -> bool:
        """Check if a storage ID has been assigned."""
        return self._storage_id is not None
```

### ID Priority Resolution

`record.id` returns the first of:

1. `storage_id` (database-assigned, once stored)
2. the legacy `_id` (set from a caller `id=`/`storage_id=` kwarg, or promoted
   from a payload id at construction)
3. a payload `id` data field
4. a metadata `id`
5. a payload `record_id` data field

…or `None` when none is present. Steps 3–5 are why a business identifier in the
data becomes the storage key on write. `record.get_user_id()` returns **only**
the payload `id` field, ignoring any assigned storage id.

## Implementation in Backends

### Centralized Helper Methods

The in-process and object-store backends resolve the write-keying rule through a
single base helper; the SQL backends apply the identical rule in their query
builders (`build_create_query` / `build_batch_create_query`). All express the
same `record.id or self._generate_id()` resolution, so no backend re-derives the
rule independently. The helpers live on `RecordStorageMixin`, a single class that
both `SyncDatabase` and `AsyncDatabase` inherit, so the write-keying rule — and
its mint hook — cannot drift between the sync and async trees.

```python
def _prepare_record_for_storage(self, record: Record) -> tuple[Record, str]:
    """Resolve a record's storage id for a write, honoring a caller id.

    The storage id is record.id (honor a caller-supplied id); a fresh id
    is minted via _generate_id() only when the record carries no id at all.
    """
    record_copy = record.copy(deep=True)
    storage_id = record.id or self._generate_id()
    record_copy.storage_id = storage_id
    return record_copy, storage_id

def _prepare_record_from_storage(self, record: Record | None, storage_id: str) -> Record | None:
    """Prepare a record retrieved from storage by ensuring storage_id is set."""
    if record:
        record_copy = record.copy(deep=True)
        if not record_copy.has_storage_id():
            record_copy.storage_id = storage_id
        return record_copy
    return None
```

### Minting a storage id — the `_generate_id()` hook

When a record carries no caller id, the storage id is minted by a single
overridable hook, `_generate_id()`, defined once on `RecordStorageMixin` and
inherited by every backend:

```python
def _generate_id(self) -> str:
    """Mint a fresh storage id for a record that carries no caller id."""
    return str(uuid.uuid4())
```

Every `create` / `create_batch` **and** `upsert` / `upsert_batch` mint fallback
routes through this hook — the base create helper above, the shared
`_resolve_upsert_id` preamble (the single-`upsert` id resolution), the SQL
create/upsert paths (which resolve `record.id or self._generate_id()` and pass
an `id_factory=self._generate_id` into `build_batch_create_query` /
`build_batch_upsert_query`), and the Postgres / Elasticsearch create/upsert
paths. It is the single extension point for a custom storage-id scheme: override
it once and every `create()` / `create_batch()` / `upsert()` / `upsert_batch()`
path on that backend mints via your implementation, uniformly. A caller-supplied
`record.id` is always honored — the hook governs only the mint fallback.

```python
import ulid
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

class UlidSQLiteDatabase(SyncSQLiteDatabase):
    def _generate_id(self) -> str:
        return str(ulid.new())   # every minted id is a ULID, on create and upsert
```

> **Scope — `create` and `upsert` paths.** The hook governs the mint fallback on
> `create` / `create_batch` / `upsert` / `upsert_batch`. `update` /
> `update_batch` never mint — every `update` takes an explicit id. A falsy
> (empty-string) record id is treated as absent and minted on both create and
> upsert (see the [Write-Keying Contract](#write-keying-contract)), so
> `upsert(Record(id=""))` keys under a freshly minted id rather than under `""`.

### Backend Usage Example

```python
# In any backend's create method
async def create(self, record: Record) -> str:
    # Resolve the storage id (honors record.id; mints only when absent)
    record_copy, storage_id = self._prepare_record_for_storage(record)

    # Store the record under its storage id, failing closed on a collision
    if storage_id in self._storage:
        raise DuplicateRecordError(storage_id)
    self._storage[storage_id] = record_copy
    return storage_id

# In any backend's read method
async def read(self, id: str) -> Record | None:
    record = self._storage.get(id)
    # Ensure the returned record carries its storage_id
    return self._prepare_record_from_storage(record, id)
```

## Usage Patterns

### Creating Records

```python
# Caller supplies an id in the data — it becomes the storage key
record = Record({"id": "user-123", "name": "Test"})
print(record.id)          # "user-123" (resolved from the data field)
print(record.storage_id)  # None (not yet stored)

storage_id = await db.create(record)
print(storage_id)         # "user-123" — the caller's id is honored

# Read it back by that key
retrieved = await db.read("user-123")
print(retrieved.id)            # "user-123"
print(retrieved.get_user_id()) # "user-123" (the data field is still present)
print(retrieved.storage_id)    # "user-123"
```

To let the backend mint the key and keep a business identifier as pure data,
store the identifier under a non-reserved field name:

```python
record = Record({"user_id": "user-123", "name": "Test"})
storage_id = await db.create(record)  # a minted UUID; "user_id" stays business data
```

### Updating Records

```python
# record.id returns the storage id once stored, so it updates the right row
retrieved.set_value("name", "Updated")
await db.update(retrieved.id, retrieved)
```

### Searching by a User-Defined Identifier

`Filter("id", ...)` is **reserved to the record's storage key** on every backend
— it does **not** search a `data` field named `id`. A "User ID" stored under
`data["id"]` (see [The Two-ID Concept](#the-two-id-concept)) is reachable through
the Record API (`record.get_user_id()` / `record.get_value("id")`) but is
**shadowed** for querying: `Filter("id", ...)` matches the storage key instead,
so a query against a *different* business value under a data `id` field silently
returns no rows.

To make a user-defined identifier **queryable independently of the storage key**,
store it under a field name **other than `id`** and filter that field directly:

```python
# Find records by a user-defined identifier — store it under a non-reserved name.
from dataknobs_data import Query, Filter, Operator, Record

await db.create(Record({"user_id": "user-123", "name": "Test"}))

query = Query(filters=[Filter("user_id", Operator.EQ, "user-123")])
results = await db.search(query)   # matches the data field "user_id"
```

If you want to filter on the storage key, `Filter("id", ...)` is exactly that —
and because the storage key honors a caller-supplied `id`, filtering
`Filter("id", "user-123")` finds the record created from
`Record({"id": "user-123", ...})`. See the query reference for the reserved-name
contract and the secondary-identifier recipe.

## Benefits

1. **Predictable keying**: a caller-supplied id is the storage key on every
   backend and both write methods; a UUID is minted only when none is supplied
2. **Fail-closed writes**: a colliding id raises `DuplicateRecordError` rather
   than silently overwriting; `upsert` is the explicit insert-or-overwrite path
3. **Read + write coherence**: the honored id is both the storage key and what
   `Filter("id", ...)` matches
4. **Backwards compatible reads**: `record.id` continues to return the right id
   before and after storage
5. **Explicit access**: `get_user_id()` gives unambiguous access to the data
   field regardless of the storage key

## Migration Guide

### Behavior to be aware of

- `create()` **honors a caller-supplied `record.id`** as the storage key
  (minting only when absent), matching `create_batch()`. A record whose data
  carries an `id` (or `record_id`) field is keyed under that value, and a
  colliding id fails closed with `DuplicateRecordError`.
- If you relied on `create()` minting a fresh UUID while a payload `id` field
  stayed pure business data, either store that identifier under a non-`id`
  field name (recommended — it is also queryable, see above) or set
  `record.storage_id` explicitly to the key you want.
- Use `upsert` where you previously depended on a second write overwriting the
  first (streaming/batched writes: `StreamConfig(on_conflict=ConflictPolicy.UPSERT)`).

### Best practices for new applications

1. Put the value you want as the storage key in `record.id` (via the data `id`
   field, or `record.storage_id = ...`).
2. Keep a queryable business identifier under a non-`id` field name.
3. Use `record.get_user_id()` when you need the user-defined `id` data field.
4. Use `record.id` for database operations (it returns the storage id once stored).
5. Check `record.has_storage_id()` to know whether a record has been stored.

## Technical Details

### Property Setter Handling

The Record class routes `record.id = ...` / `record.storage_id = ...` through
their property setters (rather than creating a data field), so assigning a
storage id never leaks into the record's business data.

### Database Utility Functions

The `database_utils` module provides:

```python
def ensure_record_id(record: Record, record_id: str) -> Record:
    """Ensure a record carries its storage ID (used when returning read results)."""
    if not record.has_storage_id() or record.storage_id != record_id:
        record = record.copy(deep=True)
        record.storage_id = record_id
    return record
```

This is used internally by backends when returning records from a read/search.

## See Also

- [Record Serialization Architecture](RECORD_SERIALIZATION.md) - How records with vector fields are serialized
- [Architecture Overview](ARCHITECTURE.md) - General system architecture
- [API Reference](API_REFERENCE.md) - Complete API documentation
