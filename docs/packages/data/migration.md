# Data Migration Utilities

The DataKnobs data package provides migration utilities for moving records
between backends, evolving record schemas, and transforming data in flight.

## Overview

The migration utilities enable:

- **Backend-to-backend migration**: Move records between any supported backends (Memory, File, SQLite, DuckDB, PostgreSQL, Elasticsearch, S3)
- **Schema evolution**: Apply reversible operations that add, remove, rename, or transform fields
- **Data transformation**: Reshape records during migration with a fluent transformer
- **Progress tracking**: Monitor progress and per-record outcomes
- **Conflict handling**: Choose how a re-run resolves ids already present in the target

The public API lives in `dataknobs_data.migration`:

| Symbol | Role |
|--------|------|
| `Migrator` | Stateless orchestrator with `migrate`, `migrate_stream`, `migrate_parallel`, `migrate_async`, `validate_migration` |
| `Migration` | An ordered, reversible set of `Operation`s between two versions |
| `AddField`, `RemoveField`, `RenameField`, `TransformField`, `CompositeOperation` | Reversible field `Operation`s |
| `Transformer` | Fluent record reshaper (`map`, `rename`, `exclude`, `add`) |
| `MigrationProgress` | Progress and outcome statistics |
| `ConflictPolicy` | Conflict resolution policy (`insert` / `upsert` / `skip`) |

## Migrator

`Migrator` is stateless — the source and target databases are passed to each
method, so one instance can drive many migrations.

```python
from dataknobs_data import database_factory
from dataknobs_data.migration import Migrator

source = database_factory.create(backend="sqlite", path="app.db")
source.connect()
target = database_factory.create(backend="duckdb", path="analytics.duckdb")
target.connect()

migrator = Migrator()
progress = migrator.migrate(source, target, batch_size=1000)

print(f"Succeeded: {progress.succeeded}")
print(f"Failed:    {progress.failed}")
print(f"Skipped:   {progress.skipped}")
print(f"Duration:  {progress.duration:.2f}s")
```

### Migration methods

| Method | Sync/Async | Notes |
|--------|-----------|-------|
| `migrate(source, target, transform=None, query=None, batch_size=1000, on_progress=None, on_error=None, on_conflict="insert")` | sync | Loads the source result set, writes in batches. |
| `migrate_stream(source, target, transform=None, query=None, config=None, on_progress=None)` | sync | Streams source → target without materializing the full dataset. Reads the conflict policy from `config.on_conflict`. |
| `migrate_parallel(source, target, transform=None, partitions=4, partition_field="partition_id", on_progress=None, on_conflict="insert")` | sync | Runs one streaming migration per partition value in parallel threads. |
| `migrate_async(source, target, transform=None, query=None, config=None, on_progress=None)` | async | Async streaming migration; reads the conflict policy from `config.on_conflict`. |

```python
from dataknobs_data import StreamConfig

# Streaming (memory-efficient) migration
progress = migrator.migrate_stream(
    source, target, config=StreamConfig(batch_size=5000)
)

# Parallel migration across partition values 0..3 of "partition_id"
progress = migrator.migrate_parallel(source, target, partitions=4)

# Async streaming migration
import asyncio

async def run():
    return await migrator.migrate_async(
        async_source, async_target, config=StreamConfig(batch_size=5000)
    )

progress = asyncio.run(run())
```

## Conflict Policy (Idempotent Re-runs)

`create()` is an atomic insert, so migrating a source into a target that
already holds one of the source ids records those ids as **failures** by
default. The conflict policy lets a re-run overwrite or skip existing rows
instead. It is threaded through all four migrate methods and defaults to strict
insert, so existing calls are unchanged.

| Policy | Behavior on a colliding id |
|--------|----------------------------|
| `"insert"` (default) | Fail closed — the id is recorded as a failure. Correct for a virgin target. |
| `"upsert"` | Overwrite the existing row so the target matches the source. Idempotent by definition. |
| `"skip"` | Leave the existing row untouched and count the id as skipped (`progress.skipped`). Idempotent top-up. |

```python
from dataknobs_data import ConflictPolicy, StreamConfig
from dataknobs_data.migration import Migrator

migrator = Migrator()

# Batched migrate / migrate_parallel take the policy directly.
progress = migrator.migrate(source, target, on_conflict="upsert")
progress = migrator.migrate_parallel(source, target, on_conflict="upsert")

# Skip ids already present, migrate the rest; safe to re-run.
progress = migrator.migrate(source, target, on_conflict=ConflictPolicy.SKIP)
print(f"Skipped {progress.skipped} already-present records")

# Streaming / async methods read the policy from StreamConfig.
progress = migrator.migrate_stream(
    source, target, config=StreamConfig(on_conflict="skip")
)
```

The `"insert"` fast-path uses the backend's native `create_batch` (or an atomic
`_write_batch` on PostgreSQL) with a per-record `create()` fallback; the
non-transactional backends (S3, Elasticsearch-sync, async-Elasticsearch) write
INSERT per-record via `create()`. `"upsert"` uses the native `upsert_batch` bulk
verb (with a
per-record `upsert` fallback); `"skip"` writes one record at a time (a
whole-batch verb cannot skip individual duplicates while inserting the rest). An
unknown policy value is rejected when the `StreamConfig` is constructed, rather
than silently falling back to insert.

Streaming `"insert"` fails closed on a colliding id — recording it as a failure
and preserving the source id — across **every** backend (streaming and batched
`migrate()` alike). For an idempotent re-run into a populated target use
`"upsert"` or `"skip"`.

## Progress Tracking

Every migrate method returns a `MigrationProgress`. Pass `on_progress` to
observe it during a long run.

```python
def show(progress):
    print(f"{progress.percent:.1f}% — "
          f"{progress.succeeded} ok, {progress.failed} failed, "
          f"{progress.skipped} skipped")

progress = migrator.migrate(source, target, on_progress=show)

# Fields and derived properties:
progress.total          # source count (when available)
progress.processed      # records seen
progress.succeeded      # written successfully
progress.failed         # write failures
progress.skipped        # filtered out or skipped by policy
progress.errors         # list of {error, record_id, timestamp, ...}
progress.duration       # seconds (property)
progress.percent        # processed / total * 100 (property)
progress.success_rate   # succeeded / processed * 100 (property)
progress.has_errors     # bool (property)
print(progress.get_summary())
```

## Transforming Records

Pass a `Transformer` (fluent reshaper) or a `Migration` (versioned operation
set) as the `transform=` argument. A `Transformer` that returns `None` for a
record filters it out — the record is counted as skipped.

```python
from dataknobs_data.migration import Transformer

transformer = (
    Transformer()
    .rename("full_name", "name")            # rename a field
    .map("email", transform=str.lower)      # transform a value in place
    .add("migrated", True)                  # add a constant field
    .add("initial", lambda r: r.get_value("name")[0])  # computed field
    .exclude("temporary", "scratch")        # drop fields
)

progress = migrator.migrate(source, target, transform=transformer)
```

`Transformer` methods (all chainable):

| Method | Effect |
|--------|--------|
| `map(source, target=None, transform=None)` | Copy/rename a field, optionally transforming its value |
| `rename(old_name, new_name)` | Rename a field |
| `exclude(*fields)` | Drop fields |
| `add(field_name, value, field_type=None)` | Add a field from a constant or a `Callable[[Record], Any]` |
| `add_rule(rule)` | Append a custom `TransformRule` |

## Schema Evolution with Migration

A `Migration` bundles reversible `Operation`s between two versions. Apply it to
records directly, or pass it as `transform=` to migrate a whole backend.

```python
from datetime import datetime
from dataknobs_data import Record
from dataknobs_data.migration import (
    Migration, AddField, RemoveField, RenameField, TransformField, CompositeOperation,
)

migration = Migration(
    from_version="1.0.0",
    to_version="2.0.0",
    description="Add status, drop legacy field, rename user_name",
)
migration.add(AddField("status", default_value="active"))
migration.add(RemoveField("legacy_field"))
migration.add(RenameField("user_name", "username"))
migration.add(TransformField("created_at", transform_fn=lambda v: v or datetime.now()))

# Apply to a single record (returns the migrated Record)
record = Record({"user_name": "alice", "legacy_field": "x"}, id="u1")
migrated = migration.apply(record)

# Reverse the migration (operations applied backwards)
original = migration.apply(migrated, reverse=True)

# Or migrate an entire backend through the migration
progress = migrator.migrate(source, target, transform=migration)
```

Available operations (each is reversible):

| Operation | Constructor |
|-----------|-------------|
| `AddField` | `AddField(field_name, default_value=None, field_type=None)` |
| `RemoveField` | `RemoveField(field_name, store_removed=False)` |
| `RenameField` | `RenameField(old_name, new_name)` |
| `TransformField` | `TransformField(field_name, transform_fn, reverse_fn=None)` |
| `CompositeOperation` | `CompositeOperation([op, ...])` |

## Filtering and Incremental Migration

Restrict the source with a `Query`. Combined with `skip`, this makes a resumable
incremental migration.

```python
from datetime import datetime, timedelta
from dataknobs_data import Query, Operator

# Only records updated in the last day, skipping any already in the target
recent = Query().filter("updated_at", Operator.GTE, datetime.now() - timedelta(days=1))
progress = migrator.migrate(source, target, query=recent, on_conflict="skip")
```

## Validating a Migration

`validate_migration` compares source and target and returns
`(is_valid, issues)`.

```python
is_valid, issues = migrator.validate_migration(source, target)
if not is_valid:
    for issue in issues:
        print(issue)
```

The source-vs-target count check is meaningful only for a from-empty `insert`
migration; under `upsert`/`skip` the counts can legitimately differ. The per-id
sample check (each source id present in the target) is valid for every policy.
Pass `sample_size=N` to check only the first `N` source ids.

## Error Handling

Pass an `on_error(exception, record) -> bool` callback to `migrate`. Return
`True` to record the failure and continue, `False` (or omit the handler) to stop
the run and re-raise.

```python
def keep_going(exc, record):
    logger.warning("Skipping %s: %s", record.id, exc)
    return True  # continue past the failure

progress = migrator.migrate(source, target, on_error=keep_going)
print(f"{progress.failed} records failed; see progress.errors for detail")
```

Migration-related failures raise `dataknobs_data.exceptions.MigrationError`
(a subclass of `OperationError`).

## Best Practices

1. **Test with a subset first** — migrate a `Query`-limited slice before the full run.
2. **Choose the right method** — `migrate` for bounded sets; `migrate_stream`/`migrate_async` for large datasets; `migrate_parallel` when the source partitions cleanly.
3. **Pick a conflict policy for re-runs** — `upsert` to make the target match the source, `skip` for idempotent top-ups; the default `insert` is correct only for a virgin target.
4. **Watch progress** — pass `on_progress` for long-running migrations.
5. **Handle errors explicitly** — supply `on_error` so one bad record does not abort the whole run.
6. **Keep migrations reversible** — provide `reverse_fn` on `TransformField` so a `Migration` can be rolled back with `apply(..., reverse=True)`.

## See Also

- [API Reference](api-reference.md) — `StreamConfig`, `StreamResult`, `ConflictPolicy`, and the operation/transformer types
- [Schema Validation](validation.md) — data validation and schema management
- [Pandas Integration](pandas-integration.md) — bulk DataFrame operations
- [Backends Overview](backends.md) — supported database backends
- [Migration Tutorial](tutorials/migration-tutorial.md) — step-by-step walkthrough
