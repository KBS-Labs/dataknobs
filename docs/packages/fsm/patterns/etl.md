# ETL Pattern

The ETL (Extract, Transform, Load) pattern is implemented by the
**`DatabaseETL`** class. It moves records from a source
[`AsyncDatabase`](../../data/index.md) to a target one, applying per-record
transformations on the way, using an FSM to drive each record through the
pipeline.

> **Class name:** the implementation is `DatabaseETL` (there is no `ETLPattern`
> class). Import it from `dataknobs_fsm.patterns.etl`.

## How a run executes

`DatabaseETL.run()` owns **extraction**: it opens the source database, streams
it in batches (`batch_size`), and feeds each batch through the FSM with
`process_batch`. Every record then traverses the per-record FSM:

```
extract ──▶ validate ──▶ transform ──▶ [enrich] ──▶ load ──▶ complete
(start,                  (field_mappings           (DatabaseUpsert
 passthrough)             + transformations)        → target_db)
```

- **extract** — the FSM start state is a passthrough. Extraction itself is done
  by `run()` (a per-record "fetch all" would be nonsensical), so the record
  arrives already populated.
- **validate** — when `validation_schema` is set, this is a real gate: a record
  that satisfies the schema flows to `transform`, and a record that fails is
  diverted to a non-loading `rejected` terminal (counted in `rejected`, not
  `errors`). When `validation_schema` is unset, `validate` is an unconditional
  passthrough. See [Validation](#validation).
- **transform** — applies `field_mappings` (key renames) and then each
  `transformations` callable, in order. This is a real per-record step.
- **enrich** — the record routes through this state only when
  `enrichment_sources` is configured; it is currently a passthrough (see
  *Current limitations*).
- **load** — a `DatabaseUpsert` upserts the record into the `target_db`
  resource, keyed by `key_columns`.

When the batches are exhausted, `run()` closes the source and **closes the
FSM**, which flushes and closes the async `target_db` adapter so the upserted
rows are durably persisted. `run()` returns a metrics dictionary.

The per-record functions (`transform`, `load`) are wired through the FSM's
`custom_functions=` channel and referenced from each state's `functions` block —
the same idiom used by `examples/database_etl.py`.

## Basic usage

```python
import asyncio
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode

config = ETLConfig(
    # source_db / target_db are AsyncDatabase backend configs. The "type" key
    # selects the backend ("memory", "sqlite", "postgres", "file", ...).
    source_db={"type": "sqlite", "path": "source.db"},
    target_db={"type": "sqlite", "path": "target.db"},
    mode=ETLMode.FULL_REFRESH,
    target_table="customers",
    key_columns=["id"],
    field_mappings={"name": "full_name"},
    transformations=[
        lambda row: {**row, "processed": True},
    ],
)

etl = DatabaseETL(config)
metrics = asyncio.run(etl.run())
print(metrics)  # {'extracted': N, 'transformed': N, 'loaded': N, 'rejected': 0, 'errors': 0, 'skipped': 0}
```

`DatabaseETL` is built from an `ETLConfig` (its `CONFIG_CLS`). You can also
construct it from a dict via the inherited `DatabaseETL.from_config({...})`.
Because `source_db` and `target_db` are required, an all-default
`DatabaseETL()` is not valid.

## `ETLConfig`

`ETLConfig` is a frozen [`StructuredConfig`](../../common/structured-config.md)
subclass — it has `from_dict()` / `to_dict()` and is **immutable** (derive a
modified copy with `dataclasses.replace(...)`). The `source_db` / `target_db`
mappings and any `transformations` callables round-trip by identity, so
`to_dict()` on a transformation-bearing config is for in-process round-tripping,
not JSON serialization.

| Field | Type | Default | Role |
|---|---|---|---|
| `source_db` | `dict` | — (required) | Source `AsyncDatabase` config; `type` selects the backend |
| `target_db` | `dict` | — (required) | Target `AsyncDatabase` config; `type` selects the backend |
| `mode` | `ETLMode` | `FULL_REFRESH` | Extraction strategy (see below) |
| `batch_size` | `int` | `1000` | Records per extraction batch |
| `parallel_workers` | `int` | `4` | Max parallel records per batch |
| `error_threshold` | `float` | `0.05` | Max error rate before aborting |
| `checkpoint_interval` | `int` | `10000` | Checkpoint cadence (records) |
| `source_query` | `str \| None` | `"SELECT * FROM source_table"` | Extraction query; pass `None` to stream the whole source |
| `target_table` | `str` | `"target_table"` | Logical table name for the upsert |
| `key_columns` | `list[str] \| None` | `None` (→ `["id"]`) | Columns forming the upsert key |
| `field_mappings` | `dict[str, str] \| None` | `None` | `old_name → new_name` renames applied in the transform step |
| `transformations` | `list[Callable] \| None` | `None` | Per-record map-style callables applied in the transform step |
| `validation_schema` | `dict \| IValidationFunction \| Callable \| None` | `None` | Per-record validation gate; see [Validation](#validation) |
| `reject_counts_as_error` | `bool` | `False` | When `True`, validation rejections count toward `error_threshold` |
| `validation_resources` | `dict[str, dict] \| None` | `None` | Resources a resource-backed `validation_schema` predicate needs (e.g. a reference table); see [Resource-backed validation](#resource-backed-validation-validate-against-a-reference-table) |
| `enrichment_sources` | `list[dict] \| None` | `None` | Reserved — not yet applied (see *Current limitations*) |

### Transformations

Each entry in `transformations` is a **map-style** callable: it receives the
current record dict and returns the transformed dict. Callables may be sync or
async (an awaitable return is awaited). A non-dict return (including `None`)
raises a `TransformError` rather than silently corrupting the record.

```python
config = ETLConfig(
    source_db={"type": "memory"},
    target_db={"type": "memory"},
    target_table="orders",
    key_columns=["id"],
    field_mappings={"name": "product_name"},     # rename a non-key field
    transformations=[
        lambda row: {**row, "total": row["qty"] * row["price"]},
        lambda row: {**row, "currency": "USD"},
    ],
)
```

`field_mappings` are applied first (key renames), then the `transformations`
callables in order.

> **`key_columns` reference post-transform names.** The `load` step derives each
> row's storage id from `key_columns` *after* the transform stage runs. If a
> `field_mappings` entry renames a key column out from under load (e.g.
> `field_mappings={"id": "order_id"}` with `key_columns=["id"]`), every row would
> collapse onto a single id — so that combination is rejected at construction
> with `InvalidConfigurationError`. Either map a non-key field, or set
> `key_columns` to the renamed name (`key_columns=["order_id"]`). The same
> applies to a `transformations` callable that drops a key column (not statically
> checkable — author such callables to preserve the key fields).

## Validation

When `validation_schema` is set, the `validate` stage becomes a real gate: each
record is checked, valid records flow to `transform`, and rejected records are
diverted to a non-loading `rejected` terminal (never reaching `load`, so they
are never written to the target). The gate routes on the record alone and
**mutates nothing**.

`validation_schema` accepts three forms (a consumer picks the right tool, or
rolls their own) — all normalized by
`dataknobs_fsm.functions.library.validators.build_record_validator`:

```python
from dataknobs_fsm.functions.library.validators import RangeValidator

# 1. Friendly dict schema (the serializable, config-authored default).
#    Per-field constraints: required / type / min / max / pattern; a constraint
#    of literal True means "field must be present". Shared with file-processing
#    (see its validation vocabulary).
ETLConfig(..., validation_schema={"age": {"type": "int", "min": 18}})

# 2. A library IValidationFunction (RangeValidator / SchemaValidator /
#    PatternValidator / TypeValidator / ... — or your own subclass). Its
#    validate() raise contract is adapted to a boolean gate.
ETLConfig(..., validation_schema=RangeValidator({"age": {"min": 18}}))

# 3. A plain callable predicate `record -> bool` (or `(record, context) -> bool`).
ETLConfig(..., validation_schema=lambda row: row.get("age", 0) >= 18)
```

The dict form round-trips through the frozen config; the validator / callable
forms are in-process only (like `transformations`).

### Reject accounting

A rejected record is an expected **data-quality** drop, not a pipeline outage,
so it is counted in `rejected` — distinct from `errors` (a `transform` / `load`
failure). By default rejections do **not** trip `error_threshold`:

```python
# A run that rejects 30% of rows completes; rejected reflects the count.
metrics = asyncio.run(etl.run())   # {'loaded': 7, 'rejected': 3, 'errors': 0, ...}
```

Set `reject_counts_as_error=True` for a strict data-quality gate where too many
invalid rows should abort the run — then `rejected` folds into the
`error_threshold` numerator and the run raises `ETLError` once the combined rate
is exceeded.

### Resource-backed validation (validate against a reference table)

The three forms above read only the record. To validate against a **reference
resource** — e.g. "reject any row whose `id` is not in a `valid_ids` lookup
table" — pass an async predicate as `validation_schema` and declare the
resources it needs in `validation_resources`. Each `{name: {"type": ...,
"config": ...}}` entry is registered as an FSM resource and bound on the `valid`
arc, so the predicate resolves it from the injected `FunctionContext`:

```python
from dataknobs_data import Query

async def id_in_reference(record, ctx) -> bool:
    reference = ctx.require_resource("valid_ids")   # resolved by name
    rows = await reference.execute_query(Query())
    return record.get("id") in {r.get("id") for r in rows}

etl = DatabaseETL(ETLConfig(
    source_db={"type": "file", "path": "source.json"},
    target_db={"type": "file", "path": "target.json"},
    target_table="records",
    key_columns=["id"],
    validation_schema=id_in_reference,
    validation_resources={
        "valid_ids": {"type": "async_database",
                      "config": {"type": "file", "path": "valid_ids.json"}},
    },
))
```

Without `validation_resources` the gate condition's `context.resources` is
empty, so `require_resource(...)` raises — and (see below) that surfaces every
record as an **error**, not a reject. A condition that raises an *unexpected*
error (a missing/down reference resource, a failing lookup) is treated as a
genuine evaluation failure: the record is counted as an **error** (tripping
`error_threshold`), never silently diverted to `rejected`. To deliberately
reject a record the predicate returns `False` (or raises `ValidationError`).
This keeps an infrastructure outage in the gate from masquerading as a clean
data-quality drop. See the
[Resources guide](../guides/resources.md#resource-backed-arc-conditions) for the
arc resource-injection contract.

## ETL modes

```python
from dataknobs_fsm.patterns.etl import ETLMode

ETLMode.FULL_REFRESH   # "full" — stream the whole source
ETLMode.INCREMENTAL    # "incremental" — filter to changed rows (updated_at > checkpoint)
ETLMode.UPSERT         # "upsert"
ETLMode.APPEND         # "append"
```

The load step is always an upsert keyed by `key_columns`. `INCREMENTAL` changes
only the **extraction query** (it filters on `updated_at` against the last
checkpoint).

## Metrics

`run()` returns the metrics dictionary it accumulates:

| Key | Meaning |
|---|---|
| `extracted` | Records actually processed this run, recomputed as `loaded + rejected + errors` |
| `transformed` | Records that completed cleanly (transformed and loaded) |
| `loaded` | Records upserted into the target |
| `rejected` | Records diverted by validation (data-quality drops, not failures) |
| `errors` | Records whose `transform` or `load` step raised |
| `skipped` | Reserved (currently always `0`) |

A record only counts toward `transformed` + `loaded` when it reached `complete`
**and** the per-record execution reported success. A record diverted by the
[validation gate](#validation) reaches the `rejected` terminal and is counted in
`rejected`. A record whose `transform` or `load` step raised is reported as a
failure by the engine and counted as an `error` — not as `loaded` — even though
the FSM still traversed to a final state. This keeps `error_threshold` honest: a
target-write outage surfaces as errors, dirty source data surfaces as
rejections, and (by default) only the former aborts the run.

Metrics are reset at the start of every `run()` (they do not accumulate across
runs), and `run()` rebuilds the FSM each call, so a single `DatabaseETL`
instance is safely re-runnable.

## Factory functions

All factories return a `DatabaseETL`. `source`/`target` may be a config dict or a
connection string (`postgresql://`, `mongodb://`, `sqlite://` are recognized by
the built-in parser).

```python
from dataknobs_fsm.patterns.etl import (
    create_etl_pipeline,
    create_database_sync,
    create_data_migration,
    create_data_warehouse_load,
    ETLMode,
)

# General-purpose pipeline. **kwargs are forwarded to ETLConfig.
etl = create_etl_pipeline(
    source={"type": "postgres", "host": "localhost", "database": "src"},
    target={"type": "postgres", "host": "localhost", "database": "dst"},
    mode=ETLMode.INCREMENTAL,
    transformations=[lambda r: {**r, "ingested": True}],
)

# Incremental sync (INCREMENTAL mode, checkpoint_interval=1000).
sync = create_database_sync(source={...}, target={...}, sync_interval=300)

# Migration with field mappings (FULL_REFRESH, batch_size=5000, parallel_workers=8).
migration = create_data_migration(
    source={...}, target={...},
    field_mappings={"customer_id": "id", "customer_name": "name"},
    transformations=[...],
)

# One pipeline per source into a warehouse (APPEND, batch_size=10000).
pipelines = create_data_warehouse_load(
    sources=[{...}, {...}],
    warehouse={...},
    aggregations=[...],   # passed through as transformations
)
```

## Current limitations

- **`enrich`** is a passthrough. `enrichment_sources` is accepted but the
  per-record database/API lookup is not yet implemented.

This is an honest no-op today — it does not silently transform data
incorrectly. Wiring it requires settling its config contract (per-record lookup
semantics) and is tracked separately. (The `validate` stage is fully wired — see
[Validation](#validation).)

- **Failure routing.** A record whose `transform` or `load` step raises is
  counted as an `error` (and trips `error_threshold`), but it is **not** routed
  to a dedicated error state — there is no conditional error arc. Once a record
  fails a state transform, the engine skips its remaining/downstream transforms,
  so a record whose `transform` raised is **not** upserted at `load` (no stale
  or untransformed write reaches the target). The record still traverses to the
  final state for accounting; it is simply reported as a failure rather than
  loaded. What is not yet implemented is explicit on-failure *routing* — sending
  the failed record to a dead-letter sink or a distinct error state for
  inspection — which requires a conditional error-arc design and is tracked
  separately.

  If you build a custom FSM (rather than using `DatabaseETL`) and need a
  recovery / cleanup / dead-letter state whose transforms must run despite an
  upstream failure, mark that state `run_on_failure: true` — the engine's
  post-failure transform skip does not apply to it (see
  [Failure handling](../guides/configuration.md#failure-handling-run_on_failure)).
  It re-enables the state's transforms only; the record is still reported as a
  failure.

## Testing

Use real constructs — a file-backed `AsyncDatabase` is reopenable, so a test can
run the pipeline and then reopen the target to assert what was persisted (no
mocks):

```python
import pytest
from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


@pytest.mark.asyncio
async def test_etl_persists_transformed_rows(tmp_path):
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")

    # Seed the source.
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": src})
    await db.upsert("1", Record({"id": "1", "name": "Alice"}))
    await db.close()

    etl = DatabaseETL(ETLConfig(
        source_db={"type": "file", "path": src},
        target_db={"type": "file", "path": tgt},
        source_query=None,
        target_table="records",
        key_columns=["id"],
        mode=ETLMode.FULL_REFRESH,
        transformations=[lambda r: {**r, "tag": "X"}],
    ))
    metrics = await etl.run()
    assert metrics["loaded"] == 1

    # Reopen the target and assert the transformed row landed.
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": tgt})
    rows = [r.to_dict() async for r in db.stream_read(Query())]
    await db.close()
    assert rows[0]["tag"] == "X"
```

## See also

- [File Processing Pattern](file-processing.md) for file-based pipelines
- [Error Recovery Pattern](error-recovery.md) for retry/circuit-breaker building blocks
- [Examples](../examples/database-etl.md) for a worked FSM-based ETL example
