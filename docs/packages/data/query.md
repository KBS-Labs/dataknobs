# Query System

## Overview

The DataKnobs query system provides a powerful and flexible way to search, filter, and retrieve records from any backend. It supports simple filters, complex boolean logic, range queries, sorting, and pagination.

## Basic Queries

### Simple Filtering

```python
from dataknobs_data import Query, Filter, Operator

# Find records by exact match
query = Query(filters=[
    Filter("status", Operator.EQ, "active")
])

# Find records with multiple conditions (AND)
query = Query(filters=[
    Filter("type", Operator.EQ, "sensor"),
    Filter("location", Operator.EQ, "warehouse")
])

# Search with comparison operators
query = Query(filters=[
    Filter("temperature", Operator.GT, 25.0),
    Filter("humidity", Operator.LT, 60.0)
])
```

> **`id` is a reserved filter/sort field name.** `Filter("id", ...)` and
> `SortSpec("id", ...)` target the record's **storage key** on every backend, not a
> `data` field named `id`. A value stored under `data["id"]` is **shadowed** — the
> filter matches the storage key and silently returns no rows. To query a
> secondary identifier, name the field something other than `id`. See the
> [reserved query field name note](api-reference.md#querying-by-identifier-and-key-prefix).

### Available Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `EQ` | Equal to | `Filter("status", Operator.EQ, "active")` |
| `NEQ` | Not equal to | `Filter("status", Operator.NEQ, "deleted")` |
| `GT` | Greater than | `Filter("age", Operator.GT, 18)` |
| `GTE` | Greater than or equal | `Filter("score", Operator.GTE, 90)` |
| `LT` | Less than | `Filter("price", Operator.LT, 100)` |
| `LTE` | Less than or equal | `Filter("quantity", Operator.LTE, 10)` |
| `IN` | Equal to one of a collection | `Filter("color", Operator.IN, ["red", "blue"])` |
| `NOT_IN` | Has a value, equal to none of a collection | `Filter("status", Operator.NOT_IN, ["deleted", "archived"])` |
| `LIKE` | SQL-style pattern (`%`, `_`) | `Filter("name", Operator.LIKE, "%john%")` |
| `NOT_LIKE` | Does not match a pattern | `Filter("name", Operator.NOT_LIKE, "test%")` |
| `REGEX` | Regular-expression search | `Filter("code", Operator.REGEX, r"^A\d+")` |
| `STARTS_WITH` | Literal, case-sensitive prefix | `Filter("path", Operator.STARTS_WITH, "docs/")` |
| `EXISTS` | Field has a value | `Filter("email", Operator.EXISTS)` |
| `NOT_EXISTS` | Field has no value | `Filter("deleted_at", Operator.NOT_EXISTS)` |
| `BETWEEN` | Between range | `Filter("age", Operator.BETWEEN, (18, 65))` |
| `NOT_BETWEEN` | Outside range | `Filter("temp", Operator.NOT_BETWEEN, (20, 30))` |

`IN` and `NOT_IN` take a collection that is neither a string nor a mapping;
anything else raises `ValueError` when the `Filter` is built. Nothing is in an
empty collection, and a `None` member matches nothing. The full rules, and how
Elasticsearch differs from the other backends, are in the
[API reference](api-reference.md#query).

## Advanced Queries

### Boolean Logic (AND, OR, NOT)

```python
from dataknobs_data import Query, Filter, Operator

# OR query - match any condition
query = Query().or_(
    Filter("sensor_id", Operator.EQ, "sensor_001"),
    Filter("sensor_id", Operator.EQ, "sensor_002"),
    Filter("sensor_id", Operator.EQ, "sensor_003")
)

# Complex boolean logic
query = Query()\
    .filter("type", Operator.EQ, "reading")\
    .and_(
        Query().or_(
            Filter("temperature", Operator.GT, 30),
            Filter("humidity", Operator.GT, 80)
        )
    )

# NOT query - exclude matches
query = Query().not_(
    Filter("status", Operator.IN, ["deleted", "archived"])
)
```

`NOT` is the complement of the condition it wraps, so the query above also
returns records with no `status`, or a `null` one: they are not deleted or
archived. `Filter("status", Operator.NOT_IN, ["deleted", "archived"])` asks
for a status outside the list, and returns neither. Add
`Filter("status", Operator.EXISTS)` to leave them out of a `NOT`.

### Nested Field Queries (Dot-Notation)

Query nested fields using dot notation.  Dots in field names are **always**
interpreted as JSON path separators — this convention is consistent across
`Record.get_value()`, in-memory filtering, and all SQL backends (PostgreSQL,
SQLite, DuckDB).

```python
# Query metadata fields (routes to the "metadata" JSONB column in SQL)
query = Query(filters=[
    Filter("metadata.type", Operator.EQ, "sensor_reading"),
    Filter("metadata.version", Operator.GTE, 2)
])

# Query nested JSON fields within record data
query = Query(filters=[
    Filter("config.features.auth", Operator.EQ, True),
    Filter("address.city", Operator.EQ, "New York")
])

# Mix flat, nested data, and metadata fields
query = Query(filters=[
    Filter("status", Operator.EQ, "active"),
    Filter("config.timeout", Operator.GT, 30),
    Filter("metadata.tenant_id", Operator.EQ, "T-1")
])
```

#### How It Works Across Backends

| Backend | `metadata.version` | `config.timeout` |
|---------|---------------------|-------------------|
| Memory / File | `record.get_value("metadata.version")` | `record.get_value("config.timeout")` |
| PostgreSQL | `metadata->>'version'` | `data->'config'->>'timeout'` |
| SQLite | `json_extract(metadata, '$.version')` | `json_extract(data, '$.config.timeout')` |
| DuckDB | `json_extract_string(metadata, '$.version')` | `json_extract_string(data, '$.config.timeout')` |

Type casting is applied automatically for numeric, boolean, and datetime
comparisons on PostgreSQL and DuckDB.

!!! warning "Literal dots in JSON keys"

    Because dots are **always** path separators, JSON keys that literally
    contain a dot (e.g. `{"my.field": 1}`) **cannot** be queried through
    the filter interface.  This matches the behaviour of
    `Record.get_value()`, which uses the same convention.  If your data
    uses dots in key names, flatten or rename them before storage.

### Tables With Their Own Columns (Native Layout)

The SQL query builder reads a table through a **column layout**. The default
is the table this package creates (`id`, and the record in the `data` and
`metadata` JSON columns), and everything above describes it.
`NativeColumnLayout` reads a table somebody else owns, with ordinary typed
columns:

```python
from dataknobs_data import NATIVE_FIELD_KEYS, Filter, Operator, Query
from dataknobs_data.backends.column_layout import NativeColumnLayout
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.schema import DatabaseSchema

schema = DatabaseSchema.from_dict({"fields": {
    "node_id": {"type": "string", "sql_type": "uuid"},
    "name": "string",
    "size": "integer",
    "status": "string",
}}, keys=NATIVE_FIELD_KEYS)
layout = NativeColumnLayout(
    schema,
    id_column="node_id",
    scope=[Filter("status", Operator.EQ, "live")],
)
builder = SQLQueryBuilder("nodes", dialect="postgres", layout=layout)
sql, params = builder.build_search_query(Query(filters=[Filter("size", Operator.GTE, 3.5)]))
# SELECT "node_id", "name", "size", "status" FROM "nodes"
#   WHERE "status" = $1 AND "size" >= CAST($2 AS numeric)
```

**No backend takes the native layout yet**: the PostgreSQL, SQLite and DuckDB
backends adopt it in later releases. What the builder guarantees:

- **Only declared columns.** A filter, sort or scope naming another column
  (or a dotted path) raises `ValidationError` before SQL is built, and a
  statement selects the declared columns rather than `*`. `id` is the
  `id_column`.
- **The key is the key column's text.** A record's storage id is that text,
  and `id` is compared with it, as `Filter.matches` compares the storage id,
  so a read finds a row by the key a search returned. A key column is a
  `string`, `text`, `uuid` or `integer` column; any other is refused, since
  engines write a float, boolean or time as different text. An integer key's
  own text (`"10"`, not `"010"` or `10`) is compared in the column's type, so
  its index serves a read.
- **The scope is in every read**: search, count, `build_where_clause`, read and
  exists. A complex query's condition is nested under it, so an `OR` cannot
  reach rows outside it. A scope filter comparing a value its column cannot
  hold is refused: it would match no row, or, negated (`!=`, `NOT IN`,
  `NOT BETWEEN`), exclude only the rows where the column is NULL.
- **Read-only.** Every statement that would write raises `OperationError`.
- **Every filter answers as `Filter.matches` answers** over the record the
  layout returns. Each column holds the kinds of value its declared type says
  (see [Field Types](field-types.md#sql-types-for-tables-with-their-own-columns)):

    | Declared | Compared with | Text operators (`LIKE`, `REGEX`, `STARTS_WITH`) |
    |---|---|---|
    | `string`, `text` | a string; a date or datetime, when the text names a time | yes |
    | `integer`, `float` | a number (never a boolean) | no: they match nothing |
    | `boolean` | a boolean (never a number) | no |
    | `datetime` | a naive datetime, a date (as its midnight), a string naming one | no |
    | `sql_type: timestamptz` | an aware datetime, a date (its midnight in UTC), a string naming one | no |
    | `sql_type: uuid` | a string or `UUID`, as its canonical lower-case text | yes, over its text |
    | `json`, `binary`, vectors | refused for every operator but `EXISTS` / `NOT_EXISTS` | — |

    A bound of another kind matches nothing and its negation every present
    value, and it is never sent to the driver, which might refuse it or,
    worse, convert it and answer wrongly. On PostgreSQL a numeric bound is
    sent as `bigint`, `double precision` or `numeric`, because asyncpg would
    otherwise send `3.5` to an `integer` column as `3`. An `integer` column
    is compared exactly, as Python compares an `int` with a `float`: a whole
    bound is sent as its `int` (`2.0**60` as `2**60`), and a fractional one
    as the value halfway between the integers it lies between (`3.25` as
    `3.5`, as `numeric` on PostgreSQL and `DECIMAL(38,1)` on DuckDB), since a
    `float` bound is compared by rounding the column past 2**53.

- **`NOT` over a native filter** matches a row whose column is `NULL`, as
  `NOT` does under the JSON layout and as `Filter.matches` answers.
- **Text sorts by code point** on every engine (`COLLATE "C"` on
  PostgreSQL), as the in-memory sort does, and a key sorts as its text.
  A time sorts by the time a filter reads it as: on SQLite, which keeps the
  text it was given, a zoned value sorts by its instant, not its wall clock.

!!! warning "Declare each column as the table holds it"

    The declared type is what every answer rests on, and nothing reads the
    table to check it. An integer column declared `string` matches no number
    you filter it with.

#### A Layout of Your Own

A table neither layout reads takes its own: subclass `ColumnLayout` and
render each method with the builder's public clause primitives, which answer
a filter as `Filter.matches` does. Here, a legacy table that holds every value
as text:

```python
from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.column_layout import ColumnLayout
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sql_types import TIME_READINGS


class TextTableLayout(ColumnLayout):
    def __init__(self, columns, key):
        self.columns, self.key = tuple(columns), key

    def _column(self, field):
        return f'"{self.key if field == "id" else field}"'

    def filter_clause(self, builder, spec, param_start):
        column = self._column(spec.field)
        if spec.operator == Operator.EXISTS:
            return f"{column} IS NOT NULL", []
        if spec.operator == Operator.NOT_EXISTS:
            return f"{column} IS NULL", []
        if spec.operator in builder.STRING_ONLY_OPERATORS:
            return builder.operator_clause(column, spec.operator, spec.value, param_start)

        def expr_for(reading):
            # Each value is a string; a time bound reads the text as a time.
            if reading == "string":
                return None, column
            if reading in TIME_READINGS:
                names_time, value = builder.time_reading(column, reading)
                return names_time, f"CASE WHEN {names_time} THEN {value} END"
            return None  # a number or a boolean relates to no value here

        return builder.typed_clause(
            spec.operator, spec.value, param_start, expr_for, f"{column} IS NOT NULL"
        )

    def sort_keys(self, builder, field):
        return [builder.code_point_order(self._column(field))]

    def select_list(self, builder):
        return ", ".join(f'"{c}"' for c in self.columns)

    def key_clause(self, builder, record_id, param_start):
        return f"{self._column('id')} = {builder.param_placeholder(param_start)}", [record_id]

    def record_from_row(self, row):
        return Record({c: row[c] for c in self.columns}, storage_id=row[self.key])


builder = SQLQueryBuilder("legacy", dialect="postgres",
                          layout=TextTableLayout(["k", "name"], key="k"))
sql, params = builder.build_search_query(Query(filters=[Filter("name", Operator.IN, ["a", 5])]))
# SELECT "k", "name" FROM "legacy" WHERE "name" = ANY($1)  -- params [['a']]
```

`typed_clause` renders the comparison, membership, range and negation from
what `expr_for` says a value is under each reading of a bound: `"string"`,
`"number"`, `"boolean"`, `NEVER` (a `None` or NaN bound), or one of
`TIME_READINGS`. Its kind test must be false, not `NULL`, for a value it
excludes, so `NOT` over the clause still matches that value. A layout reads
only: the builder's write statements are rendered for the JSON layout's
columns, so a layout other than `JsonbLayout` that sets `writable` is refused
when a builder is given it.

### Range Queries

Use BETWEEN for efficient range queries:

```python
from datetime import datetime, timedelta

# Time range query
start = datetime.now() - timedelta(days=7)
end = datetime.now()
query = Query(filters=[
    Filter("created_at", Operator.BETWEEN, (start, end))
])

# Numeric range
query = Query(filters=[
    Filter("price", Operator.BETWEEN, (10.0, 100.0))
])

# Find outliers (NOT_BETWEEN)
normal_range = (18.0, 25.0)
outliers_query = Query(filters=[
    Filter("temperature", Operator.NOT_BETWEEN, normal_range)
])
```

## Query Builder Pattern

Use the QueryBuilder for fluent query construction:

```python
from dataknobs_data import QueryBuilder, Operator

# Build complex queries step by step
builder = QueryBuilder()

# Add base conditions
builder.where("type", Operator.EQ, "sensor_reading")
builder.where("location", Operator.IN, ["warehouse", "factory"])

# Add time range
builder.where("timestamp", Operator.BETWEEN, (start_time, end_time))

# Add OR conditions
builder.or_(
    Filter("alert_level", Operator.EQ, "critical"),
    Filter("temperature", Operator.GT, 40)
)

# Build final query
query = builder.build()
```

## Sorting and Pagination

### Sorting Results

```python
from dataknobs_data import Query, SortSpec, SortOrder

# Sort by single field
query = Query(
    filters=[Filter("type", Operator.EQ, "reading")],
    sort_specs=[SortSpec("timestamp", SortOrder.DESC)]
)

# Multi-field sorting
query = Query(
    filters=[Filter("status", Operator.EQ, "active")],
    sort_specs=[
        SortSpec("priority", SortOrder.DESC),
        SortSpec("created_at", SortOrder.ASC)
    ]
)
```

### Pagination

```python
# Limit results
query = Query(
    filters=[Filter("type", Operator.EQ, "log")],
    limit_value=100
)

# Offset for pagination
page_size = 20
page = 3
query = Query(
    filters=[Filter("status", Operator.EQ, "active")],
    limit_value=page_size,
    offset_value=(page - 1) * page_size
)
```

## Complex Query Examples

### Multi-criteria Search

```python
def search_critical_sensors(
    min_battery: float = 20.0,
    locations: list = None,
    time_window: tuple = None
) -> Query:
    """Find sensors needing attention."""
    
    builder = QueryBuilder()
    
    # Base condition
    builder.where("type", Operator.EQ, "sensor")
    
    # Critical conditions (OR)
    critical = QueryBuilder()
    
    # Low battery
    critical.or_(Filter("battery", Operator.LT, min_battery))
    
    # High temperature
    critical.or_(Filter("temperature", Operator.GT, 35))
    
    # Offline sensors
    if time_window:
        critical.or_(
            Filter("last_seen", Operator.NOT_BETWEEN, time_window)
        )
    
    builder.and_(critical)
    
    # Location filter
    if locations:
        builder.where("location", Operator.IN, locations)
    
    return builder.build()
```

### Aggregation-like Queries

```python
def get_statistics_query(
    metric: str,
    group_by: str,
    time_range: tuple
) -> Query:
    """Build query for statistics."""
    
    return Query(
        filters=[
            Filter("metric_name", Operator.EQ, metric),
            Filter("timestamp", Operator.BETWEEN, time_range)
        ],
        sort_specs=[SortSpec(group_by, SortOrder.ASC)]
    )
```

## Using Queries with Backends

```python
from dataknobs_data.backends import SyncMemoryDatabase

# Initialize database
db = SyncMemoryDatabase()

# Execute query
query = Query(filters=[
    Filter("status", Operator.EQ, "active"),
    Filter("score", Operator.GTE, 80)
])

results = db.search(query)

# Process results
for record in results:
    print(f"ID: {record.id}")
    print(f"Name: {record['name']}")  # Using new dict-like access
    print(f"Score: {record.score}")   # Using new attribute access
```

## Query Optimization Tips

1. **Use indexes** - Create indexes on frequently queried fields
2. **Limit results** - Always use limits for large datasets
3. **Use BETWEEN** - More efficient than combining GT and LT
4. **Filter early** - Apply most selective filters first
5. **Project fields** - Only retrieve needed fields when possible

## See Also

- [Record Model](record-model.md) - Understanding records and fields
- [Backends](backends.md) - Backend-specific query features
- [API Reference](api-reference.md#query) - Complete Query API