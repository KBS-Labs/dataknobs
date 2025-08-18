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

### Available Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `EQ` | Equal to | `Filter("status", Operator.EQ, "active")` |
| `NE` | Not equal to | `Filter("status", Operator.NE, "deleted")` |
| `GT` | Greater than | `Filter("age", Operator.GT, 18)` |
| `GTE` | Greater than or equal | `Filter("score", Operator.GTE, 90)` |
| `LT` | Less than | `Filter("price", Operator.LT, 100)` |
| `LTE` | Less than or equal | `Filter("quantity", Operator.LTE, 10)` |
| `IN` | In list | `Filter("color", Operator.IN, ["red", "blue"])` |
| `NOT_IN` | Not in list | `Filter("status", Operator.NOT_IN, ["deleted", "archived"])` |
| `CONTAINS` | Contains substring | `Filter("name", Operator.CONTAINS, "john")` |
| `BETWEEN` | Between range | `Filter("age", Operator.BETWEEN, (18, 65))` |
| `NOT_BETWEEN` | Outside range | `Filter("temp", Operator.NOT_BETWEEN, (20, 30))` |

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

### Nested Field Queries

Query nested fields using dot notation:

```python
# Query metadata fields
query = Query(filters=[
    Filter("metadata.type", Operator.EQ, "sensor_reading"),
    Filter("metadata.version", Operator.GTE, 2)
])

# Query nested JSON fields
query = Query(filters=[
    Filter("config.features.auth", Operator.EQ, True),
    Filter("address.city", Operator.EQ, "New York")
])
```

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
    sort=[SortSpec("timestamp", SortOrder.DESC)]
)

# Multi-field sorting
query = Query(
    filters=[Filter("status", Operator.EQ, "active")],
    sort=[
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
    limit=100
)

# Offset for pagination
page_size = 20
page = 3
query = Query(
    filters=[Filter("status", Operator.EQ, "active")],
    limit=page_size,
    offset=(page - 1) * page_size
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
        sort=[SortSpec(group_by, SortOrder.ASC)]
    )
```

## Using Queries with Backends

```python
from dataknobs_data import SyncMemoryDatabase

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