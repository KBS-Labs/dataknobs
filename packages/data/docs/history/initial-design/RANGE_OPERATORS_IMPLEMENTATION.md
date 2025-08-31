# Range Operators Implementation Summary

## Overview
Implemented comprehensive range operators (BETWEEN and NOT_BETWEEN) for the DataKnobs data package with type-aware comparisons and backend-specific optimizations.

## Key Features Added

### 1. New Operators
- `Operator.BETWEEN` - Inclusive range check (value >= lower AND value <= upper)
- `Operator.NOT_BETWEEN` - Exclusive range check (value < lower OR value > upper)

### 2. Type-Aware Comparisons
The implementation handles multiple data types intelligently:

#### Numeric Ranges
```python
Filter("price", Operator.BETWEEN, (100, 500))  # Works with int and float
Filter("temperature", Operator.BETWEEN, (20.5, 30.0))
```

#### Temporal Ranges
```python
Filter("timestamp", Operator.BETWEEN, (start_date, end_date))  # datetime objects
Filter("timestamp", Operator.BETWEEN, ("2025-01-01", "2025-12-31"))  # ISO strings
```

#### String Ranges (Lexicographic)
```python
Filter("name", Operator.BETWEEN, ("Alice", "David"))  # Alphabetical range
Filter("code", Operator.BETWEEN, ("A000", "B999"))  # Code ranges
```

### 3. Backend Optimizations

#### PostgreSQL
- Uses native SQL `BETWEEN` operator
- Type-specific casting for optimal performance:
  - `::numeric` for numbers
  - `::timestamp` for dates
  - Direct comparison for strings

```sql
-- Generated SQL example
(data->>'price')::numeric BETWEEN %(param_0_lower)s AND %(param_0_upper)s
```

#### Elasticsearch
- Uses native Elasticsearch range query
- Single range query with `gte` and `lte` bounds
- Efficient index utilization

```json
{
  "range": {
    "data.price": {
      "gte": 100,
      "lte": 500
    }
  }
}
```

#### Memory Backend
- Filter.matches() method with type-aware comparisons
- Handles datetime string parsing automatically
- Mixed type comparisons (int vs float) work correctly

### 4. Edge Cases Handled
- Invalid range formats (not a tuple/list) return no matches
- NULL/None values are excluded from range matches
- Inclusive boundaries (both bounds are included)
- Empty ranges handled gracefully

## Usage Examples

### Simple Range Query
```python
from dataknobs_data import Query, Filter, Operator

# Numeric range
query = Query(filters=[
    Filter("temperature", Operator.BETWEEN, (20, 30))
])

# Date range
from datetime import datetime, timedelta
today = datetime.now()
last_week = today - timedelta(days=7)
query = Query(filters=[
    Filter("created_at", Operator.BETWEEN, (last_week, today))
])
```

### Combined with Other Filters
```python
# Find sensors with readings in a specific range and location
query = Query(filters=[
    Filter("temperature", Operator.BETWEEN, (22, 28)),
    Filter("metadata.location", Operator.EQ, "room_a"),
    Filter("timestamp", Operator.BETWEEN, (start_time, end_time))
])
```

### Using NOT_BETWEEN
```python
# Find outliers outside normal range
query = Query(filters=[
    Filter("value", Operator.NOT_BETWEEN, (10, 90))  # < 10 or > 90
])
```

## Testing
Comprehensive test suite added in `tests/test_range_operators.py`:
- 11 test cases covering all data types
- Edge cases and error conditions
- Backend-specific optimizations verified
- Integration with sensor dashboard example

## Performance Considerations
1. **Backend-native queries** are much faster than in-memory filtering
2. **Type casting** is handled automatically for datetime strings
3. **Index usage** in databases for range queries is optimized
4. **Nested field support** works with BETWEEN (e.g., "metrics.cpu")

## Migration Guide
For existing code using GT/LT combinations:

### Before
```python
# Old way - two separate filters
query = Query(filters=[
    Filter("price", Operator.GTE, 100),
    Filter("price", Operator.LTE, 500)
])
```

### After  
```python
# New way - single BETWEEN filter
query = Query(filters=[
    Filter("price", Operator.BETWEEN, (100, 500))
])
```

## Future Enhancements
Potential future improvements:
1. **Exclusive bounds option** - BETWEEN_EXCLUSIVE for (lower, upper) exclusive
2. **Open-ended ranges** - Support None for unbounded ranges
3. **Multi-range support** - Multiple disjoint ranges in single filter
4. **Performance hints** - Backend-specific query optimization hints