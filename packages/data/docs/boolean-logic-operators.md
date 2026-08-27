# Boolean Logic Operators Implementation

## Overview
Added comprehensive boolean logic operators (OR, AND, NOT) to the DataKnobs data package, enabling complex queries with grouped and nested conditions.

## Key Features

### 1. Simple OR Queries
```python
from dataknobs_data import Query, Filter, Operator

# Find records where city is "New York" OR "Los Angeles"
query = Query().or_(
    Filter("city", Operator.EQ, "New York"),
    Filter("city", Operator.EQ, "Los Angeles")
)
```

### 2. Combining AND and OR
```python
# Find active users in New York OR Los Angeles
# Creates: active=True AND (city="New York" OR city="Los Angeles")
query = Query().filter("active", Operator.EQ, True).or_(
    Filter("city", Operator.EQ, "New York"),
    Filter("city", Operator.EQ, "Los Angeles")
)
```

### 3. NOT Queries
```python
# Find active users NOT in New York
query = Query().filter("active", Operator.EQ, True).not_(
    Filter("city", Operator.EQ, "New York")
)
```

### 4. Complex Nested Conditions
```python
from dataknobs_data import QueryBuilder

# Find: (age > 30 AND city="New York") OR (age < 30 AND active=True)
builder = QueryBuilder()

# First condition group
group1 = QueryBuilder().where("age", Operator.GT, 30).where("city", Operator.EQ, "New York")

# Second condition group  
group2 = QueryBuilder().where("age", Operator.LT, 30).where("active", Operator.EQ, True)

# Combine with OR
query = builder.or_(group1, group2).build()
```

## Implementation Details

### Architecture
- **Shared implementation** in base `Database` classes for maximum code reuse
- **Backend-specific optimizations** can override for native boolean logic support
- **ComplexQuery** class for representing boolean logic structures
- **QueryBuilder** for fluent API construction of complex queries

### Classes Added
1. **ComplexQuery** - Represents queries with boolean logic
2. **QueryBuilder** - Fluent API for building complex queries
3. **LogicOperator** - Enum for AND, OR, NOT operations
4. **Condition** - Abstract base for query conditions
5. **FilterCondition** - Single filter wrapped as condition
6. **LogicCondition** - Logical combination of conditions

### How It Works
1. Simple `Query` objects can create `ComplexQuery` via `or_()` and `not_()` methods
2. `ComplexQuery` attempts to convert to simple query when possible for efficiency
3. When conversion isn't possible, in-memory filtering is performed
4. Individual backends can override `_search_with_complex_query()` for native support

## Usage Examples

### Example 1: Multi-field OR
```python
# Find products that are either expensive OR popular
query = Query().or_(
    Filter("price", Operator.GT, 100),
    Filter("rating", Operator.GTE, 4.5)
)
```

### Example 2: Exclusion with NOT
```python
# Find all products except those from specific brands
query = Query().not_(
    Filter("brand", Operator.IN, ["BrandA", "BrandB"])
)
```

### Example 3: Complex Business Logic
```python
# Premium customers: (spent > 1000 OR member_level="gold") AND active=True
builder = QueryBuilder()
builder.where("active", Operator.EQ, True)
builder.and_(
    QueryBuilder().or_(
        Filter("total_spent", Operator.GT, 1000),
        Filter("member_level", Operator.EQ, "gold")
    )
)
query = builder.build()
```

### Example 4: Combining with BETWEEN
```python
# Find users aged 25-35 OR in specific cities
query = Query().or_(
    Filter("age", Operator.BETWEEN, (25, 35)),
    Filter("city", Operator.IN, ["NYC", "LA", "Chicago"])
)
```

## Backend Support

### Memory Backend
- Full support via shared implementation
- Efficient in-memory filtering

### PostgreSQL Backend
- Uses shared implementation currently
- Future: Could optimize with native SQL boolean operators

### Elasticsearch Backend
- Uses shared implementation currently
- Future: Could optimize with native bool queries

### File/S3 Backends
- Full support via shared implementation
- Filters applied during record loading

## Performance Considerations

1. **Simple queries are preferred** - Use basic AND filters when possible
2. **Conversion optimization** - ComplexQuery converts to simple Query when all conditions are AND
3. **Backend-native support** - Future backends can implement native boolean logic
4. **Memory usage** - Complex queries may load all records for in-memory filtering

## Testing

Comprehensive test suite in `tests/test_boolean_logic.py`:
- 17 test cases covering all operators
- Sync and async database support
- Nested and grouped conditions
- Serialization/deserialization
- Edge cases and error conditions

## Migration Guide

### Before (Multiple Queries)
```python
# Had to run multiple queries and combine results manually
results1 = db.search(Query().filter("status", Operator.EQ, "active"))
results2 = db.search(Query().filter("priority", Operator.EQ, "high"))
combined = list(set(results1) | set(results2))
```

### After (Single Query)
```python
# Single query with OR logic
query = Query().or_(
    Filter("status", Operator.EQ, "active"),
    Filter("priority", Operator.EQ, "high")
)
results = db.search(query)
```

## Future Enhancements

1. **Native backend support** - Implement PostgreSQL and Elasticsearch native boolean queries
2. **Query optimization** - Analyze and optimize complex query patterns
3. **Query validation** - Validate complex queries before execution
4. **Visual query builder** - UI component for building complex queries
5. **Query caching** - Cache results of complex queries

## API Reference

### Query Methods
- `or_(*filters)` - Create ComplexQuery with OR logic
- `and_(*filters)` - Add filters with AND logic (convenience)
- `not_(filter)` - Create ComplexQuery with NOT logic

### QueryBuilder Methods
- `where(field, operator, value)` - Add filter condition
- `and_(*conditions)` - Add AND conditions
- `or_(*conditions)` - Add OR conditions
- `not_(condition)` - Add NOT condition
- `build()` - Build final ComplexQuery

### ComplexQuery Properties
- `condition` - Root condition tree
- `matches(record)` - Check if record matches
- `to_simple_query()` - Convert to Query if possible
- `to_dict()/from_dict()` - Serialization support