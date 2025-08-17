# Data Package API Improvements

Based on implementing the Sensor Dashboard example, here are observations about the data package API and potential improvements:

## 1. Nested Field Queries Don't Work ❌ CRITICAL

**Issue**: The Query/Filter API appears to support nested field paths like `"metadata.type"` but the memory database implementation doesn't actually handle them. The `record.get_value()` method only looks for direct field names, not nested paths.

**Impact**: Users cannot query by metadata fields or nested structures, forcing inefficient workarounds like fetching all records and filtering in memory.

**Current Workaround**: Store queryable data as top-level fields in the Record.

**Proposed Solutions**:
1. Enhance `Record.get_value()` to support dot-notation paths
2. Add a separate `get_nested_value()` method  
3. Document this limitation clearly in the API docs

## 2. Missing Range Operators (Generic)

**Issue**: No built-in operators for range comparisons across different data types:
- **Temporal ranges**: No BETWEEN, DATE_GT, DATE_LT for timestamps/dates
- **Numeric ranges**: No efficient way to query "price between 100 and 500"
- **String ranges**: No lexicographic range queries (e.g., names starting with A-M)
- **Version ranges**: No semantic version comparisons

**Impact**: 
- Time-series data requires custom filtering after fetching all records
- Numeric range queries are inefficient
- Cannot leverage database-native range optimizations

**Proposed Solution**: 
- Add generic BETWEEN operator that works with comparable types
- Add type-aware comparison operators (GT_DATE, LT_DATE, GT_NUM, etc.)
- Or implement type inference in the operator to handle ranges appropriately

## 3. Limited Boolean Logic in Queries

**Issue**: Query filters only support AND logic (all filters must match). Missing:
- **OR operator**: Cannot express "sensor_1 OR sensor_2"
- **NOT operator**: Cannot easily express negation beyond NEQ
- **Grouped conditions**: Cannot express "(A AND B) OR (C AND D)"

**Impact**: 
- Forces multiple queries and client-side merging
- Cannot express common query patterns efficiently
- Leads to over-fetching and filtering in memory

**Proposed Solution**: 
- Add OR, NOT operators
- Support filter groups/nested conditions
- Consider a query builder pattern for complex logic

## 4. Inconsistent Batch Processing APIs

**Issue**: Multiple batch-related configurations with unclear relationships:
- `BatchConfig` in pandas module with different parameters than expected
- `StreamConfig` for streaming operations  
- Sync vs Async have different interfaces (list vs async iterable)
- No clear documentation on when to use which

**Impact**: Developers struggle to understand:
- How to configure batch operations with error handling
- Difference between batch_size, chunk_size, etc.
- How to handle partial failures consistently

**Observations**:
- `BatchConfig` has `error_handling: "raise"/"skip"/"log"` but not `continue_on_error` boolean
- `StreamConfig` has `on_error` callback but no `batch_config` parameter
- `StreamResult` lacks `total_batches` property (only has `total_processed`)
- Sync streaming accepts lists, async requires async iterables

**Proposed Solution**: 
- Unify batch configuration across the package
- Create adapter methods to handle list→async iterable conversion
- Clear documentation of each config's purpose
- Consistent parameter naming and error handling patterns

## 5. Field Access Ergonomics

**Issue**: Multiple inconsistencies in field access:
- Records store fields as Field objects with metadata
- Accessing values requires `record.get_value()` or `record.fields[name].value`
- No simple dot notation (`record.temperature`) or dict-like access (`record['temperature']`)
- Projection returns new Records, not simple dicts

**Impact**: 
- Verbose code for simple field access
- Confusion between Field objects and values
- Harder to integrate with existing code expecting dict-like objects

**Proposed Solution**: 
- Add `__getitem__` for dict-like access to values
- Consider `__getattr__` for dot notation (with careful design)
- Provide `to_dict()` method for simple value extraction
- Keep Field objects accessible for when metadata is needed

## 6. Missing Convenience Methods

**Issue**: Common operations require verbose workarounds:
- No `database.list()` or `database.all()` - must use empty Query
- No `database.count()` - must fetch all and count
- No `database.exists(id)` - must try read and check None
- No bulk delete by query

**Impact**: 
- Unintuitive API for common operations
- Inefficient patterns (fetch all to count)
- More boilerplate code

**Proposed Solution**: 
- Add convenience methods for common operations
- Ensure they're optimized at the backend level
- Provide both sync and async versions

## 7. Type System Improvements

**Issue**: Type handling could be more robust:
- FieldType enum doesn't cover all common types (e.g., DATETIME separate from TIMESTAMP)
- No support for array/list types
- No support for nested object types
- Type coercion rules are unclear

**Impact**:
- Cannot model complex data structures naturally
- Type mismatches between backends
- Confusion about how types are handled

**Proposed Solution**:
- Expand FieldType enum with more specific types
- Add support for collection types
- Document type coercion rules clearly
- Consider a type registry for custom types

## 8. Query Result Enhancements

**Issue**: Query results could provide more metadata:
- No total count when using limit/offset (for pagination)
- No query execution statistics
- No way to get distinct values for a field
- Sorting doesn't work reliably across backends

**Impact**:
- Cannot implement proper pagination UIs
- No visibility into query performance
- Must fetch all records for distinct values
- Must sort in memory after retrieval

**Proposed Solution**:
- Add `count()` method to Query for total results
- Return query stats in result metadata
- Add DISTINCT operator or `distinct()` method
- Fix sorting implementation across backends

## Positive Findings ✅

- Database abstraction allows backend switching successfully
- Error handling patterns in batch operations work well
- Connection pooling integration is clean
- Async/sync parallel APIs are well-designed (despite differences)
- Record metadata concept is powerful for filtering

## Recommendations (Prioritized)

1. **Priority 1**: Fix nested field queries - this is a fundamental feature
2. **Priority 2**: Add generic range operators for all comparable types
3. **Priority 3**: Implement boolean logic operators (OR, NOT, groups)
4. **Priority 4**: Unify and document batch processing APIs
5. **Priority 5**: Improve field access ergonomics
6. **Priority 6**: Add convenience methods for common operations
7. **Nice to Have**: Type system expansion, query result enhancements