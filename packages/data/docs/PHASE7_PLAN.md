# Phase 7: Pandas Integration - Implementation Plan

## Overview
Implement seamless integration between DataKnobs Records/Fields and Pandas DataFrames for efficient data analysis and manipulation.

## Key Components

### 1. Pandas Converter Module (`src/dataknobs_data/pandas/converter.py`)
- **Purpose**: Core conversion utilities between Records and DataFrames
- **Classes**:
  - `DataFrameConverter`: Main converter class
  - `TypeMapper`: Handles type mapping between Field types and pandas dtypes
  
**Key Methods**:
```python
class DataFrameConverter:
    def records_to_dataframe(records: List[Record], include_metadata: bool = False) -> pd.DataFrame
    def dataframe_to_records(df: pd.DataFrame, preserve_index: bool = False) -> List[Record]
    def record_to_series(record: Record) -> pd.Series
    def series_to_record(series: pd.Series, record_id: Optional[str] = None) -> Record
```

### 2. Type Mapping (`src/dataknobs_data/pandas/type_mapper.py`)
- **Purpose**: Accurate type conversion between DataKnobs and pandas
- **Mappings**:
  - FieldType.STRING → pd.StringDtype()
  - FieldType.INTEGER → pd.Int64Dtype() (nullable)
  - FieldType.FLOAT → pd.Float64Dtype()
  - FieldType.BOOLEAN → pd.BooleanDtype()
  - FieldType.DATETIME → pd.DatetimeTZDtype()
  - FieldType.JSON → object (with custom serialization)
  - FieldType.BINARY → object (bytes)
  - FieldType.TEXT → pd.StringDtype()

### 3. Batch Operations (`src/dataknobs_data/pandas/batch_ops.py`)
- **Purpose**: Efficient batch operations using DataFrame capabilities
- **Features**:
  - Bulk insert from DataFrame
  - Query results as DataFrame
  - Aggregations and transformations
  - Chunked processing for large datasets

### 4. Metadata Preservation (`src/dataknobs_data/pandas/metadata.py`)
- **Purpose**: Preserve Record metadata during conversions
- **Strategies**:
  - Store metadata in DataFrame.attrs
  - Optional metadata columns
  - Round-trip preservation

## Implementation Steps

### Step 1: Create Pandas Package Structure
```
src/dataknobs_data/pandas/
├── __init__.py
├── converter.py
├── type_mapper.py
├── batch_ops.py
├── metadata.py
└── utils.py
```

### Step 2: Implement Core Converter
1. Records to DataFrame conversion
2. DataFrame to Records conversion
3. Handle nested fields (JSON type)
4. Preserve field metadata

### Step 3: Type Mapping System
1. Create bidirectional type mappings
2. Handle nullable types properly
3. Custom converters for complex types
4. Type inference from data

### Step 4: Batch Operations
1. Implement bulk_insert_dataframe()
2. Add query_as_dataframe() to Database classes
3. Create DataFrame-based transformations
4. Add chunking for memory efficiency

### Step 5: Testing
1. Unit tests for all converters
2. Round-trip conversion tests
3. Large dataset performance tests
4. Type preservation tests
5. Metadata preservation tests

## Design Decisions

### 1. Nullable Types
- Use pandas nullable dtypes (Int64, Float64, etc.) to preserve None values
- Maintain distinction between missing and zero values

### 2. JSON Fields
- Store as Python objects in DataFrame
- Provide utility functions for JSON operations
- Consider using pd.json_normalize for nested data

### 3. Index Handling
- Record IDs become DataFrame index by default
- Option to preserve or reset index during conversion
- Support for multi-index from composite keys

### 4. Memory Efficiency
- Implement lazy loading for large datasets
- Use chunking for batch operations
- Optimize dtype selection for memory usage

### 5. Compatibility
- Support pandas 2.0+ features
- Graceful degradation for older versions
- Clear documentation of version requirements

## API Examples

```python
from dataknobs_data.pandas import DataFrameConverter
from dataknobs_data.backends.memory import MemoryDatabase

# Initialize
db = MemoryDatabase()
converter = DataFrameConverter()

# Records to DataFrame
records = db.search(Query())
df = converter.records_to_dataframe(records)

# DataFrame to Records
new_records = converter.dataframe_to_records(df)
for record in new_records:
    db.create(record)

# Batch operations
db.bulk_insert_dataframe(df)
results_df = db.query_as_dataframe(Query().filter("age", ">", 18))

# With type preservation
df = converter.records_to_dataframe(
    records,
    include_metadata=True,
    preserve_types=True
)
```

## Performance Targets
- Conversion of 100k records < 1 second
- Memory overhead < 20% compared to raw DataFrame
- Type conversion accuracy: 100%
- Metadata preservation: 100%

## Testing Strategy
1. **Unit Tests**: Each converter function
2. **Integration Tests**: Full round-trip conversions
3. **Performance Tests**: Large dataset handling
4. **Edge Cases**: Empty data, missing values, mixed types
5. **Compatibility Tests**: Different pandas versions

## Documentation Requirements
1. API reference for all public methods
2. Conversion guide with examples
3. Performance best practices
4. Type mapping reference
5. Migration guide from raw pandas

## Success Criteria
- ✅ All conversion functions implemented
- ✅ 100% type mapping coverage
- ✅ Batch operations functional
- ✅ Metadata preserved in round-trips
- ✅ Performance targets met
- ✅ >95% test coverage
- ✅ Documentation complete