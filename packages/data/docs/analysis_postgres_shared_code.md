# PostgreSQL Backend Code Sharing Analysis

## Current Implementation Overview

### Class Hierarchy
- **SyncPostgresDatabase**: SyncDatabase, ConfigurableBase, VectorOperationsMixin, SQLRecordSerializer
- **AsyncPostgresDatabase**: AsyncDatabase, ConfigurableBase, VectorCapable

## Duplicated Code Identified

### 1. Configuration and Initialization
Both classes have nearly identical:
- `__init__()` - Same config handling logic
- `from_config()` - Identical implementation
- Table/schema name extraction logic

### 2. Record Serialization
- `_record_to_row()` - Both now use SQLRecordSerializer but defined separately
- `_row_to_record()` - Both now use SQLRecordSerializer but defined separately

### 3. Vector Support Detection
- `_detect_vector_support()` - Similar logic, one sync, one async
- Vector dimension tracking (`_vector_dimensions` dict)

### 4. Table Management
- `_ensure_table()` - Same SQL, different execution (sync vs async)
- Table creation SQL is duplicated

### 5. CRUD Operations Pattern
All CRUD methods follow the same pattern:
- Check connection
- Build query
- Execute
- Convert result

### 6. Batch Operations
- Similar batch processing logic
- Query building is shared via SQLQueryBuilder
- Result processing is duplicated

## Already Shared Code

### Good Examples:
1. **SQLQueryBuilder** - Query construction logic
2. **SQLRecordSerializer** - Record/JSON conversion
3. **postgres_vector.py** - Vector utilities

## Recommended Refactoring

### 1. Create PostgresBaseConfig Mixin
```python
class PostgresBaseConfig:
    """Shared configuration logic for PostgreSQL backends."""
    
    def _parse_config(self, config: dict) -> tuple[str, str, dict]:
        """Extract table, schema, and connection config."""
        table_name = config.pop("table", "records")
        schema_name = config.pop("schema", "public")
        return table_name, schema_name, config
    
    def _init_vector_support(self):
        """Initialize vector support attributes."""
        self._vector_enabled = False
        self._vector_dimensions = {}
```

### 2. Create PostgresTableManager Mixin
```python
class PostgresTableManager:
    """Shared table management SQL."""
    
    @staticmethod
    def get_create_table_sql(schema_name: str, table_name: str) -> str:
        """Get SQL for creating the table."""
        return f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
            id TEXT PRIMARY KEY,
            data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_data 
        ON {schema_name}.{table_name} USING GIN (data);
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
        ON {schema_name}.{table_name} USING GIN (metadata);
        """
```

### 3. Create PostgresVectorSupport Mixin
```python
class PostgresVectorSupport:
    """Shared vector support detection and management."""
    
    def _has_vector_fields(self, record: Record) -> bool:
        """Check if record has vector fields."""
        from ..fields import VectorField
        return any(isinstance(field, VectorField) 
                   for field in record.fields.values())
    
    def _extract_vector_dimensions(self, record: Record) -> dict[str, int]:
        """Extract dimensions from vector fields."""
        from ..fields import VectorField
        dimensions = {}
        for name, field in record.fields.items():
            if isinstance(field, VectorField):
                dimensions[name] = field.dimensions
        return dimensions
```

### 4. Consolidate Error Handling
```python
class PostgresErrorHandler:
    """Shared error handling logic."""
    
    @staticmethod
    def handle_connection_error(e: Exception):
        """Handle connection errors consistently."""
        logger.error(f"PostgreSQL connection error: {e}")
        raise RuntimeError(f"Database connection failed: {e}")
    
    @staticmethod
    def handle_query_error(e: Exception, operation: str):
        """Handle query execution errors."""
        logger.error(f"PostgreSQL {operation} error: {e}")
        raise RuntimeError(f"Database {operation} failed: {e}")
```

### 5. Use More SQLRecordSerializer Methods
Both classes should delegate more to SQLRecordSerializer:
- Record to JSON conversion ✓ (already done)
- Row to Record conversion ✓ (already done)
- Vector field extraction (could be added)

## Priority Refactoring Tasks

1. **High Priority**:
   - Create PostgresBaseConfig mixin
   - Create PostgresTableManager mixin
   - Consolidate vector support initialization

2. **Medium Priority**:
   - Create PostgresVectorSupport mixin
   - Standardize error handling

3. **Low Priority**:
   - Extract common logging patterns
   - Standardize connection validation

## Benefits of Refactoring

1. **Reduced Duplication**: ~200-300 lines of code could be eliminated
2. **Consistency**: Ensures sync and async versions behave identically
3. **Maintainability**: Bug fixes and features only need to be implemented once
4. **Testing**: Shared logic can be tested once
5. **Extensibility**: Easier to add new PostgreSQL-specific features

## Implementation Notes

- Use mixins rather than inheritance to maintain flexibility
- Keep sync/async separation clear - shared code should be sync-agnostic
- Ensure backward compatibility
- Add comprehensive tests for shared components
