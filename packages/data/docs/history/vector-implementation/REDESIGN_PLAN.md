# Validation and Migration Module Redesign Plan

## Context

During Phase 9 (Testing & Quality) implementation, we discovered significant design issues in both the validation and migration modules while attempting to write tests. The APIs were inconsistent, unpredictable, and difficult to test. Since this code hasn't been released yet, we can do a complete redesign without backwards compatibility concerns.

## Problems Discovered

### Validation Module Issues

1. **Constraint API Inconsistency**
   - `validate()` returns only boolean, no error messages
   - Error messages obtained through separate `get_error_message()` method
   - `UniqueConstraint` is stateful (tracks seen values) while others are stateless
   - `from_dict()` factory method passes entire config dict to constructor, causing unexpected 'type' parameter

2. **Type Coercion Confusion**
   - No consistent return type - sometimes returns None, sometimes raises exceptions
   - Can't distinguish between successful coercion to None vs failure
   - No clear success/failure indication

3. **Over-complicated Schema System**
   - Too many classes: Schema, FieldDefinition, SchemaValidator, ValidationResult, ValidationError
   - Validation logic split across multiple classes
   - Unclear separation of concerns

### Migration Module Issues

1. **API Naming Inconsistencies**
   - Migration constructor expects 'operations' but tests assumed 'changes'
   - DataTransformer has `exclude_fields()` but tests expected `exclude_field()`
   - TransformationPipeline has `add()` but tests expected `add_transformer()`

2. **Structural Problems**
   - SchemaEvolution stores versions as dict but tests expected list-like access
   - Tight coupling between migration operations and record structure
   - Progress tracking mixed with migration logic
   - No clear error handling for partial migrations

## Design Principles for Redesign

1. **Consistency**: All methods return predictable types
2. **Simplicity**: Fewer classes, clearer responsibilities
3. **Composability**: Components can be easily combined
4. **Testability**: Simple, predictable behavior that's easy to test
5. **Fluent APIs**: Chainable methods for readable code
6. **Separation of Concerns**: Each class has one clear responsibility
7. **Fail-Safe**: Clear error handling, no surprising exceptions

## Redesign Specifications

### Database API Enhancement: Streaming Support

#### Problem
Current database API requires loading all records into memory for migrations and bulk operations. This is inefficient and can cause memory issues with large datasets.

#### Solution: Streaming API

```python
from typing import Iterator, AsyncIterator, Optional, Callable
from dataclasses import dataclass

@dataclass
class StreamConfig:
    """Configuration for streaming operations."""
    batch_size: int = 1000
    prefetch: int = 2  # Number of batches to prefetch
    timeout: Optional[float] = None
    on_error: Optional[Callable[[Exception, Record], bool]] = None  # Return True to continue

class Database(ABC):
    """Enhanced database base class with streaming support."""
    
    # Existing methods...
    
    @abstractmethod
    def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> Iterator[Record]:
        """
        Stream records from database.
        Yields records one at a time, fetching in batches internally.
        """
        pass
    
    @abstractmethod
    def stream_write(
        self,
        records: Iterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """
        Stream records into database.
        Accepts an iterator and writes in batches.
        """
        pass
    
    def stream_transform(
        self,
        query: Optional[Query] = None,
        transform: Optional[Callable[[Record], Optional[Record]]] = None,
        config: Optional[StreamConfig] = None
    ) -> Iterator[Record]:
        """
        Stream records through a transformation.
        Default implementation, can be overridden for efficiency.
        """
        for record in self.stream_read(query, config):
            if transform:
                transformed = transform(record)
                if transformed:  # None means filter out
                    yield transformed
            else:
                yield record

class AsyncDatabase(ABC):
    """Async version with streaming support."""
    
    @abstractmethod
    async def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[Record]:
        """Async streaming read."""
        pass
    
    @abstractmethod  
    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Async streaming write."""
        pass

@dataclass
class StreamResult:
    """Result of streaming operation."""
    total_processed: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]]
    duration: float
```

#### Backend Implementations

```python
# Memory backend example
class MemoryDatabase(Database):
    def stream_read(self, query=None, config=None):
        """Memory backend can yield directly from storage."""
        config = config or StreamConfig()
        records = self._apply_query(query)
        
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

# File backend example  
class FileDatabase(Database):
    def stream_read(self, query=None, config=None):
        """File backend streams from disk."""
        config = config or StreamConfig()
        
        # For JSON Lines format
        with open(self.filepath, 'r') as f:
            batch = []
            for line in f:
                record = self._deserialize(line)
                if self._matches_query(record, query):
                    yield record

# PostgreSQL example
class PostgresDatabase(Database):
    def stream_read(self, query=None, config=None):
        """Use server-side cursor for efficient streaming."""
        config = config or StreamConfig()
        
        with self.connection.cursor(name='stream_cursor') as cursor:
            cursor.itersize = config.batch_size
            cursor.execute(self._build_query(query))
            
            for row in cursor:
                yield self._row_to_record(row)
    
    def stream_write(self, records, config=None):
        """Use COPY for efficient bulk insert."""
        config = config or StreamConfig()
        result = StreamResult(0, 0, 0, [], 0)
        
        with self.connection.cursor() as cursor:
            # Use PostgreSQL COPY for efficiency
            copy_sql = "COPY records FROM STDIN WITH CSV"
            with cursor.copy(copy_sql) as copy:
                batch = []
                for record in records:
                    batch.append(self._record_to_row(record))
                    if len(batch) >= config.batch_size:
                        copy.write_all(batch)
                        result.total_processed += len(batch)
                        batch = []
                if batch:
                    copy.write_all(batch)
                    result.total_processed += len(batch)
        
        return result

# S3 backend example
class S3Database(Database):
    def stream_read(self, query=None, config=None):
        """Stream objects from S3."""
        config = config or StreamConfig()
        
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                response = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                record = self._deserialize(response['Body'].read())
                if self._matches_query(record, query):
                    yield record
```

#### Migration Integration

```python
class Migrator:
    """Enhanced migrator using streaming."""
    
    def migrate_stream(
        self,
        source: Database,
        target: Database,
        transform: Optional[Transformer] = None,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> MigrationProgress:
        """
        Stream-based migration for memory efficiency.
        Never loads full dataset into memory.
        """
        config = config or StreamConfig()
        progress = MigrationProgress()
        
        # Create a streaming pipeline
        source_stream = source.stream_read(query, config)
        
        if transform:
            source_stream = (
                transform.transform(record) 
                for record in source_stream
            )
        
        # Stream into target
        result = target.stream_write(source_stream, config)
        
        progress.total = result.total_processed
        progress.succeeded = result.successful
        progress.failed = result.failed
        
        return progress
    
    def migrate_parallel(
        self,
        source: Database,
        target: Database,
        transform: Optional[Transformer] = None,
        partitions: int = 4
    ) -> MigrationProgress:
        """
        Parallel streaming migration.
        Partition data and migrate in parallel streams.
        """
        import concurrent.futures
        
        def migrate_partition(partition_id):
            query = Query().filter("partition_id", "=", partition_id)
            return self.migrate_stream(source, target, transform, query)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=partitions) as executor:
            futures = [
                executor.submit(migrate_partition, i) 
                for i in range(partitions)
            ]
            
            total_progress = MigrationProgress()
            for future in concurrent.futures.as_completed(futures):
                partition_progress = future.result()
                total_progress.merge(partition_progress)
            
            return total_progress
```

#### Benefits

1. **Memory Efficiency**: Never load entire dataset into memory
2. **Performance**: Use database-specific optimizations (cursors, COPY, etc.)
3. **Scalability**: Handle datasets larger than available RAM
4. **Composability**: Chain streaming operations
5. **Error Recovery**: Process errors per-record without stopping
6. **Progress Tracking**: Natural batching for progress updates
7. **Parallel Processing**: Easy to partition and parallelize

### Validation Module v2

#### Core Components

```python
# 1. Unified Result Object
@dataclass
class ValidationResult:
    """All validation operations return this."""
    valid: bool
    value: Any  # The (possibly coerced) value
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        """Allow: if result: usage"""
        return self.valid
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Combine results for composite validation."""
        return ValidationResult(
            valid=self.valid and other.valid,
            value=other.value if other.valid else self.value,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings
        )

# 2. Clean Constraint API
class Constraint(ABC):
    """Base for all constraints."""
    
    @abstractmethod
    def check(self, value: Any, context: Optional[ValidationContext] = None) -> ValidationResult:
        """Single method for validation."""
        pass
    
    def __and__(self, other: 'Constraint') -> 'All':
        """Combine with AND: both must pass."""
        return All([self, other])
    
    def __or__(self, other: 'Constraint') -> 'Any':
        """Combine with OR: at least one must pass."""
        return Any([self, other])

# 3. Simple Schema Definition
class Schema:
    """Schema definition and validation in one."""
    
    def __init__(self, name: str, strict: bool = False):
        """
        Args:
            name: Schema name
            strict: If True, reject unknown fields
        """
        self.name = name
        self.strict = strict
        self.fields: Dict[str, Field] = {}
    
    def field(self, name: str, type: FieldType, **options) -> 'Schema':
        """Add a field (fluent API)."""
        self.fields[name] = Field(name, type, **options)
        return self
    
    def validate(self, record: Record, coerce: bool = False) -> ValidationResult:
        """Validate a record."""
        # Single, clear validation method
        pass
```

#### Constraint Implementations

```python
# Simple, focused constraints
class Required(Constraint):
    """Field must be present and non-null."""
    def __init__(self, allow_empty: bool = False):
        self.allow_empty = allow_empty

class Range(Constraint):
    """Numeric value in range."""
    def __init__(self, min: Optional[Number] = None, max: Optional[Number] = None):
        self.min = min
        self.max = max

class Length(Constraint):
    """String/collection length."""
    def __init__(self, min: Optional[int] = None, max: Optional[int] = None):
        self.min = min
        self.max = max

class Pattern(Constraint):
    """Regex pattern match."""
    def __init__(self, pattern: str):
        self.regex = re.compile(pattern)

class Enum(Constraint):
    """Value in allowed set."""
    def __init__(self, values: List[Any]):
        self.allowed = set(values)

class Unique(Constraint):
    """Value uniqueness (stateless via context)."""
    def check(self, value: Any, context: ValidationContext) -> ValidationResult:
        # Use context to track seen values, not internal state
        pass
```

#### Type Coercion

```python
class Coercer:
    """Type coercion with predictable results."""
    
    def coerce(self, value: Any, target_type: Type) -> ValidationResult:
        """
        Always returns ValidationResult.
        Never raises exceptions.
        """
        if value is None:
            return ValidationResult(
                valid=False,
                value=None,
                errors=[f"Cannot coerce None to {target_type.__name__}"]
            )
        
        try:
            coerced = self._coerce_value(value, target_type)
            return ValidationResult(valid=True, value=coerced)
        except Exception as e:
            return ValidationResult(
                valid=False,
                value=value,  # Return original
                errors=[f"Coercion failed: {e}"]
            )
```

### Migration Module v2

#### Core Components

```python
# 1. Clear Operation Definition
@dataclass
class Operation:
    """Single, reversible operation."""
    
    def apply(self, record: Record) -> Record:
        """Apply operation."""
        pass
    
    def reverse(self, record: Record) -> Record:
        """Reverse operation."""
        pass

# 2. Simple Migration
class Migration:
    """Migration between versions."""
    
    def __init__(self, from_version: str, to_version: str):
        self.from_version = from_version
        self.to_version = to_version
        self.operations: List[Operation] = []
    
    def add(self, operation: Operation) -> 'Migration':
        """Add operation (fluent)."""
        self.operations.append(operation)
        return self
    
    def apply(self, record: Record, reverse: bool = False) -> Record:
        """Apply all operations."""
        ops = reversed(self.operations) if reverse else self.operations
        for op in ops:
            record = op.reverse(record) if reverse else op.apply(record)
        return record

# 3. Stateless Transformer
class Transformer:
    """Record transformation."""
    
    def __init__(self):
        self.rules: List[TransformRule] = []
    
    def map(self, source: str, target: str = None, transform: Callable = None) -> 'Transformer':
        """Map field (fluent)."""
        self.rules.append(MapRule(source, target or source, transform))
        return self
    
    def exclude(self, *fields: str) -> 'Transformer':
        """Exclude fields (fluent)."""
        self.rules.append(ExcludeRule(fields))
        return self
    
    def transform(self, record: Record) -> Record:
        """Apply all rules."""
        result = record.copy()
        for rule in self.rules:
            result = rule.apply(result)
        return result

# 4. Separate Progress Tracking
class MigrationProgress:
    """Progress tracking, separate from logic."""
    
    def __init__(self):
        self.total = 0
        self.processed = 0
        self.succeeded = 0
        self.failed = 0
        self.errors: List[Dict] = []
    
    @property
    def percent(self) -> float:
        return (self.processed / self.total * 100) if self.total else 0

# 5. Clean Migrator
class Migrator:
    """Data migration orchestrator."""
    
    def migrate(
        self,
        source: Database,
        target: Database,
        transform: Optional[Transformer] = None,
        batch_size: int = 1000,
        on_progress: Optional[Callable[[MigrationProgress], None]] = None
    ) -> MigrationProgress:
        """
        Migrate data between databases.
        Progress tracking completely separate from logic.
        """
        progress = MigrationProgress()
        # Clean migration logic
        return progress
```

## Implementation Steps

### Phase 1: Database Streaming API
1. Add streaming methods to base Database class
2. Implement StreamConfig and StreamResult classes
3. Add stream_read and stream_write to all backends
4. Implement efficient backend-specific optimizations
5. Add comprehensive streaming tests

### Phase 2: Create New Modules
1. Create `validation_v2/` directory with new implementation
2. Create `migration_v2/` directory with new implementation
3. Keep old modules temporarily for reference

### Phase 3: Implement Core Components
1. Implement ValidationResult and core validation classes
2. Implement all constraint types with consistent API
3. Implement Schema with fluent field definition
4. Implement Coercer with predictable behavior
5. Implement Migration operations and transformers
6. Update Migrator to use streaming API

### Phase 4: Write Comprehensive Tests

#### Testing Strategy: Use Real Components Instead of Mocks
- **Use MemoryDatabase** instead of mock databases for testing migrations
- **Use actual Transformer instances** instead of mock transformers
- **Use real ValidationResult objects** instead of mock results
- **Use FileDatabase with temp files** for testing file operations
- **Benefits**: Better integration testing, higher code coverage, more realistic test scenarios

#### Test Coverage Plan
1. **Streaming API Tests**
   - Test with MemoryDatabase (always available, no external deps)
   - Test with FileDatabase using temp files
   - Test error recovery with real error scenarios
   - Test batch processing with actual data

2. **Validation Tests**
   - Use real Field and Record objects
   - Test constraint combinations with actual constraints
   - Test schema validation with real schemas
   - Test coercion with actual TypeCoercer

3. **Migration Tests**
   - Use MemoryDatabase as source and target
   - Test with real Transformer instances
   - Test streaming between different backend types
   - Test parallel migrations with actual partitioning

4. **Integration Tests**
   - Memory → File migration with real databases
   - File → Memory migration with transformations
   - Schema validation during migration
   - Progress tracking with actual callbacks

5. **Component Tests**
   - Test each constraint individually with real Records
   - Test constraint composition (AND/OR) with actual constraints
   - Test schema validation with various real scenarios
   - Test coercion for all type combinations
   - Test migration operations and reversibility
   - Test transformer rules with real data
   - Test streaming migrations with actual streams
   - Test parallel migrations with real parallelization

### Phase 5: Integration
1. Update factory classes to use new modules
2. Update any dependent code
3. Remove old validation/ and migration/ directories
4. Update all documentation

### Phase 6: Validation
1. Ensure test coverage > 90% for new modules
2. Run performance benchmarks
3. Update PROGRESS_CHECKLIST.md

## Success Criteria

1. **API Consistency**: Every method returns predictable types
2. **Test Coverage**: >90% coverage for new modules
3. **Documentation**: Every public method documented
4. **No Surprises**: No unexpected exceptions or behaviors
5. **Composability**: Components easily combined
6. **Performance**: No regression from current implementation

## Benefits of This Redesign

1. **Predictable**: Developers can guess behavior from names
2. **Testable**: Simple classes with single responsibilities
3. **Maintainable**: Clear structure, easy to extend
4. **Usable**: Fluent APIs, helpful error messages
5. **Robust**: Consistent error handling throughout
6. **Scalable**: Streaming API handles datasets of any size
7. **Efficient**: Memory-bounded operations, database-specific optimizations
8. **Composable**: Stream transformations can be chained
9. **Parallel**: Built-in support for parallel streaming

## Testing Philosophy: Real Components Over Mocks

### Example Test Structure

```python
# Instead of mocking, use real components
import tempfile
from pathlib import Path

def test_migration_with_validation():
    """Test migration with real components, no mocks."""
    
    # Use real databases (no external dependencies)
    source = MemoryDatabase()
    target = MemoryDatabase()
    
    # Create real test data
    for i in range(100):
        record = Record()
        record.add_field("id", i)
        record.add_field("name", f"item_{i}")
        record.add_field("value", i * 10)
        source.create(record)
    
    # Use real schema for validation
    schema = Schema("TestSchema")
        .field("id", FieldType.INTEGER, required=True)
        .field("name", FieldType.STRING, constraints=[Pattern(r"^item_\d+$")])
        .field("value", FieldType.INTEGER, constraints=[Range(min=0, max=1000)])
    
    # Use real transformer
    transformer = Transformer()
        .map("value", transform=lambda x: x * 2)
        .exclude("internal_field")
    
    # Use real migrator with streaming
    migrator = Migrator()
    progress = migrator.migrate_stream(
        source=source,
        target=target,
        transform=transformer,
        config=StreamConfig(batch_size=10)
    )
    
    # Verify with real queries
    results = list(target.stream_read())
    assert len(results) == 100
    assert all(schema.validate(r).valid for r in results)
    assert all(r.fields["value"].value == i * 20 for i, r in enumerate(results))

def test_file_backend_integration():
    """Test with real file backend using temp files."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use real file database
        file_db = FileDatabase(Path(tmpdir) / "test.json")
        
        # Use real memory database
        memory_db = MemoryDatabase()
        
        # Real data
        test_data = [Record().add_field("id", i) for i in range(50)]
        
        # Test real streaming between backends
        file_db.stream_write(iter(test_data))
        
        # Stream from file to memory
        migrator = Migrator()
        result = migrator.migrate_stream(file_db, memory_db)
        
        assert result.succeeded == 50
        assert memory_db.count() == 50

def test_parallel_migration_with_real_parallelization():
    """Test parallel migration with actual threading."""
    
    source = MemoryDatabase()
    target = MemoryDatabase()
    
    # Create data with partition field
    for i in range(1000):
        record = Record()
        record.add_field("id", i)
        record.add_field("partition", i % 4)
        source.create(record)
    
    # Real parallel migration
    migrator = Migrator()
    progress = migrator.migrate_parallel(
        source=source,
        target=target,
        partitions=4
    )
    
    assert progress.succeeded == 1000
    assert target.count() == 1000
```

### Benefits of This Approach

1. **Better Coverage**: Testing real code paths increases coverage
2. **Integration Testing**: Tests how components actually work together
3. **Realistic Scenarios**: Tests reflect actual usage patterns
4. **No Mock Maintenance**: No need to update mocks when APIs change
5. **Performance Testing**: Can measure real performance characteristics
6. **Edge Case Discovery**: Real components reveal actual edge cases

## Notes

- No backwards compatibility needed (unreleased code)
- Focus on developer experience and API clarity
- Prioritize simplicity over flexibility
- Make the common case easy, the complex case possible
- Use real components in tests instead of mocks whenever possible