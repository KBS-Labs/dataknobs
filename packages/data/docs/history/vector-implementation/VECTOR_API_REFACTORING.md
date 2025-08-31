# Vector API Refactoring Plan

## Overview
This document outlines the refactoring of vector-related APIs to make them more intuitive and easier to use, based on issues discovered during testing.

## Goals
1. Simplify initialization patterns
2. Provide sensible defaults
3. Support both simple and advanced use cases
4. Improve discoverability through clear method names
5. Ensure consistency across all vector APIs

## Refactoring Details

### 1. VectorField

**Current Issues:**
- Requires `name` as first argument even when used in Record context
- Name is redundant when field name is the dictionary key

**Proposed Changes:**
```python
class VectorField(Field):
    def __init__(
        self,
        value: np.ndarray | list[float],
        name: str = None,  # Make optional, default to None
        dimensions: int = None,  # Auto-detect from value
        source_field: str = None,
        model_name: str = None,
        model_version: str = None,
        metadata: dict = None,
    ):
        # Auto-detect dimensions if not provided
        if dimensions is None and value is not None:
            dimensions = len(value) if isinstance(value, list) else value.shape[0]
        
        # Name will be set by Record when added
        if name is None:
            name = "embedding"  # Default name
    
    @classmethod
    def from_text(
        cls,
        text: str,
        embedding_fn: Callable,
        dimensions: int = None,
        **kwargs
    ) -> "VectorField":
        """Create a VectorField from text using an embedding function."""
        embedding = embedding_fn(text)
        return cls(value=embedding, dimensions=dimensions, **kwargs)
```

### 2. VectorTextSynchronizer

**Current Issues:**
- No clear way to configure field mappings
- Relies on database schema that may not exist
- Complex initialization

**Proposed Changes:**
```python
class VectorTextSynchronizer:
    def __init__(
        self,
        database: Database,
        embedding_fn: Callable,
        text_fields: list[str] = None,  # Primary configuration
        vector_field: str = "embedding",  # Sensible default
        field_separator: str = " ",
        auto_sync: bool = True,  # Auto-sync on create/update
        batch_size: int = 100,
        model_name: str = None,
        model_version: str = None,
    ):
        """Simplified initialization with common parameters."""
        
    async def sync_all(self, force: bool = False) -> SyncResult:
        """Sync all records in the database."""
        
    async def sync_record(self, record: Record) -> bool:
        """Sync a single record."""
        
    async def sync_batch(self, records: list[Record]) -> SyncResult:
        """Sync a batch of records."""
    
    @classmethod
    def from_config(cls, database: Database, config: SyncConfig) -> "VectorTextSynchronizer":
        """Create from a config object for advanced use cases."""
```

### 3. VectorMigration

**Current Issues:**
- Confusing start/wait pattern
- No simple run method
- Configuration is complex

**Proposed Changes:**
```python
class VectorMigration:
    def __init__(
        self,
        source_db: Database,
        target_db: Database,
        embedding_fn: Callable,
        text_fields: list[str] = None,
        vector_field: str = "embedding",
        field_separator: str = " ",
        batch_size: int = 100,
        model_name: str = None,
        model_version: str = None,
    ):
        """Simplified initialization."""
    
    async def run(self, progress_callback: Callable = None) -> MigrationResult:
        """Run the complete migration."""
        await self.start()
        return await self.wait_for_completion()
    
    async def run_incremental(
        self, 
        checkpoint_interval: int = 1000,
        resume_from: str = None
    ) -> MigrationResult:
        """Run migration with checkpointing for large datasets."""
    
    @property
    def status(self) -> MigrationStatus:
        """Get current migration status."""
    
    @classmethod
    def from_config(
        cls, 
        source_db: Database,
        target_db: Database,
        config: MigrationConfig
    ) -> "VectorMigration":
        """Create from config for advanced use cases."""
```

### 4. IncrementalVectorizer

**Current Issues:**
- Requires single source field
- No support for multiple text fields
- Complex initialization

**Proposed Changes:**
```python
class IncrementalVectorizer:
    def __init__(
        self,
        database: Database,
        embedding_fn: Callable,
        text_fields: list[str] | str = None,  # Support multiple fields
        vector_field: str = "embedding",
        field_separator: str = " ",
        batch_size: int = 100,
        checkpoint_interval: int = 1000,
        model_name: str = None,
        model_version: str = None,
    ):
        """Simplified initialization with sensible defaults."""
        # Convert single field to list
        if isinstance(text_fields, str):
            text_fields = [text_fields]
    
    async def run(self, resume_from: str = None) -> VectorizationResult:
        """Run the complete vectorization."""
    
    async def run_batch(self, limit: int = None) -> VectorizationResult:
        """Process a limited number of records."""
    
    @property
    def progress(self) -> VectorizationProgress:
        """Get current progress."""
    
    async def get_checkpoint(self) -> str:
        """Get checkpoint ID for resuming."""
```

### 5. Configuration Classes

**Simplify configuration with builder pattern:**
```python
@dataclass
class VectorConfig:
    """Unified configuration for vector operations."""
    text_fields: list[str] = field(default_factory=lambda: ["content"])
    vector_field: str = "embedding"
    field_separator: str = " "
    dimensions: int = None  # Auto-detect
    metric: str = "cosine"
    model_name: str = None
    model_version: str = None
    
    def with_fields(self, *fields: str) -> "VectorConfig":
        """Builder method for setting text fields."""
        self.text_fields = list(fields)
        return self
    
    def with_model(self, name: str, version: str = None) -> "VectorConfig":
        """Builder method for setting model info."""
        self.model_name = name
        self.model_version = version
        return self
```

## Usage Examples After Refactoring

### Simple Usage
```python
# VectorField - no name required when used in Record
record = Record({
    "title": "Machine Learning",
    "content": "Introduction to ML",
    "embedding": VectorField(embedding_values)  # Name inferred from key
})

# VectorTextSynchronizer - simple initialization
sync = VectorTextSynchronizer(
    db, 
    embedding_fn,
    text_fields=["title", "content"]  # Everything else has defaults
)
result = await sync.sync_all()

# VectorMigration - simple run
migration = VectorMigration(
    source_db,
    target_db, 
    embedding_fn,
    text_fields=["title", "content"]
)
result = await migration.run()

# IncrementalVectorizer - multiple fields support
vectorizer = IncrementalVectorizer(
    db,
    embedding_fn,
    text_fields=["title", "content", "tags"]  # Multiple fields
)
result = await vectorizer.run()
```

### Advanced Usage with Config
```python
# Using builder pattern
config = VectorConfig() \
    .with_fields("title", "content", "summary") \
    .with_model("all-MiniLM-L6-v2", "1.0.0")

sync = VectorTextSynchronizer.from_config(db, config)
migration = VectorMigration.from_config(source_db, target_db, config)
```

## Migration Strategy

1. **Phase 1**: Add new methods/signatures alongside existing ones
2. **Phase 2**: Update examples and tests to use new APIs
3. **Phase 3**: Add deprecation warnings to old methods
4. **Phase 4**: Remove old methods in next major version

## Documentation Requirements

Each class should have:
1. Clear docstring with purpose and common use cases
2. Example usage in the class docstring
3. Parameter descriptions with defaults explained
4. Return type documentation
5. Examples showing both simple and advanced usage

## Testing Requirements

1. Unit tests for each refactored method
2. Integration tests showing real-world usage
3. Tests for backward compatibility during migration
4. Performance tests to ensure no regression