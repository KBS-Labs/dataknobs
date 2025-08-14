# API Reference

## Core Classes

### Record

The `Record` class represents a data record with fields and metadata.

```python
class Record:
    """Represents a data record."""
    
    def __init__(self, fields: Dict[str, Any] = None):
        """
        Initialize a record.
        
        Args:
            fields: Dictionary of field values
        """
        self.fields = fields or {}
        self.metadata = {}
    
    def get_value(self, field_name: str, default: Any = None) -> Any:
        """
        Get field value.
        
        Args:
            field_name: Name of the field
            default: Default value if field not found
        
        Returns:
            Field value or default
        """
    
    def set_value(self, field_name: str, value: Any) -> None:
        """
        Set field value.
        
        Args:
            field_name: Name of the field
            value: Value to set
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Record':
        """Create record from dictionary."""
```

### Query

The `Query` class provides a fluent interface for building database queries.

```python
class Query:
    """Query builder for database operations."""
    
    def filter(self, field: str, operator: str, value: Any) -> 'Query':
        """
        Add a filter condition.
        
        Args:
            field: Field name to filter on
            operator: Comparison operator (=, !=, >, >=, <, <=, IN, NOT IN, LIKE)
            value: Value to compare against
        
        Returns:
            Self for chaining
        """
    
    def sort(self, field: str, order: str = "ASC") -> 'Query':
        """
        Add sorting.
        
        Args:
            field: Field to sort by
            order: Sort order (ASC or DESC)
        
        Returns:
            Self for chaining
        """
    
    def limit(self, limit: int) -> 'Query':
        """
        Set result limit.
        
        Args:
            limit: Maximum number of results
        
        Returns:
            Self for chaining
        """
    
    def offset(self, offset: int) -> 'Query':
        """
        Set result offset.
        
        Args:
            offset: Number of results to skip
        
        Returns:
            Self for chaining
        """
    
    def project(self, fields: List[str]) -> 'Query':
        """
        Select specific fields.
        
        Args:
            fields: List of field names to include
        
        Returns:
            Self for chaining
        """
```

## Database Interface

### Database Abstract Base Class

```python
class Database(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    def create(self, record: Record) -> str:
        """
        Create a new record.
        
        Args:
            record: Record to create
        
        Returns:
            Record ID
        """
    
    @abstractmethod
    def read(self, record_id: str) -> Optional[Record]:
        """
        Read a record by ID.
        
        Args:
            record_id: ID of the record
        
        Returns:
            Record or None if not found
        """
    
    @abstractmethod
    def update(self, record_id: str, record: Record) -> bool:
        """
        Update an existing record.
        
        Args:
            record_id: ID of the record
            record: Updated record data
        
        Returns:
            True if successful, False otherwise
        """
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        Delete a record.
        
        Args:
            record_id: ID of the record
        
        Returns:
            True if successful, False otherwise
        """
    
    @abstractmethod
    def search(self, query: Query) -> List[Record]:
        """
        Search for records.
        
        Args:
            query: Query object with filters and options
        
        Returns:
            List of matching records
        """
    
    @abstractmethod
    def count(self, query: Optional[Query] = None) -> int:
        """
        Count records.
        
        Args:
            query: Optional query to filter records
        
        Returns:
            Number of records
        """
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all records from the database."""
    
    def batch_create(self, records: List[Record]) -> List[str]:
        """
        Create multiple records.
        
        Args:
            records: List of records to create
        
        Returns:
            List of record IDs
        """
    
    def batch_read(self, record_ids: List[str]) -> List[Optional[Record]]:
        """
        Read multiple records.
        
        Args:
            record_ids: List of record IDs
        
        Returns:
            List of records (None for not found)
        """
    
    def batch_update(self, updates: List[Tuple[str, Record]]) -> List[bool]:
        """
        Update multiple records.
        
        Args:
            updates: List of (record_id, record) tuples
        
        Returns:
            List of success flags
        """
    
    def batch_delete(self, record_ids: List[str]) -> List[bool]:
        """
        Delete multiple records.
        
        Args:
            record_ids: List of record IDs
        
        Returns:
            List of success flags
        """
```

## Factory Pattern

### DatabaseFactory

```python
class DatabaseFactory(FactoryBase):
    """Factory for creating database instances."""
    
    def create(self, **config) -> Database:
        """
        Create a database instance.
        
        Args:
            backend: Backend type (memory, file, postgres, elasticsearch, s3)
            **config: Backend-specific configuration
        
        Returns:
            Database instance
        
        Raises:
            ValueError: If backend is not supported
            ImportError: If backend dependencies are not installed
        """
    
    def get_available_backends(self) -> List[str]:
        """
        Get list of available backends.
        
        Returns:
            List of backend names
        """
    
    def get_backend_info(self, backend: str) -> Dict[str, Any]:
        """
        Get information about a backend.
        
        Args:
            backend: Backend name
        
        Returns:
            Dictionary with backend information:
            - description: Backend description
            - persistent: Whether backend persists data
            - requires_install: Installation command if needed
            - required_params: List of required parameters
            - optional_params: List of optional parameters
        """
    
    def is_backend_available(self, backend: str) -> bool:
        """
        Check if backend is available.
        
        Args:
            backend: Backend name
        
        Returns:
            True if backend can be used
        """
    
    def register_backend(self, name: str, backend_class: Type[Database]) -> None:
        """
        Register a custom backend.
        
        Args:
            name: Backend name
            backend_class: Backend class
        """
```

### database_factory

```python
# Pre-configured factory instance
database_factory = DatabaseFactory()
```

## Backend Implementations

### MemoryDatabase

```python
class MemoryDatabase(Database, ConfigurableBase):
    """In-memory database implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize memory database.
        
        Args:
            config: Optional configuration
        """
```

### FileDatabase

```python
class FileDatabase(Database, ConfigurableBase):
    """File-based database implementation."""
    
    def __init__(self, path: str = None, format: str = "json", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize file database.
        
        Args:
            path: File path
            format: File format (json, csv, parquet)
            config: Optional configuration
        """
```

### PostgresDatabase

```python
class PostgresDatabase(Database, ConfigurableBase):
    """PostgreSQL database implementation."""
    
    def __init__(self, host: str = None, port: int = None,
                 database: str = None, user: str = None,
                 password: str = None, table: str = "records",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL database.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            table: Table name
            config: Optional configuration
        """
```

### ElasticsearchDatabase

```python
class ElasticsearchDatabase(Database, ConfigurableBase):
    """Elasticsearch database implementation."""
    
    def __init__(self, hosts: List[str] = None, index: str = "records",
                 username: str = None, password: str = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Elasticsearch database.
        
        Args:
            hosts: List of Elasticsearch hosts
            index: Index name
            username: Optional username
            password: Optional password
            config: Optional configuration
        """
```

### S3Database

```python
class S3Database(Database, ConfigurableBase):
    """AWS S3 database implementation."""
    
    def __init__(self, bucket: str = None, prefix: str = "",
                 region: str = "us-east-1", endpoint_url: str = None,
                 access_key_id: str = None, secret_access_key: str = None,
                 max_workers: int = 10,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize S3 database.
        
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
            region: AWS region
            endpoint_url: Custom endpoint (for LocalStack/MinIO)
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            max_workers: Number of parallel workers
            config: Optional configuration
        """
```

## Configuration Support

### ConfigurableBase

All database backends inherit from `ConfigurableBase`:

```python
class ConfigurableBase:
    """Base class for configuration support."""
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ConfigurableBase':
        """
        Create instance from configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Configured instance
        """
```

## Utility Functions

### Type Conversion

```python
def convert_type(value: Any, target_type: type) -> Any:
    """
    Convert value to target type.
    
    Args:
        value: Value to convert
        target_type: Target type
    
    Returns:
        Converted value
    """
```

### ID Generation

```python
def generate_id() -> str:
    """
    Generate a unique record ID.
    
    Returns:
        UUID string
    """
```

## Exceptions

```python
class DatabaseError(Exception):
    """Base exception for database errors."""

class ConnectionError(DatabaseError):
    """Database connection error."""

class QueryError(DatabaseError):
    """Query execution error."""

class RecordNotFoundError(DatabaseError):
    """Record not found error."""

class BackendNotAvailableError(DatabaseError):
    """Backend not available or not installed."""
```

## Type Hints

```python
from typing import Dict, Any, List, Optional, Tuple, Type
from abc import ABC, abstractmethod

# Type aliases
RecordID = str
FieldName = str
FieldValue = Any
QueryOperator = Literal["=", "!=", ">", ">=", "<", "<=", "IN", "NOT IN", "LIKE"]
SortOrder = Literal["ASC", "DESC"]
```

## Constants

```python
# Default values
DEFAULT_BATCH_SIZE = 100
DEFAULT_POOL_SIZE = 10
DEFAULT_TIMEOUT = 30
DEFAULT_CACHE_TTL = 60

# S3 specific
S3_MULTIPART_THRESHOLD = 5 * 1024 * 1024  # 5MB
S3_MULTIPART_CHUNKSIZE = 5 * 1024 * 1024  # 5MB
S3_MAX_WORKERS = 10

# Elasticsearch specific
ES_DEFAULT_INDEX = "records"
ES_DEFAULT_DOC_TYPE = "_doc"
ES_BATCH_SIZE = 500

# PostgreSQL specific
PG_DEFAULT_TABLE = "records"
PG_DEFAULT_PORT = 5432
PG_POOL_SIZE = 20
```