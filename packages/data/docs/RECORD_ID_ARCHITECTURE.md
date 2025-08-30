# Record ID Architecture

## Overview

The DataKnobs Data Package implements a sophisticated ID management system that cleanly separates user-defined identifiers from system-assigned storage identifiers. This architecture ensures data integrity while maintaining flexibility for user applications.

## The Two-ID Concept

### 1. Storage ID (System ID)
- **Purpose**: Unique identifier assigned by the storage backend
- **Generation**: Automatically created when a record is stored
- **Format**: Typically UUID v4 (backend-dependent)
- **Access**: Via `record.storage_id` property
- **Mutability**: Set once by the database, immutable thereafter

### 2. User ID (Data Field)
- **Purpose**: Application-specific identifier in the record's data
- **Location**: Stored as a field named "id" in the record's data
- **Format**: Any user-defined format (string, integer, etc.)
- **Access**: Via `record.get_user_id()` or `record.get_value("id")`
- **Mutability**: Can be changed by the application

## The ID Conflict Problem

### Background
Records can have an "id" field as part of their business data. This creates ambiguity:
- When a record has `{"id": "user-123", "name": "Test"}`, what does `record.id` return?
- When the database assigns storage ID "uuid-456", which ID is used for updates?

### Historical Issue
```python
# User creates a record with an ID field
record = Record({"id": 1, "title": "Test"})
print(record.id)  # Returns "1" (the field value)

# Database stores it with a UUID
record_id = await db.create(record)  # Returns "abc-123-def"

# Later, trying to update fails
await db.update(record.id, updated_record)  # Tries to update "1", not "abc-123-def"!
```

## The Solution: ID Priority System

### Record Class Properties

```python
class Record:
    _storage_id: str | None  # System-assigned storage ID
    fields: dict             # User data (may include "id" field)
    
    @property
    def storage_id(self) -> str | None:
        """Get the storage system ID."""
        return self._storage_id
    
    @property
    def id(self) -> str | None:
        """Get the record ID with smart priority:
        1. Storage ID (if set)
        2. User-defined 'id' field
        3. None
        """
        if self._storage_id is not None:
            return self._storage_id
        return self.get_value("id")
    
    def get_user_id(self) -> str | None:
        """Explicitly get the user-defined ID field."""
        return self.get_value("id")
    
    def has_storage_id(self) -> bool:
        """Check if storage ID is assigned."""
        return self._storage_id is not None
```

### Priority Resolution

The `record.id` property uses intelligent priority:

1. **After storage**: Returns storage_id (database operations)
2. **Before storage**: Returns user's "id" field (if present)
3. **No ID**: Returns None

This ensures backwards compatibility while preventing ID conflicts.

## Implementation in Backends

### Centralized Helper Methods

All database backends inherit these helper methods from the base `Database` class:

```python
def _prepare_record_for_storage(self, record: Record) -> tuple[Record, str]:
    """Prepare a record for storage by ensuring it has a storage_id.
    
    Returns:
        Tuple of (prepared_record_copy, storage_id)
    """
    record_copy = record.copy(deep=True)
    
    if not record_copy.has_storage_id():
        storage_id = str(uuid.uuid4())
        record_copy.storage_id = storage_id
    else:
        storage_id = record_copy.storage_id
        
    return record_copy, storage_id

def _prepare_record_from_storage(self, record: Record | None, storage_id: str) -> Record | None:
    """Prepare a record retrieved from storage by ensuring storage_id is set.
    
    Returns:
        Record with storage_id set, or None
    """
    if record:
        record_copy = record.copy(deep=True)
        if not record_copy.has_storage_id():
            record_copy.storage_id = storage_id
        return record_copy
    return None
```

### Backend Usage Example

```python
# In any backend's create method
async def create(self, record: Record) -> str:
    # Use centralized method to prepare record
    record_copy, storage_id = self._prepare_record_for_storage(record)
    
    # Store the record with its storage ID
    self._storage[storage_id] = record_copy
    return storage_id

# In any backend's read method
async def read(self, id: str) -> Record | None:
    record = self._storage.get(id)
    # Use centralized method to ensure storage_id is set
    return self._prepare_record_from_storage(record, id)
```

## Usage Patterns

### Creating Records

```python
# User creates record with their own ID
record = Record({"id": "user-123", "name": "Test"})
print(record.id)  # "user-123" (user field)
print(record.storage_id)  # None

# Store in database
storage_id = await db.create(record)
print(storage_id)  # "uuid-generated-456"

# Read it back
retrieved = await db.read(storage_id)
print(retrieved.id)  # "uuid-generated-456" (storage_id takes priority)
print(retrieved.get_user_id())  # "user-123" (user field still accessible)
print(retrieved.storage_id)  # "uuid-generated-456"
```

### Updating Records

```python
# After retrieval, use record.id for updates (it returns storage_id)
retrieved.fields["name"].value = "Updated"
success = await db.update(retrieved.id, retrieved)  # Uses storage_id correctly
```

### Searching by User ID

```python
# Find records by user-defined ID field
from dataknobs_data import Query

query = Query().filter("id", "==", "user-123")
results = await db.search(query)
# This searches the "id" field in data, not storage_id
```

## Benefits

1. **No ID Conflicts**: System and user IDs are clearly separated
2. **Backwards Compatible**: Existing code using `record.id` continues to work
3. **Consistent Behavior**: All backends use the same ID management logic
4. **Explicit Access**: Methods like `get_user_id()` provide unambiguous access
5. **Database Integrity**: Storage operations always use the correct system ID

## Migration Guide

### For Existing Applications

Most existing code requires no changes:
- `record.id` continues to work, with smarter behavior
- Database operations remain unchanged
- Search queries on "id" field work as before

### For New Applications

Best practices:
1. Use `record.storage_id` when you explicitly need the database ID
2. Use `record.get_user_id()` when you need the user-defined ID
3. Use `record.id` for database operations (it intelligently returns the right ID)
4. Check `record.has_storage_id()` to know if a record has been stored

## Technical Details

### Property Setter Handling

The Record class overrides `__setattr__` to properly handle property setters:

```python
def __setattr__(self, name: str, value: Any) -> None:
    # Handle properties with setters specially
    if name in ("id", "storage_id"):
        # Use the property setter
        object.__setattr__(self, name, value)
    # ... handle other attributes
```

This ensures that `record.storage_id = "value"` correctly invokes the property setter rather than creating a field.

### Database Utility Functions

The `database_utils` module provides:

```python
def ensure_record_id(record: Record, record_id: str) -> Record:
    """Ensure a record has its storage ID set."""
    if not record.has_storage_id() or record.storage_id != record_id:
        record = record.copy(deep=True)
        record.storage_id = record_id
    return record
```

This is used internally by backends when processing search results.

## See Also

- [Record Serialization Architecture](RECORD_SERIALIZATION.md) - How records with vector fields are serialized
- [Architecture Overview](ARCHITECTURE.md) - General system architecture
- [API Reference](API_REFERENCE.md) - Complete API documentation