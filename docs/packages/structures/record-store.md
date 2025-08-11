# RecordStore API Documentation

The `RecordStore` class provides a wrapper for managing collections of records in memory and on disk.

## Overview

RecordStore manages records as:

- In-memory list of dictionaries
- Optional pandas DataFrame representation  
- Persistent TSV file storage
- CRUD operations for record management

## Class Definition

```python
from dataknobs_structures import RecordStore
```

## Constructor

```python
RecordStore(tsv_fpath: Optional[str], df: Optional[pd.DataFrame] = None, sep: str = "\t")
```

Creates a new record store.

**Parameters:**
- `tsv_fpath` (str | None): Path to TSV file for persistence (None for memory-only)
- `df` (pd.DataFrame | None): Initial DataFrame to populate the store
- `sep` (str): File separator character (default: tab)

**Example:**
```python
# Memory-only store
memory_store = RecordStore(None)

# File-backed store
file_store = RecordStore("data/records.tsv")

# Store with initial data
import pandas as pd
initial_df = pd.DataFrame([{"id": 1, "name": "John"}])
store = RecordStore("data.tsv", df=initial_df)
```

## Properties

### df
```python
@property
def df(self) -> Optional[pd.DataFrame]
```
Returns the records as a pandas DataFrame.

**Example:**
```python
store = RecordStore(None)
store.add_rec({"name": "Alice", "age": 30})
store.add_rec({"name": "Bob", "age": 25})

df = store.df
print(df.shape)  # (2, 2)
print(df.columns.tolist())  # ['name', 'age']
```

### records
```python
@property
def records(self) -> List[Dict[str, Any]]
```
Returns the records as a list of dictionaries.

**Example:**
```python
store = RecordStore(None)
store.add_rec({"id": 1, "value": "test"})

records = store.records
print(records)  # [{"id": 1, "value": "test"}]
```

## Methods

### add_rec()
```python
def add_rec(self, rec: Dict[str, Any]) -> None
```
Adds a record to the store.

**Parameters:**
- `rec` (Dict[str, Any]): Dictionary representing the record

**Example:**
```python
store = RecordStore(None)

# Add single record
store.add_rec({"id": 1, "name": "Alice", "department": "Engineering"})

# Add multiple records
records = [
    {"id": 2, "name": "Bob", "department": "Sales"},
    {"id": 3, "name": "Carol", "department": "Marketing"}
]

for record in records:
    store.add_rec(record)

print(len(store.records))  # 3
```

### clear()
```python
def clear(self) -> None
```
Removes all records from the store (without auto-saving).

**Example:**
```python
store = RecordStore(None)
store.add_rec({"id": 1, "name": "Test"})
print(len(store.records))  # 1

store.clear()
print(len(store.records))  # 0
```

### save()
```python
def save(self) -> None
```
Saves the records to disk as a TSV file (if file path was provided).

**Example:**
```python
store = RecordStore("data/employees.tsv")
store.add_rec({"id": 1, "name": "Alice", "salary": 50000})
store.add_rec({"id": 2, "name": "Bob", "salary": 60000})

# Save to disk
store.save()  # Creates/updates employees.tsv
```

### restore()
```python
def restore(self, df: Optional[pd.DataFrame] = None) -> None
```
Restores records from disk or provided DataFrame, discarding changes.

**Parameters:**
- `df` (pd.DataFrame | None): DataFrame to restore from (optional)

**Example:**
```python
store = RecordStore("data/backup.tsv")
store.add_rec({"id": 1, "temp": "data"})  # Temporary change

# Restore from file (loses temporary data)
store.restore()

# Or restore from specific DataFrame
backup_df = pd.DataFrame([{"id": 1, "name": "Restored"}])
store.restore(df=backup_df)
```

## Usage Examples

### Basic CRUD Operations

```python
from dataknobs_structures import RecordStore

# Create store
store = RecordStore("data/users.tsv")

# Create records
users = [
    {"id": 1, "username": "alice", "email": "alice@example.com", "active": True},
    {"id": 2, "username": "bob", "email": "bob@example.com", "active": False},
    {"id": 3, "username": "carol", "email": "carol@example.com", "active": True}
]

for user in users:
    store.add_rec(user)

# Read records
all_users = store.records
print(f"Total users: {len(all_users)}")

# Filter active users
active_users = [user for user in store.records if user.get("active")]
print(f"Active users: {len(active_users)}")

# Update record (by replacing)
updated_users = []
for user in store.records:
    if user["id"] == 2:
        user["active"] = True  # Update Bob's status
    updated_users.append(user)

# Clear and re-add updated records
store.clear()
for user in updated_users:
    store.add_rec(user)

# Delete record (by filtering)
store.clear()
for user in users:
    if user["id"] != 3:  # Remove Carol
        store.add_rec(user)

# Save changes
store.save()
```

### Working with Pandas DataFrame

```python
from dataknobs_structures import RecordStore
import pandas as pd

# Create store with data
store = RecordStore("data/sales.tsv")

# Add sales records
sales_data = [
    {"date": "2024-01-01", "product": "Widget A", "amount": 100.50, "quantity": 2},
    {"date": "2024-01-02", "product": "Widget B", "amount": 250.00, "quantity": 5},
    {"date": "2024-01-03", "product": "Widget A", "amount": 75.25, "quantity": 1}
]

for sale in sales_data:
    store.add_rec(sale)

# Get DataFrame for analysis
df = store.df

# Perform pandas operations
total_sales = df["amount"].sum()
avg_quantity = df["quantity"].mean()
product_counts = df["product"].value_counts()

print(f"Total sales: ${total_sales}")
print(f"Average quantity: {avg_quantity}")
print("Product distribution:")
print(product_counts)

# Add computed columns
df["unit_price"] = df["amount"] / df["quantity"]

# Convert back to records (if needed)
updated_records = df.to_dict("records")
store.clear()
for record in updated_records:
    store.add_rec(record)
```

### Batch Processing

```python
from dataknobs_structures import RecordStore
import json

def process_json_file(json_file_path, store):
    """Process a JSON file and add records to store"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        for item in data:
            store.add_rec(item)
    else:
        store.add_rec(data)

# Create store
store = RecordStore("data/processed.tsv")

# Process multiple files
json_files = ["data1.json", "data2.json", "data3.json"]
for file_path in json_files:
    try:
        process_json_file(file_path, store)
        print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"Total records processed: {len(store.records)}")
store.save()
```

### Data Validation and Cleaning

```python
from dataknobs_structures import RecordStore

def validate_record(record):
    """Validate a record before adding to store"""
    required_fields = ["id", "name", "email"]
    
    # Check required fields
    for field in required_fields:
        if field not in record or not record[field]:
            return False, f"Missing required field: {field}"
    
    # Validate email format (simple check)
    email = record["email"]
    if "@" not in email:
        return False, "Invalid email format"
    
    return True, None

# Create store with validation
store = RecordStore("data/clean_users.tsv")
invalid_records = []

# Sample data with some invalid entries
raw_data = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "", "email": "bob@example.com"},  # Invalid: empty name
    {"id": 3, "name": "Carol", "email": "invalid-email"},  # Invalid: bad email
    {"id": 4, "name": "David", "email": "david@example.com"}
]

# Process with validation
for record in raw_data:
    is_valid, error_msg = validate_record(record)
    if is_valid:
        store.add_rec(record)
        print(f"Added record: {record['name']}")
    else:
        invalid_records.append((record, error_msg))
        print(f"Rejected record {record.get('id', 'unknown')}: {error_msg}")

print(f"\nValid records: {len(store.records)}")
print(f"Invalid records: {len(invalid_records)}")
```

### Memory vs File-backed Storage

```python
from dataknobs_structures import RecordStore
import os

# Memory-only store (fast, not persistent)
memory_store = RecordStore(None)
memory_store.add_rec({"temp": "data"})
print("Memory store created")

# File-backed store (persistent)
file_store = RecordStore("data/persistent.tsv")
file_store.add_rec({"persistent": "data"})
file_store.save()
print("File store saved")

# Verify persistence
new_store = RecordStore("data/persistent.tsv")  # Loads existing data
print(f"Loaded {len(new_store.records)} records from file")

# Clean up
if os.path.exists("data/persistent.tsv"):
    os.remove("data/persistent.tsv")
```

### Custom Separators

```python
from dataknobs_structures import RecordStore

# CSV format (comma-separated)
csv_store = RecordStore("data/records.csv", sep=",")
csv_store.add_rec({"name": "Alice", "city": "New York"})
csv_store.save()

# Pipe-separated
pipe_store = RecordStore("data/records.psv", sep="|")  
pipe_store.add_rec({"name": "Bob", "city": "Los Angeles"})
pipe_store.save()

print("Records saved with different separators")
```

## Integration Examples

### With JSON Processing

```python
from dataknobs_structures import RecordStore
# from dataknobs_utils import json_utils  # Example integration

# Create store from JSON data
json_data = [
    {"id": 1, "product": "Widget", "price": 19.99},
    {"id": 2, "product": "Gadget", "price": 29.99}
]

store = RecordStore("data/products.tsv")
for item in json_data:
    store.add_rec(item)

# Could export back to JSON using utils
# json_output = json_utils.to_json(store.records)
```

### With Tree Structures

```python
from dataknobs_structures import RecordStore, Tree

# Store tree node data in RecordStore
tree = Tree({"id": 1, "name": "root", "type": "folder"})
child1 = tree.add_child({"id": 2, "name": "child1", "type": "file"})
child2 = tree.add_child({"id": 3, "name": "child2", "type": "file"})

# Flatten tree to records
store = RecordStore("data/tree_nodes.tsv")

def add_tree_to_store(node, parent_id=None):
    record = node.data.copy()
    record["parent_id"] = parent_id
    store.add_rec(record)
    
    for child in (node.children or []):
        add_tree_to_store(child, record["id"])

add_tree_to_store(tree)
print(f"Stored {len(store.records)} tree nodes")
```

## Error Handling

```python
from dataknobs_structures import RecordStore
import os

def safe_record_store_operations():
    """Demonstrate error handling with RecordStore"""
    
    try:
        # Attempt to create store with invalid path
        store = RecordStore("/invalid/path/file.tsv")
        store.add_rec({"test": "data"})
        
        # This might fail due to permissions/path issues
        store.save()
        
    except PermissionError:
        print("Permission denied - cannot write to file")
        # Fallback to memory-only store
        store = RecordStore(None)
        store.add_rec({"test": "data"})
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
        
    return store

# Use safe operations
store = safe_record_store_operations()
if store:
    print(f"Store created with {len(store.records)} records")
```

## Performance Considerations

- **Memory Usage**: Records are kept in memory; consider memory limits for large datasets
- **File I/O**: Save operations write entire file; frequent saves may be slow
- **DataFrame Conversion**: Creating DataFrame from records has overhead for large datasets
- **Batch Operations**: Add multiple records before saving for better performance

## Best Practices

1. **Choose Appropriate Storage**: Use file-backed stores for persistent data
2. **Batch Additions**: Add multiple records before saving to reduce I/O
3. **Validate Data**: Implement validation before adding records
4. **Handle Errors**: Wrap file operations in try-except blocks
5. **Memory Management**: Clear store periodically for long-running processes
6. **Consistent Schema**: Maintain consistent field names across records

## Limitations

- All records are loaded into memory
- No built-in indexing or query optimization
- Limited to flat record structures (no nested objects)
- File format is fixed (TSV/CSV)
- No built-in data type preservation in files

## See Also

- [Document API](document.md) - For text documents with metadata
- [Tree API](tree.md) - For hierarchical data structures
- [Utils Package](../utils/index.md) - For additional data processing utilities