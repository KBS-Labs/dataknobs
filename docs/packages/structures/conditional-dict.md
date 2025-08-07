# Conditional Dictionary API Documentation

The `cdict` class provides a dictionary that conditionally accepts key-value pairs based on custom validation logic.

## Overview

The `cdict` (conditional dictionary) implements the strategy pattern to:

- Accept or reject key-value pairs based on custom logic
- Store rejected pairs separately for inspection
- Maintain full dictionary interface compatibility
- Provide flexible validation strategies

## Class Definition

```python
from dataknobs_structures import cdict
```

## Constructor

```python
cdict(accept_fn: Callable[[Dict, Any, Any], bool], *args, **kwargs)
```

Creates a new conditional dictionary.

**Parameters:**
- `accept_fn` (Callable): Function that takes (dict, key, value) and returns True to accept or False to reject
- `*args`: Additional positional arguments for dictionary initialization
- `**kwargs`: Additional keyword arguments for dictionary initialization

**Example:**
```python
def accept_positive(d, key, value):
    """Only accept positive numeric values"""
    return isinstance(value, (int, float)) and value > 0

cd = cdict(accept_positive)
cd["good"] = 10    # Accepted
cd["bad"] = -5     # Rejected
```

## Properties

### rejected
```python
@property
def rejected(self) -> Dict
```
Returns a dictionary of rejected key-value pairs.

**Example:**
```python
def accept_strings(d, key, value):
    return isinstance(value, str)

cd = cdict(accept_strings)
cd["name"] = "Alice"  # Accepted
cd["age"] = 30        # Rejected

print(cd)            # {"name": "Alice"}
print(cd.rejected)   # {"age": 30}
```

## Methods

### Standard Dictionary Methods

The `cdict` inherits from `dict` and supports all standard dictionary operations, with conditional acceptance applied to modifications.

#### \_\_setitem\_\_()
```python
def __setitem__(self, key: Any, value: Any) -> None
```
Sets a key-value pair if accepted by the validation function.

**Example:**
```python
def accept_even(d, key, value):
    return isinstance(value, int) and value % 2 == 0

cd = cdict(accept_even)
cd["two"] = 2    # Accepted (even)
cd["three"] = 3  # Rejected (odd)

print(cd)         # {"two": 2}
print(cd.rejected) # {"three": 3}
```

#### setdefault()
```python
def setdefault(self, key: Any, default: Any = None) -> Any
```
Sets key to default if key not present and default is accepted.

**Example:**
```python
def accept_letters(d, key, value):
    return isinstance(value, str) and value.isalpha()

cd = cdict(accept_letters, {"existing": "word"})

# Key exists - returns existing value
result1 = cd.setdefault("existing", "new")
print(result1)  # "word"

# Key doesn't exist, default accepted
result2 = cd.setdefault("new_key", "hello")
print(result2)  # "hello"

# Key doesn't exist, default rejected
result3 = cd.setdefault("bad_key", "123")
print(result3)  # None (rejected)
print(cd.rejected)  # {"bad_key": "123"}
```

#### update()
```python
def update(self, *args, **kwargs) -> None
```
Updates the dictionary with key-value pairs, applying conditional acceptance.

**Example:**
```python
def accept_positive(d, key, value):
    return isinstance(value, (int, float)) and value > 0

cd = cdict(accept_positive)

# Update with dictionary
cd.update({"a": 1, "b": -2, "c": 3})
print(cd)         # {"a": 1, "c": 3}
print(cd.rejected) # {"b": -2}

# Update with keyword arguments
cd.update(d=4, e=-5)
print(cd)         # {"a": 1, "c": 3, "d": 4}
print(cd.rejected) # {"b": -2, "e": -5}
```

## Usage Examples

### Basic Validation

```python
from dataknobs_structures import cdict

# Accept only string values
def accept_strings(d, key, value):
    return isinstance(value, str)

string_dict = cdict(accept_strings)
string_dict["name"] = "Alice"      # ✓ Accepted
string_dict["age"] = 30            # ✗ Rejected
string_dict["city"] = "New York"   # ✓ Accepted

print(string_dict)          # {"name": "Alice", "city": "New York"}
print(string_dict.rejected) # {"age": 30}
```

### Range Validation

```python
from dataknobs_structures import cdict

# Accept values in range [0, 100]
def accept_percentage(d, key, value):
    return isinstance(value, (int, float)) and 0 <= value <= 100

scores = cdict(accept_percentage)
scores["math"] = 85      # ✓ Accepted
scores["science"] = 92   # ✓ Accepted  
scores["english"] = 150  # ✗ Rejected (> 100)
scores["history"] = -10  # ✗ Rejected (< 0)

print(scores)          # {"math": 85, "science": 92}
print(scores.rejected) # {"english": 150, "history": -10}
```

### Key-Based Validation

```python
from dataknobs_structures import cdict

# Accept only if key starts with 'valid_'
def accept_valid_keys(d, key, value):
    return isinstance(key, str) and key.startswith("valid_")

prefixed_dict = cdict(accept_valid_keys)
prefixed_dict["valid_item"] = "accepted"    # ✓ Accepted
prefixed_dict["invalid_item"] = "rejected"  # ✗ Rejected
prefixed_dict["valid_data"] = 42            # ✓ Accepted

print(prefixed_dict)          # {"valid_item": "accepted", "valid_data": 42}
print(prefixed_dict.rejected) # {"invalid_item": "rejected"}
```

### Context-Aware Validation

```python
from dataknobs_structures import cdict

# Accept numbers that maintain ascending order
def accept_ascending(d, key, value):
    if not isinstance(value, (int, float)):
        return False
    
    # Allow if dictionary is empty
    if not d:
        return True
        
    # Check if value is greater than all existing values
    return all(value > existing_val for existing_val in d.values())

ascending_dict = cdict(accept_ascending)
ascending_dict["first"] = 10   # ✓ Accepted (first value)
ascending_dict["second"] = 20  # ✓ Accepted (20 > 10)
ascending_dict["third"] = 15   # ✗ Rejected (15 < 20)
ascending_dict["fourth"] = 25  # ✓ Accepted (25 > 20)

print(ascending_dict)          # {"first": 10, "second": 20, "fourth": 25}
print(ascending_dict.rejected) # {"third": 15}
```

### Complex Validation Logic

```python
from dataknobs_structures import cdict

# Accept user records with validation rules
def accept_user(d, key, value):
    # Must be a dictionary
    if not isinstance(value, dict):
        return False
    
    # Must have required fields
    required_fields = ["name", "email", "age"]
    if not all(field in value for field in required_fields):
        return False
        
    # Name must be non-empty string
    if not isinstance(value["name"], str) or not value["name"].strip():
        return False
        
    # Email must contain @
    if "@" not in value["email"]:
        return False
        
    # Age must be positive integer
    if not isinstance(value["age"], int) or value["age"] <= 0:
        return False
        
    return True

users = cdict(accept_user)

# Valid user
users["user1"] = {
    "name": "Alice Smith",
    "email": "alice@example.com", 
    "age": 30
}  # ✓ Accepted

# Invalid user - missing email
users["user2"] = {
    "name": "Bob Jones",
    "age": 25
}  # ✗ Rejected

# Invalid user - bad email
users["user3"] = {
    "name": "Carol White",
    "email": "invalid-email",
    "age": 35
}  # ✗ Rejected

print(f"Accepted users: {len(users)}")      # 1
print(f"Rejected users: {len(users.rejected)}") # 2
```

### Data Type Filtering

```python
from dataknobs_structures import cdict

# Create different dictionaries for different data types
def create_type_filter(accepted_type):
    def accept_type(d, key, value):
        return isinstance(value, accepted_type)
    return accept_type

# Separate containers for different types
strings = cdict(create_type_filter(str))
numbers = cdict(create_type_filter((int, float)))
lists = cdict(create_type_filter(list))

# Mixed input data
mixed_data = {
    "name": "Alice",
    "age": 30,
    "height": 5.6,
    "hobbies": ["reading", "hiking"],
    "city": "New York",
    "scores": [85, 90, 78]
}

# Distribute data to appropriate containers
for key, value in mixed_data.items():
    strings[key] = value
    numbers[key] = value
    lists[key] = value

print("Strings:", dict(strings))    # {"name": "Alice", "city": "New York"}
print("Numbers:", dict(numbers))    # {"age": 30, "height": 5.6}
print("Lists:", dict(lists))        # {"hobbies": [...], "scores": [...]}
```

### Configuration Validation

```python
from dataknobs_structures import cdict

# Configuration validator
def validate_config(d, key, value):
    """Validate configuration settings"""
    valid_configs = {
        "max_connections": (int, lambda x: 1 <= x <= 1000),
        "timeout": (float, lambda x: 0.1 <= x <= 60.0),
        "debug": (bool, lambda x: True),  # Always valid for bool
        "log_level": (str, lambda x: x.upper() in ["DEBUG", "INFO", "WARN", "ERROR"])
    }
    
    if key not in valid_configs:
        return False
        
    expected_type, validator = valid_configs[key]
    
    # Check type
    if not isinstance(value, expected_type):
        return False
        
    # Check value constraints
    return validator(value)

config = cdict(validate_config)

# Valid configurations
config["max_connections"] = 100   # ✓
config["timeout"] = 5.0           # ✓
config["debug"] = True            # ✓
config["log_level"] = "INFO"      # ✓

# Invalid configurations
config["max_connections"] = 2000  # ✗ Out of range
config["timeout"] = -1            # ✗ Negative
config["invalid_key"] = "value"   # ✗ Unknown key
config["log_level"] = "TRACE"     # ✗ Invalid level

print("Valid config:", dict(config))
print("Invalid config:", config.rejected)
```

## Integration Examples

### With JSON Loading

```python
from dataknobs_structures import cdict
import json

def accept_valid_json_field(d, key, value):
    """Accept common JSON field types"""
    return isinstance(value, (str, int, float, bool, list, dict, type(None)))

# Load and validate JSON data
json_data = {
    "name": "Alice",
    "age": 30,
    "active": True,
    "scores": [85, 90],
    "metadata": {"created": "2024-01-01"},
    "invalid": object()  # This will be rejected
}

validated_data = cdict(accept_valid_json_field)
for key, value in json_data.items():
    validated_data[key] = value

# Can safely serialize accepted data
clean_json = json.dumps(dict(validated_data))
print(f"Clean JSON: {clean_json}")
print(f"Rejected: {validated_data.rejected}")
```

### With Data Processing Pipeline

```python
from dataknobs_structures import cdict

# Multi-stage validation pipeline
def create_pipeline_validator(stage_name):
    validators = {
        "input": lambda d, k, v: isinstance(v, (int, float, str)),
        "processed": lambda d, k, v: isinstance(v, (int, float)) and v >= 0,
        "output": lambda d, k, v: isinstance(v, float) and 0 <= v <= 1
    }
    return validators.get(stage_name, lambda d, k, v: False)

# Create pipeline stages
input_stage = cdict(create_pipeline_validator("input"))
processed_stage = cdict(create_pipeline_validator("processed"))
output_stage = cdict(create_pipeline_validator("output"))

# Process data through pipeline
raw_data = {"a": 10, "b": -5, "c": "invalid", "d": 3.5}

# Stage 1: Input validation
for key, value in raw_data.items():
    input_stage[key] = value

# Stage 2: Process and validate
for key, value in input_stage.items():
    if isinstance(value, str):
        continue  # Skip strings
    processed_value = abs(value)  # Example processing
    processed_stage[key] = processed_value

# Stage 3: Normalize and validate
for key, value in processed_stage.items():
    normalized = value / 100.0  # Example normalization
    output_stage[key] = normalized

print("Pipeline results:")
print(f"Input: {dict(input_stage)}")
print(f"Processed: {dict(processed_stage)}")  
print(f"Output: {dict(output_stage)}")
```

## Error Handling

```python
from dataknobs_structures import cdict

def safe_validator(d, key, value):
    """Validator with error handling"""
    try:
        # Complex validation logic that might fail
        if hasattr(value, '__len__'):
            return len(value) > 0
        else:
            return value is not None
    except Exception as e:
        print(f"Validation error for {key}={value}: {e}")
        return False  # Reject on error

safe_dict = cdict(safe_validator)
safe_dict["valid"] = "hello"      # ✓
safe_dict["empty"] = ""           # ✗ 
safe_dict["none"] = None          # ✗
safe_dict["number"] = 42          # ✓ (has no __len__ but is not None)
```

## Performance Considerations

- Validation function is called for every assignment
- Keep validation logic simple for better performance  
- Consider caching validation results for expensive checks
- Rejected items are stored in memory alongside accepted items

## Best Practices

1. **Clear Validation Logic**: Make acceptance criteria obvious and well-documented
2. **Fail Fast**: Reject obviously invalid data early in validation
3. **Consistent Return Types**: Always return boolean from acceptance function
4. **Error Handling**: Handle exceptions in validation functions gracefully
5. **Monitor Rejections**: Check `rejected` property to understand what's being filtered

## Limitations

- No built-in validation for existing dictionary items when acceptance function changes
- Rejected items consume memory until explicitly cleared
- No automatic conversion or transformation of rejected items
- Single validation function per dictionary (no composition)

## See Also

- [Tree API](tree.md) - For hierarchical data with validation
- [Document API](document.md) - For validated text documents
- [Record Store API](record-store.md) - For collections with validation