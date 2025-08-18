# Schema Validation

The DataKnobs data package provides a comprehensive schema validation system for ensuring data integrity and consistency across all backends.

## Overview

The validation system enables:

- **Schema definition**: Declarative schema definitions with field types and constraints
- **Data validation**: Validate records against schemas with detailed error reporting
- **Type coercion**: Automatic type conversion with customizable rules
- **Custom validators**: Extend validation with custom business logic
- **Batch validation**: Efficient validation of large datasets
- **Schema evolution**: Support for schema versioning and migration

## Core Components

### Schema Definition

The `Schema` class defines the structure and validation rules for records.

```python
from dataknobs_data.validation import Schema, FieldDefinition
from dataknobs_data.validation.constraints import *

# Define a user schema
user_schema = Schema(
    name="UserSchema",
    version="1.0.0",
    fields={
        "id": FieldDefinition(
            name="id",
            type=str,
            required=True,
            constraints=[
                UniqueConstraint(),
                PatternConstraint(r"^[A-Z0-9]{8}$")
            ]
        ),
        "email": FieldDefinition(
            name="email",
            type=str,
            required=True,
            constraints=[
                EmailConstraint(),
                UniqueConstraint()
            ]
        ),
        "age": FieldDefinition(
            name="age",
            type=int,
            required=False,
            default=0,
            constraints=[
                MinValueConstraint(0),
                MaxValueConstraint(150)
            ]
        ),
        "status": FieldDefinition(
            name="status",
            type=str,
            required=True,
            default="active",
            constraints=[
                EnumConstraint(["active", "inactive", "suspended"])
            ]
        )
    }
)
```

### Field Definitions

Field definitions specify the type, requirements, and constraints for each field.

```python
from dataknobs_data.validation import FieldDefinition
from dataknobs_data.fields import FieldType

# Basic field definition
name_field = FieldDefinition(
    name="name",
    type=str,  # or FieldType.STRING
    required=True,
    description="User's full name"
)

# Field with default value
status_field = FieldDefinition(
    name="status",
    type=str,
    required=False,
    default="pending",
    description="Account status"
)

# Field with multiple constraints
email_field = FieldDefinition(
    name="email",
    type=str,
    required=True,
    constraints=[
        EmailConstraint(),
        UniqueConstraint(scope="global"),
        MaxLengthConstraint(255)
    ]
)

# Nested object field
address_field = FieldDefinition(
    name="address",
    type=dict,
    required=False,
    schema=Schema(  # Nested schema
        name="AddressSchema",
        fields={
            "street": FieldDefinition(name="street", type=str),
            "city": FieldDefinition(name="city", type=str),
            "zip": FieldDefinition(name="zip", type=str, 
                                 constraints=[PatternConstraint(r"^\d{5}$")])
        }
    )
)

# Array field
tags_field = FieldDefinition(
    name="tags",
    type=list,
    required=False,
    default=[],
    item_type=str,  # Type of array items
    constraints=[
        MaxLengthConstraint(10),  # Max 10 tags
        UniqueItemsConstraint()    # No duplicate tags
    ]
)
```

## Built-in Constraints

### Value Constraints

```python
from dataknobs_data.validation.constraints import *

# Numeric constraints
age = FieldDefinition(
    name="age",
    type=int,
    constraints=[
        MinValueConstraint(0),
        MaxValueConstraint(120),
        MultipleOfConstraint(1)  # Must be whole number
    ]
)

# String constraints
username = FieldDefinition(
    name="username",
    type=str,
    constraints=[
        MinLengthConstraint(3),
        MaxLengthConstraint(20),
        PatternConstraint(r"^[a-zA-Z0-9_]+$"),
        NotEmptyConstraint()
    ]
)

# Enum constraint
category = FieldDefinition(
    name="category",
    type=str,
    constraints=[
        EnumConstraint(["electronics", "clothing", "food", "other"])
    ]
)
```

### Format Constraints

```python
# Email validation
email = FieldDefinition(
    name="email",
    type=str,
    constraints=[EmailConstraint()]
)

# URL validation
website = FieldDefinition(
    name="website",
    type=str,
    constraints=[URLConstraint(schemes=["http", "https"])]
)

# Date/time validation
created_at = FieldDefinition(
    name="created_at",
    type=str,
    constraints=[
        DateTimeConstraint(format="%Y-%m-%d %H:%M:%S")
    ]
)

# UUID validation
record_id = FieldDefinition(
    name="id",
    type=str,
    constraints=[UUIDConstraint(version=4)]
)

# JSON validation
metadata = FieldDefinition(
    name="metadata",
    type=str,
    constraints=[JSONConstraint()]
)
```

### Relationship Constraints

```python
# Unique constraint
email = FieldDefinition(
    name="email",
    type=str,
    constraints=[
        UniqueConstraint(scope="database")  # Unique across database
    ]
)

# Reference constraint (foreign key)
user_id = FieldDefinition(
    name="user_id",
    type=str,
    constraints=[
        ReferenceConstraint(
            schema="UserSchema",
            field="id",
            on_delete="cascade"
        )
    ]
)

# Dependency constraint
end_date = FieldDefinition(
    name="end_date",
    type=str,
    constraints=[
        DependencyConstraint(
            depends_on="start_date",
            validator=lambda end, start: end > start
        )
    ]
)
```

## Custom Constraints

Create custom constraints by extending the base `Constraint` class:

```python
from dataknobs_data.validation.constraints import Constraint
from dataknobs_data.validation import ValidationError

class PhoneNumberConstraint(Constraint):
    """Validate phone numbers"""
    
    def __init__(self, country_code="US"):
        self.country_code = country_code
        self.name = "phone_number"
    
    def validate(self, value, field_name=None, record=None):
        if not value:
            return None  # Skip if empty
        
        # Remove non-numeric characters
        digits = ''.join(c for c in value if c.isdigit())
        
        if self.country_code == "US":
            if len(digits) != 10:
                raise ValidationError(
                    field_name,
                    f"Invalid US phone number: must be 10 digits"
                )
            
            # Check valid area code
            if digits[0] in '01':
                raise ValidationError(
                    field_name,
                    f"Invalid area code: cannot start with 0 or 1"
                )
        
        return value

# Use custom constraint
phone = FieldDefinition(
    name="phone",
    type=str,
    constraints=[PhoneNumberConstraint(country_code="US")]
)
```

### Business Logic Validators

```python
class CreditCardConstraint(Constraint):
    """Validate and mask credit card numbers"""
    
    def validate(self, value, field_name=None, record=None):
        if not value:
            return None
        
        # Remove spaces and dashes
        card_number = value.replace(" ", "").replace("-", "")
        
        # Check length
        if len(card_number) not in [13, 14, 15, 16]:
            raise ValidationError(field_name, "Invalid card number length")
        
        # Luhn algorithm validation
        if not self._luhn_check(card_number):
            raise ValidationError(field_name, "Invalid card number")
        
        # Return masked version
        return f"****-****-****-{card_number[-4:]}"
    
    def _luhn_check(self, card_number):
        """Implement Luhn algorithm"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        
        return checksum % 10 == 0
```

## Type Coercion

Automatic type conversion with the `TypeCoercer` class:

```python
from dataknobs_data.validation import TypeCoercer

# Create coercer with custom rules
coercer = TypeCoercer(
    rules={
        (str, int): lambda v: int(v) if v.isdigit() else None,
        (str, float): lambda v: float(v) if v.replace(".", "").isdigit() else None,
        (str, bool): lambda v: v.lower() in ["true", "yes", "1"],
        (int, str): str,
        (float, str): lambda v: f"{v:.2f}",
        (bool, str): lambda v: "true" if v else "false"
    }
)

# Apply coercion
schema = Schema(
    name="ProductSchema",
    fields={
        "price": FieldDefinition(name="price", type=float),
        "quantity": FieldDefinition(name="quantity", type=int),
        "in_stock": FieldDefinition(name="in_stock", type=bool)
    },
    coercer=coercer
)

# Input with wrong types
record = Record(fields={
    "price": Field(name="price", type=FieldType.STRING, value="19.99"),
    "quantity": Field(name="quantity", type=FieldType.STRING, value="5"),
    "in_stock": Field(name="in_stock", type=FieldType.STRING, value="yes")
})

# Validate with coercion
result = schema.validate(record, coerce=True)
# record.fields["price"].value is now 19.99 (float)
# record.fields["quantity"].value is now 5 (int)
# record.fields["in_stock"].value is now True (bool)
```

### Date/Time Coercion

```python
from datetime import datetime

def parse_date(value):
    """Parse various date formats"""
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Cannot parse date: {value}")

# Use in coercer
coercer = TypeCoercer(
    rules={
        (str, datetime): parse_date,
        (datetime, str): lambda d: d.isoformat()
    }
)
```

## Validation Process

### Single Record Validation

```python
from dataknobs_data.validation import SchemaValidator

# Create validator
validator = SchemaValidator(user_schema)

# Validate a record
result = validator.validate(record)

if result.is_valid:
    print("Record is valid")
else:
    print("Validation errors:")
    for error in result.errors:
        print(f"  {error.field}: {error.message}")
```

### Batch Validation

```python
# Validate multiple records
records = database.search(Query())
results = validator.validate_batch(records)

# Process results
valid_records = []
invalid_records = []

for record, result in results:
    if result.is_valid:
        valid_records.append(record)
    else:
        invalid_records.append((record, result.errors))

print(f"Valid: {len(valid_records)}, Invalid: {len(invalid_records)}")
```

### Validation with Fix Attempts

```python
class AutoFixValidator:
    """Validator that attempts to fix common issues"""
    
    def __init__(self, schema):
        self.schema = schema
        self.fixers = {
            "trim": lambda v: v.strip() if isinstance(v, str) else v,
            "lowercase": lambda v: v.lower() if isinstance(v, str) else v,
            "remove_special": lambda v: ''.join(c for c in v if c.isalnum()),
            "default": lambda v, default: default if v is None else v
        }
    
    def validate_and_fix(self, record):
        """Validate and attempt to fix issues"""
        # First validation
        result = self.schema.validate(record)
        
        if result.is_valid:
            return result
        
        # Attempt fixes
        for error in result.errors:
            field_name = error.field
            field = record.fields.get(field_name)
            
            if not field:
                # Add missing field with default
                field_def = self.schema.fields.get(field_name)
                if field_def and field_def.default is not None:
                    record.fields[field_name] = Field(
                        name=field_name,
                        type=field_def.type,
                        value=field_def.default
                    )
            elif "whitespace" in error.message.lower():
                # Trim whitespace
                field.value = self.fixers["trim"](field.value)
            elif "case" in error.message.lower():
                # Fix case
                field.value = self.fixers["lowercase"](field.value)
        
        # Re-validate after fixes
        return self.schema.validate(record)
```

## Schema Registry

Manage multiple schemas with the `SchemaRegistry`:

```python
from dataknobs_data.validation import SchemaRegistry

# Create registry
registry = SchemaRegistry()

# Register schemas
registry.register(user_schema)
registry.register(product_schema)
registry.register(order_schema)

# Get schema by name
schema = registry.get("UserSchema")

# Get schema by version
schema = registry.get("UserSchema", version="1.0.0")

# List all schemas
for name, versions in registry.list_schemas().items():
    print(f"{name}: {', '.join(versions)}")

# Validate using registry
record_type = record.metadata.get("schema_name", "UserSchema")
schema = registry.get(record_type)
result = schema.validate(record)
```

### Schema Inheritance

```python
# Base schema
base_schema = Schema(
    name="BaseEntity",
    fields={
        "id": FieldDefinition(name="id", type=str, required=True),
        "created_at": FieldDefinition(name="created_at", type=datetime),
        "updated_at": FieldDefinition(name="updated_at", type=datetime)
    }
)

# Extend base schema
user_schema = Schema(
    name="UserSchema",
    extends=base_schema,  # Inherit fields from base
    fields={
        "email": FieldDefinition(name="email", type=str, required=True),
        "name": FieldDefinition(name="name", type=str, required=True)
    }
)

# User schema now has: id, created_at, updated_at, email, name
```

## Integration with Backends

### Database-Level Validation

```python
from dataknobs_data.backends.postgres import PostgresDatabase
from dataknobs_data.validation import ValidatingDatabase

# Wrap database with validation
db = ValidatingDatabase(
    backend=PostgresDatabase.from_config(config),
    schema=user_schema,
    validate_on_write=True,  # Validate create/update
    validate_on_read=False,   # Skip validation on read
    coerce_types=True         # Auto-coerce types
)

# All operations now validate automatically
try:
    db.create(invalid_record)
except ValidationError as e:
    print(f"Validation failed: {e.errors}")

# Batch operations with validation
results = db.create_many(records)
print(f"Created: {results.successful}")
print(f"Failed: {results.failed}")
for error in results.errors:
    print(f"  Record {error.record_id}: {error.message}")
```

### Query-Time Validation

```python
# Validate query results
class ValidatingQuery:
    """Add validation to query results"""
    
    def __init__(self, database, schema):
        self.database = database
        self.schema = schema
    
    def search(self, query, validate=True):
        """Search with optional validation"""
        records = self.database.search(query)
        
        if not validate:
            return records
        
        valid_records = []
        for record in records:
            result = self.schema.validate(record)
            if result.is_valid:
                valid_records.append(record)
            else:
                # Log invalid records
                logger.warning(f"Invalid record {record.id}: {result.errors}")
        
        return valid_records
```

## Performance Optimization

### Caching Validation Results

```python
from functools import lru_cache
import hashlib

class CachedValidator:
    """Validator with result caching"""
    
    def __init__(self, schema, cache_size=1000):
        self.schema = schema
        self.cache_size = cache_size
        self._cache = {}
    
    def _record_hash(self, record):
        """Generate hash for record"""
        # Create deterministic hash of record content
        content = json.dumps(record.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def validate(self, record_hash):
        """Cached validation by hash"""
        return self.schema.validate(self._records[record_hash])
    
    def validate_record(self, record):
        """Validate with caching"""
        record_hash = self._record_hash(record)
        
        if record_hash in self._cache:
            return self._cache[record_hash]
        
        result = self.schema.validate(record)
        
        # Cache result
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[record_hash] = result
        return result
```

### Parallel Validation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def validate_parallel(records, schema, max_workers=4):
    """Validate records in parallel"""
    
    def validate_record(record):
        return schema.validate(record)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, validate_record, record)
            for record in records
        ]
        results = await asyncio.gather(*tasks)
    
    return results

# Usage
results = asyncio.run(validate_parallel(records, schema))
```

## Error Reporting

### Detailed Error Messages

```python
from dataknobs_data.validation import ValidationResult, ValidationError

# Custom error formatter
class ErrorFormatter:
    """Format validation errors for display"""
    
    @staticmethod
    def format_errors(result: ValidationResult, format="text"):
        """Format errors in various formats"""
        
        if format == "text":
            lines = []
            for error in result.errors:
                lines.append(f"Field '{error.field}': {error.message}")
                if error.constraint:
                    lines.append(f"  Constraint: {error.constraint}")
                if error.value is not None:
                    lines.append(f"  Value: {error.value}")
            return "\n".join(lines)
        
        elif format == "json":
            return json.dumps([
                {
                    "field": error.field,
                    "message": error.message,
                    "constraint": error.constraint,
                    "value": error.value
                }
                for error in result.errors
            ], indent=2)
        
        elif format == "html":
            html = "<ul class='validation-errors'>"
            for error in result.errors:
                html += f"<li><strong>{error.field}:</strong> {error.message}</li>"
            html += "</ul>"
            return html

# Usage
result = schema.validate(record)
if not result.is_valid:
    print(ErrorFormatter.format_errors(result, format="text"))
```

## Best Practices

1. **Define schemas explicitly**: Always define clear schemas for your data
2. **Use appropriate constraints**: Apply constraints that match business requirements
3. **Validate at boundaries**: Validate data at system boundaries (API, database)
4. **Provide clear error messages**: Make validation errors actionable
5. **Consider performance**: Cache validation results for frequently validated data
6. **Version your schemas**: Use schema versioning for evolution
7. **Test edge cases**: Test validation with boundary values and invalid data
8. **Document constraints**: Document why each constraint exists
9. **Use type coercion carefully**: Be explicit about type conversions
10. **Monitor validation failures**: Track patterns in validation errors

## See Also

- [Migration Utilities](migration.md) - Schema evolution and data migration
- [Record Model](record-model.md) - Understanding the Record structure
- [Field Types](field-types.md) - Available field types
<!-- TODO: Add when tutorial is created:
- [Validation Tutorial](tutorials/validation-tutorial.md) - Step-by-step validation guide
-->