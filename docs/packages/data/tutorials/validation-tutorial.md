# Data Validation Tutorial

This tutorial will guide you through the DataKnobs validation system, from basic field validation to complex business rules and data quality checks.

## Prerequisites

```python
from dataknobs_data.validation import (
    Schema, Field, FieldType,
    Required, Range, Length, Pattern, Enum, Unique, Custom,
    ValidationResult, Coercer
)
from dataknobs_data import Record, MemoryDatabase
from datetime import datetime
import re
```

## Part 1: Basic Schema Definition

### Creating Your First Schema

Let's start by defining a schema for user registration:

```python
# Simple schema definition
user_schema = Schema("UserRegistration")

# Add fields with types
user_schema.field("username", "STRING", required=True)
user_schema.field("email", "STRING", required=True)
user_schema.field("age", "INTEGER", required=False)
user_schema.field("terms_accepted", "BOOLEAN", required=True)

# Create a test record
test_record = Record(data={
    "username": "john_doe",
    "email": "john@example.com",
    "age": 25,
    "terms_accepted": True
})

# Validate the record
result = user_schema.validate(test_record)
if result.valid:
    print("✓ Validation passed!")
    print(f"Validated data: {result.value.data}")
else:
    print("✗ Validation failed!")
    print(f"Errors: {result.errors}")
```

### Fluent API for Schema Building

Use the fluent API for more readable schema definitions:

```python
# Build schema using method chaining
product_schema = (Schema("Product", strict=True)
    .field("sku", "STRING", required=True)
    .field("name", "STRING", required=True)
    .field("price", "FLOAT", required=True)
    .field("quantity", "INTEGER", default=0)
    .field("category", "STRING", required=True)
    .field("tags", "LIST", default=[])
    .field("metadata", "DICT", required=False)
)

# Test with sample data
product = Record(data={
    "sku": "PROD-001",
    "name": "Laptop",
    "price": 999.99,
    "category": "Electronics"
})

result = product_schema.validate(product, coerce=True)
print(f"Valid: {result.valid}")
print(f"Data with defaults: {result.value.data}")
# Note: quantity will be set to 0, tags to []
```

## Part 2: Constraint-Based Validation

### Built-in Constraints

Add constraints to enforce business rules:

```python
# Schema with constraints
account_schema = (Schema("Account")
    .field("username", "STRING", 
           required=True,
           constraints=[
               Length(min=3, max=20),
               Pattern(r"^[a-zA-Z0-9_]+$", "Username can only contain letters, numbers, and underscores")
           ])
    .field("email", "STRING",
           required=True,
           constraints=[
               Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", "Invalid email format")
           ])
    .field("password", "STRING",
           required=True,
           constraints=[
               Length(min=8, max=128),
               Pattern(r".*[A-Z].*", "Password must contain at least one uppercase letter"),
               Pattern(r".*[a-z].*", "Password must contain at least one lowercase letter"),
               Pattern(r".*\d.*", "Password must contain at least one number"),
               Pattern(r".*[@$!%*?&].*", "Password must contain at least one special character")
           ])
    .field("age", "INTEGER",
           constraints=[
               Range(min=13, max=120, message="Age must be between 13 and 120")
           ])
    .field("account_type", "STRING",
           constraints=[
               Enum(["free", "premium", "enterprise"], message="Invalid account type")
           ])
)

# Test various inputs
test_cases = [
    {"username": "ab", "email": "invalid", "password": "weak", "age": 5, "account_type": "super"},
    {"username": "john_doe", "email": "john@example.com", "password": "Str0ng!Pass", "age": 25, "account_type": "premium"}
]

for i, data in enumerate(test_cases):
    record = Record(data=data)
    result = account_schema.validate(record)
    print(f"\nTest case {i + 1}:")
    if result.valid:
        print("✓ Valid")
    else:
        print("✗ Invalid")
        for error in result.errors:
            print(f"  - {error}")
```

### Custom Constraints

Create custom validation logic for complex business rules:

```python
# Custom constraint for credit card validation
def luhn_check(card_number):
    """Validate credit card using Luhn algorithm"""
    card_number = str(card_number).replace(" ", "").replace("-", "")
    if not card_number.isdigit():
        return False
    
    total = 0
    reverse = card_number[::-1]
    for i, digit in enumerate(reverse):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0

# Custom constraint for business hours
def business_hours_check(time_str):
    """Check if time is within business hours (9 AM - 5 PM)"""
    try:
        hour = datetime.strptime(time_str, "%H:%M").hour
        return 9 <= hour < 17
    except:
        return False

# Schema with custom constraints
payment_schema = (Schema("Payment")
    .field("card_number", "STRING",
           required=True,
           constraints=[
               Custom(luhn_check, "Invalid credit card number")
           ])
    .field("amount", "FLOAT",
           required=True,
           constraints=[
               Range(min=0.01, max=10000, message="Amount must be between $0.01 and $10,000")
           ])
    .field("appointment_time", "STRING",
           constraints=[
               Custom(business_hours_check, "Appointment must be during business hours (9 AM - 5 PM)")
           ])
)

# Test custom validation
test_payment = Record(data={
    "card_number": "4532015112830366",  # Valid Visa test number
    "amount": 99.99,
    "appointment_time": "14:30"
})

result = payment_schema.validate(test_payment)
print(f"Payment validation: {'✓ Valid' if result.valid else '✗ Invalid'}")
```

## Part 3: Advanced Constraint Composition

### Logical Operators

Combine constraints using logical operators:

```python
from dataknobs_data.validation import Required, Range, Enum, Pattern

# OR operator: value must satisfy at least one constraint
price_constraint = Range(min=0, max=100) | Range(min=1000, max=10000)
# Accepts: 50 (first range) or 5000 (second range)
# Rejects: 500 (neither range)

# AND operator: value must satisfy all constraints
username_constraint = Length(min=3, max=20) & Pattern(r"^[a-z][a-z0-9_]*$")
# Must be 3-20 chars AND start with lowercase letter

# NOT operator: value must not satisfy the constraint
not_admin = ~Enum(["admin", "root", "superuser"])
# Rejects: "admin", "root", "superuser"
# Accepts: any other value

# Complex composition
password_constraint = (
    Length(min=8) & 
    (Pattern(r".*[A-Z].*") | Pattern(r".*[!@#$%^&*].*"))
)
# At least 8 chars AND (contains uppercase OR special char)

# Use in schema
flexible_schema = (Schema("FlexibleProduct")
    .field("price", "FLOAT", constraints=[price_constraint])
    .field("username", "STRING", constraints=[username_constraint])
    .field("role", "STRING", constraints=[not_admin])
    .field("password", "STRING", constraints=[password_constraint])
)

# Test the composed constraints
test_data = Record(data={
    "price": 75.0,  # Valid: in first range
    "username": "user123",  # Valid: starts with 'u', alphanumeric
    "role": "user",  # Valid: not in admin list
    "password": "Secret123"  # Valid: 8+ chars with uppercase
})

result = flexible_schema.validate(test_data)
print(f"Composed validation: {result}")
```

## Part 4: Data Coercion

### Automatic Type Coercion

Convert data to the correct types during validation:

```python
# Schema expecting specific types
order_schema = (Schema("Order", strict=False)  # strict=False allows coercion
    .field("order_id", "INTEGER", required=True)
    .field("quantity", "INTEGER", required=True)
    .field("price", "FLOAT", required=True)
    .field("is_express", "BOOLEAN", default=False)
    .field("order_date", "STRING", required=True)
)

# Input with wrong types (e.g., from JSON/form data)
raw_input = Record(data={
    "order_id": "12345",  # String instead of int
    "quantity": "3",      # String instead of int
    "price": "29.99",     # String instead of float
    "is_express": "true", # String instead of bool
    "order_date": "2024-01-15"
})

# Validate with coercion
result = order_schema.validate(raw_input, coerce=True)
if result.valid:
    print("✓ Coercion successful!")
    print(f"Coerced types: {result.value.data}")
    # All values now have correct types
else:
    print(f"✗ Coercion failed: {result.errors}")
```

### Manual Coercion

Use the Coercer class for explicit type conversion:

```python
coercer = Coercer()

# Individual coercion
test_values = [
    ("123", int),      # "123" -> 123
    ("45.67", float),  # "45.67" -> 45.67
    ("true", bool),    # "true" -> True
    ("yes", bool),     # "yes" -> True
    ("1", bool),       # "1" -> True
    ("false", bool),   # "false" -> False
    ("no", bool),      # "no" -> False
    ("0", bool),       # "0" -> False
]

for value, target_type in test_values:
    result = coercer.coerce(value, target_type)
    if result.success:
        print(f"✓ '{value}' -> {result.value} ({type(result.value).__name__})")
    else:
        print(f"✗ Failed to coerce '{value}': {result.error}")

# Batch coercion
raw_data = {
    "user_id": "42",
    "score": "95.5",
    "active": "yes",
    "tags": "python,data,validation"  # Will remain string
}

type_map = {
    "user_id": int,
    "score": float,
    "active": bool,
    "tags": str
}

coerced = coercer.coerce_many(raw_data, type_map)
print(f"\nBatch coercion result: {coerced}")
```

## Part 5: Unique Constraints and Context

### Validating Uniqueness Across Records

Ensure field values are unique across multiple records:

```python
# Schema with unique constraint
user_database_schema = (Schema("UserDatabase")
    .field("email", "STRING", 
           required=True,
           constraints=[
               Pattern(r"^[\w\.-]+@[\w\.-]+\.\w+$"),
               Unique("email")  # Email must be unique
           ])
    .field("username", "STRING",
           required=True,
           constraints=[
               Length(min=3, max=20),
               Unique("username")  # Username must be unique
           ])
    .field("employee_id", "INTEGER",
           constraints=[
               Unique("employee_id")  # Employee ID must be unique
           ])
)

# Test records with potential duplicates
test_records = [
    Record(data={"email": "john@example.com", "username": "john", "employee_id": 1001}),
    Record(data={"email": "jane@example.com", "username": "jane", "employee_id": 1002}),
    Record(data={"email": "john@example.com", "username": "john2", "employee_id": 1003}),  # Duplicate email
    Record(data={"email": "bob@example.com", "username": "jane", "employee_id": 1004}),    # Duplicate username
]

# Validate with shared context for uniqueness checking
results = user_database_schema.validate_many(test_records)

for i, result in enumerate(results):
    print(f"\nRecord {i + 1}:")
    if result.valid:
        print("✓ Valid and unique")
    else:
        print("✗ Validation failed:")
        for error in result.errors:
            print(f"  - {error}")
```

## Part 6: Schema Evolution and Migration

### Validating During Migration

Combine validation with data migration:

```python
from dataknobs_data.migration import Transformer

# Old schema (v1)
old_schema = (Schema("UserV1")
    .field("name", "STRING", required=True)
    .field("email", "STRING", required=True)
)

# New schema (v2) with additional requirements
new_schema = (Schema("UserV2")
    .field("first_name", "STRING", required=True, constraints=[Length(min=1, max=50)])
    .field("last_name", "STRING", required=True, constraints=[Length(min=1, max=50)])
    .field("email", "STRING", required=True, 
           constraints=[Pattern(r"^[\w\.-]+@[\w\.-]+\.\w+$")])
    .field("status", "STRING", required=True,
           constraints=[Enum(["active", "inactive", "pending"])])
    .field("created_at", "STRING", required=True)
)

# Transform old records to new format
def split_name(full_name):
    parts = full_name.split(" ", 1)
    return {
        "first_name": parts[0] if parts else "",
        "last_name": parts[1] if len(parts) > 1 else ""
    }

# Migration transformer
transformer = Transformer()
transformer.exclude("name")  # Remove old field

# Add transformation function
def transform_record(record):
    names = split_name(record.get_value("name", ""))
    return Record(data={
        **record.data,
        **names,
        "status": "active",
        "created_at": datetime.now().isoformat()
    })

# Migrate and validate
old_records = [
    Record(data={"name": "John Doe", "email": "john@example.com"}),
    Record(data={"name": "Jane", "email": "invalid-email"}),  # Will fail validation
]

migrated_records = []
for old_record in old_records:
    # Transform
    new_record = transform_record(old_record)
    
    # Validate against new schema
    result = new_schema.validate(new_record)
    
    if result.valid:
        migrated_records.append(result.value)
        print(f"✓ Migrated: {result.value.data}")
    else:
        print(f"✗ Migration failed for {old_record.data}: {result.errors}")
```

## Part 7: Practical Examples

### Example 1: E-commerce Order Validation

```python
# Complex e-commerce order validation
def validate_shipping_address(address):
    """Validate shipping address completeness"""
    required_fields = ["street", "city", "state", "zip"]
    if not isinstance(address, dict):
        return False
    return all(field in address and address[field] for field in required_fields)

def validate_payment_method(payment):
    """Validate payment method data"""
    if not isinstance(payment, dict) or "type" not in payment:
        return False
    
    if payment["type"] == "credit_card":
        return "card_number" in payment and "cvv" in payment
    elif payment["type"] == "paypal":
        return "email" in payment
    return False

order_validation_schema = (Schema("Order", strict=False)
    .field("customer_id", "INTEGER", required=True)
    .field("items", "LIST", required=True,
           constraints=[
               Length(min=1, message="Order must contain at least one item")
           ])
    .field("subtotal", "FLOAT", required=True,
           constraints=[
               Range(min=0.01, message="Subtotal must be positive")
           ])
    .field("tax", "FLOAT", default=0.0,
           constraints=[
               Range(min=0, message="Tax cannot be negative")
           ])
    .field("shipping", "FLOAT", default=0.0,
           constraints=[
               Range(min=0, message="Shipping cannot be negative")
           ])
    .field("total", "FLOAT", required=True)
    .field("shipping_address", "DICT", required=True,
           constraints=[
               Custom(validate_shipping_address, "Invalid shipping address")
           ])
    .field("payment_method", "DICT", required=True,
           constraints=[
               Custom(validate_payment_method, "Invalid payment method")
           ])
    .field("status", "STRING", default="pending",
           constraints=[
               Enum(["pending", "processing", "shipped", "delivered", "cancelled"])
           ])
)

# Test order validation
test_order = Record(data={
    "customer_id": 12345,
    "items": [
        {"product_id": 1, "quantity": 2, "price": 29.99},
        {"product_id": 2, "quantity": 1, "price": 49.99}
    ],
    "subtotal": 109.97,
    "tax": 10.00,
    "shipping": 5.99,
    "total": 125.96,
    "shipping_address": {
        "street": "123 Main St",
        "city": "Boston",
        "state": "MA",
        "zip": "02101"
    },
    "payment_method": {
        "type": "credit_card",
        "card_number": "4532015112830366",
        "cvv": "123"
    }
})

result = order_validation_schema.validate(test_order, coerce=True)
print(f"Order validation: {'✓ Valid' if result.valid else '✗ Invalid'}")
if not result.valid:
    for error in result.errors:
        print(f"  - {error}")
```

### Example 2: API Request Validation

```python
# API request validation with nested schemas
api_request_schema = (Schema("APIRequest")
    .field("method", "STRING", required=True,
           constraints=[
               Enum(["GET", "POST", "PUT", "DELETE", "PATCH"])
           ])
    .field("endpoint", "STRING", required=True,
           constraints=[
               Pattern(r"^/api/v\d+/.*", "Endpoint must start with /api/v{version}/")
           ])
    .field("headers", "DICT", default={})
    .field("body", "DICT", required=False)
    .field("query_params", "DICT", default={})
    .field("auth_token", "STRING", required=True,
           constraints=[
               Pattern(r"^Bearer [A-Za-z0-9\-._~\+\/]+=*$", "Invalid auth token format")
           ])
    .field("timestamp", "STRING", required=True)
)

# Validate incoming API request
incoming_request = Record(data={
    "method": "POST",
    "endpoint": "/api/v2/users",
    "headers": {
        "Content-Type": "application/json",
        "Accept": "application/json"
    },
    "body": {
        "username": "new_user",
        "email": "user@example.com"
    },
    "auth_token": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    "timestamp": datetime.now().isoformat()
})

result = api_request_schema.validate(incoming_request)
if result.valid:
    print("✓ API request is valid")
    # Process the request
else:
    print("✗ Invalid API request:")
    for error in result.errors:
        print(f"  - {error}")
    # Return 400 Bad Request with errors
```

### Example 3: Data Quality Validation

```python
# Data quality checks for analytics pipeline
def check_data_freshness(timestamp_str):
    """Ensure data is not older than 24 hours"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return age_hours <= 24
    except:
        return False

def check_completeness(record_data):
    """Check if critical fields are not null/empty"""
    critical_fields = ["user_id", "event_type", "timestamp"]
    for field in critical_fields:
        if field not in record_data or not record_data[field]:
            return False
    return True

analytics_schema = (Schema("AnalyticsEvent", strict=False)
    .field("user_id", "STRING", required=True,
           constraints=[
               Pattern(r"^[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}$", 
                      "Invalid UUID format")
           ])
    .field("event_type", "STRING", required=True,
           constraints=[
               Enum(["page_view", "click", "purchase", "signup", "login", "logout"])
           ])
    .field("timestamp", "STRING", required=True,
           constraints=[
               Custom(check_data_freshness, "Data is older than 24 hours")
           ])
    .field("properties", "DICT", default={})
    .field("session_id", "STRING", required=True)
)

# Add meta-validation for data quality
def validate_data_quality(records):
    """Perform data quality checks on batch of records"""
    quality_report = {
        "total_records": len(records),
        "valid_records": 0,
        "invalid_records": 0,
        "completeness_score": 0,
        "freshness_score": 0,
        "errors": []
    }
    
    valid_count = 0
    fresh_count = 0
    complete_count = 0
    
    for record in records:
        # Schema validation
        result = analytics_schema.validate(record, coerce=True)
        if result.valid:
            valid_count += 1
        else:
            quality_report["errors"].extend(result.errors)
        
        # Completeness check
        if check_completeness(record.data):
            complete_count += 1
        
        # Freshness check
        if "timestamp" in record.data and check_data_freshness(record.data["timestamp"]):
            fresh_count += 1
    
    quality_report["valid_records"] = valid_count
    quality_report["invalid_records"] = len(records) - valid_count
    quality_report["completeness_score"] = (complete_count / len(records)) * 100
    quality_report["freshness_score"] = (fresh_count / len(records)) * 100
    
    return quality_report

# Test data quality validation
test_events = [
    Record(data={
        "user_id": "550E8400-E29B-41D4-A716-446655440000",
        "event_type": "page_view",
        "timestamp": datetime.now().isoformat(),
        "session_id": "session_123"
    }),
    Record(data={
        "user_id": "invalid-uuid",
        "event_type": "unknown",
        "timestamp": "2020-01-01T00:00:00",  # Old data
        "session_id": "session_456"
    })
]

quality_report = validate_data_quality(test_events)
print(f"\nData Quality Report:")
print(f"  Total Records: {quality_report['total_records']}")
print(f"  Valid: {quality_report['valid_records']}")
print(f"  Invalid: {quality_report['invalid_records']}")
print(f"  Completeness: {quality_report['completeness_score']:.1f}%")
print(f"  Freshness: {quality_report['freshness_score']:.1f}%")
```

## Best Practices

1. **Define Clear Schemas**: Make schemas explicit and well-documented
2. **Use Appropriate Constraints**: Choose constraints that match business rules
3. **Provide Meaningful Error Messages**: Help users understand what went wrong
4. **Enable Coercion Carefully**: Only coerce when data sources are trusted
5. **Validate Early**: Catch errors as close to the source as possible
6. **Use Composition**: Combine constraints for complex validation logic
7. **Test Edge Cases**: Validate with extreme and boundary values
8. **Performance Consideration**: Cache schemas for repeated validation
9. **Maintain Schema Versions**: Track schema changes over time
10. **Monitor Validation Metrics**: Track validation success/failure rates

## Summary

You've learned how to:

- Define schemas with field types and constraints
- Use built-in and custom constraints
- Compose constraints with logical operators
- Implement type coercion
- Validate uniqueness across records
- Combine validation with migration
- Build real-world validation scenarios

Next, explore the [Pandas Integration Tutorial](pandas-tutorial.md) to learn about DataFrame operations and analytics.