# FSM Configuration Guide

This guide provides comprehensive documentation for creating FSM configurations in the DataKnobs FSM framework.

## Table of Contents
1. [Basic Structure](#basic-structure)
2. [States](#states)
3. [Arcs (Transitions)](#arcs-transitions)
4. [Networks and Subnetworks](#networks-and-subnetworks)
5. [Functions](#functions)
6. [Data Modes](#data-modes)
7. [Transaction Management](#transaction-management)
8. [Resources](#resources)
9. [Execution Strategies](#execution-strategies)
10. [Complete Examples](#complete-examples)
11. [Configuration Formats](#configuration-formats)
12. [Environment Variables](#environment-variables)
13. [Best Practices](#best-practices)

## Basic Structure

Every FSM configuration must have the following top-level structure:

```python
config = {
    "name": "MyFSM",                    # Required: Name of the FSM
    "version": "1.0.0",                  # Optional: Version (default: "1.0.0")
    "description": "Description",        # Optional: FSM description
    "main_network": "main",              # Required: Name of the main network to execute
    "networks": [                        # Required: List of network definitions
        {
            "name": "main",              # Network name (must match main_network)
            "states": [...],             # List of state definitions
            # Note: arcs can be defined at network OR state level (see below)
        }
    ],
    "data_mode": {...},                  # Optional: Data handling configuration
    "transaction": {...},                # Optional: Transaction configuration
    "resources": [...],                  # Optional: External resource definitions
    "execution_strategy": "depth_first", # Optional: Execution strategy
    "max_transitions": 1000,             # Optional: Maximum transitions (default: 1000)
    "timeout_seconds": 60,               # Optional: Execution timeout
    "metadata": {...}                    # Optional: Additional metadata
}
```

## States

States are the nodes in your FSM graph. Each state has a name and metadata that defines its behavior.

### State Properties

```python
{
    "name": "state_name",                # Required: Unique name within the network
    "is_start": True,                    # Optional: Marks this as an initial state (default: False)
    "is_end": True,                      # Optional: Marks this as a final state (default: False)

    # Schema validation (JSON Schema format)
    "schema": {                          # Optional: JSON schema for data validation
        "type": "object",
        "properties": {...},
        "required": [...]
    },

    # Function definitions
    "pre_validators": [...],             # Optional: Pre-validation functions (run before state)
    "validators": [...],                 # Optional: Validation functions
    "transforms": [...],                 # Optional: Transform functions (run on state entry)

    # Arc definitions (state-level)
    "arcs": [...],                       # Optional: Outgoing transitions (see Arcs section)

    # Resources and configuration
    "resources": ["db", "api"],          # Optional: Required resource names
    "data_mode": "copy",                 # Optional: Override data mode for this state
    "metadata": {...}                    # Optional: Additional metadata
}
```

### State Types

1. **Initial States** (`is_start: true`)
   - Entry points for the FSM
   - At least one required per network
   - Multiple initial states allowed for different entry scenarios

2. **Final States** (`is_end: true`)
   - Terminal states where execution ends
   - No outgoing arcs allowed from final states
   - Can have transform functions for final data processing

3. **Normal States** (neither start nor end)
   - Intermediate processing states
   - Must have at least one incoming and one outgoing arc

4. **Start-End States** (`is_start: true` AND `is_end: true`)
   - States that can serve as both entry and exit points
   - Useful for simple pass-through networks

### Example State Definitions

```python
"states": [
    {
        "name": "input",
        "is_start": true,
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "count": {"type": "integer", "minimum": 0}
            },
            "required": ["data"]
        },
        "arcs": [
            {"target": "validate"}  # Simple arc to next state
        ]
    },
    {
        "name": "validate",
        "validators": [
            {
                "type": "inline",
                "code": "lambda state: state.data.get('count', 0) > 0"
            }
        ],
        "arcs": [
            {
                "target": "process",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: state.data.get('valid', True)"
                }
            },
            {
                "target": "error",
                "condition": {
                    "type": "inline",
                    "code": "lambda state: not state.data.get('valid', True)"
                }
            }
        ]
    },
    {
        "name": "process",
        "transforms": [
            {
                "type": "inline",
                "code": "lambda state: {'processed': state.data['data'].upper()}"
            }
        ],
        "resources": ["database"],  # Requires database resource
        "arcs": [
            {"target": "output"}
        ]
    },
    {
        "name": "output",
        "is_end": true
    },
    {
        "name": "error",
        "is_end": true
    }
]
```

## Arcs (Transitions)

Arcs define the transitions between states and the conditions under which they occur. Arcs can be defined in two ways:

### Method 1: State-Level Arcs (Nested)

Define arcs as a property of the source state:

```python
{
    "name": "state_a",
    "arcs": [
        {
            "target": "state_b",         # Required: Target state name
            "condition": {...},          # Optional: Condition function
            "transform": {...},          # Optional: Transform function
            "priority": 0,               # Optional: Priority (higher = higher priority)
            "metadata": {...}            # Optional: Additional metadata
        }
    ]
}
```

### Method 2: Network-Level Arcs (Adjacent)

Define arcs at the network level with explicit `from` and `to` fields:

```python
{
    "name": "main",
    "states": [...],
    "arcs": [
        {
            "from": "state_a",           # Required: Source state name
            "to": "state_b",             # Required: Target state name
            "name": "arc_name",          # Optional: Arc name
            "condition": {...},          # Optional: Condition function
            "transform": {...},          # Optional: Transform function
            "priority": 0,               # Optional: Priority
            "metadata": {...}            # Optional: Additional metadata
        }
    ]
}
```

### Arc Properties

- **target/to**: The destination state name
- **condition**: Function that determines if the arc should be taken
- **transform**: Function that transforms data during the transition
- **priority**: Integer priority for arc selection (higher values = higher priority, default: 0)
- **metadata**: Additional information about the arc

### Push Arcs (Subnetwork Transitions)

Push arcs allow transitions to different networks (subnetworks):

```python
{
    "target": "initial_state",           # Initial state in target network
    "target_network": "validation",      # Required: Target network name
    "return_state": "continue",          # Optional: State to return to after subnetwork
    "data_isolation": "copy",            # Optional: Data handling mode
    "condition": {...},                  # Optional: Condition function
    "transform": {...}                   # Optional: Transform function
}
```

**Format**: `"target_network[:initial_state]"`
- If initial_state is omitted, the subnetwork's default initial state is used
- Example: `"validation:start"` pushes to the "validation" network's "start" state

### Conditional Transitions

```python
{
    "target": "process",
    "condition": {
        "type": "inline",
        "code": "lambda state: state.data.get('valid', False)"
    }
}
```

### Transform on Transition

```python
{
    "target": "transform",
    "transform": {
        "type": "inline",
        "code": "lambda state: {'records': [r.upper() for r in state.data['records']]}"
    }
}
```

### Arc Priority and Selection

When multiple arcs from a state have conditions that evaluate to true:
1. Arcs are evaluated in priority order (highest priority first)
2. Within the same priority, arcs are evaluated in definition order
3. The first arc whose condition evaluates to true is taken

## Networks and Subnetworks

Networks are collections of states and arcs that define a complete FSM or sub-FSM.

### Network Properties

```python
{
    "name": "network_name",              # Required: Unique network name
    "states": [...],                     # Required: List of states
    "arcs": [...],                       # Optional: Network-level arc definitions
    "resources": ["db", "api"],          # Optional: Resource names used by this network
    "streaming": {                       # Optional: Streaming configuration
        "enabled": true,
        "chunk_size": 100,
        "parallelism": 4
    },
    "metadata": {...}                    # Optional: Additional metadata
}
```

### Multi-Network FSMs

FSMs can have multiple networks for modular design:

```python
{
    "name": "MultiNetworkFSM",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {
                    "name": "start",
                    "is_start": true,
                    "arcs": [
                        {
                            "target": "validate",
                            "target_network": "validation",  # Push to validation network
                            "return_state": "process"        # Return here after validation
                        }
                    ]
                },
                {
                    "name": "process",
                    "arcs": [{"target": "end"}]
                },
                {
                    "name": "end",
                    "is_end": true
                }
            ]
        },
        {
            "name": "validation",
            "states": [
                {
                    "name": "validate",
                    "is_start": true,
                    "validators": [...],
                    "arcs": [{"target": "complete"}]
                },
                {
                    "name": "complete",
                    "is_end": true
                }
            ]
        }
    ]
}
```

## Functions

Functions define the processing logic for states and transitions. They receive specific parameters and must return appropriate values.

### Function Types

1. **Inline Functions** - Lambda expressions or simple Python code
```python
{
    "type": "inline",
    "code": "lambda state: {'result': state.data['value'] * 2}"
}
```

2. **Registered Functions** - Pre-registered Python functions
```python
{
    "type": "registered",
    "name": "process_data"
}
```

3. **Built-in Functions** - Framework-provided functions
```python
{
    "type": "builtin",
    "name": "validators.validate_json",
    "params": {"schema": {...}}
}
```

4. **Custom Functions** - Functions from external modules
```python
{
    "type": "custom",
    "module": "my_module",
    "name": "my_function"
}
```

### Function Interfaces and Parameters

#### Validation Functions

Validation functions check data validity and are called with:

**Input Parameters:**
- `data`: The current state data (dict or object)
- `context`: Optional execution context containing:
  - `state_name`: Current state name
  - `metadata`: State metadata
  - `resources`: Available resources
  - `variables`: Shared variables

**Return Value:**
- Boolean: `True` if validation passes, `False` otherwise
- OR ExecutionResult object with `success` property

```python
# Example validation function
def validate_data(data, context=None):
    """Validate that required fields exist and are valid."""
    if not data.get('user_id'):
        return False
    if data.get('amount', 0) < 0:
        return False
    return True
```

#### Transform Functions

Transform functions modify data and are called with:

**Input Parameters:**
- `data`: The current state data
- `context`: Optional execution context (same as validation)

**Return Value:**
- Modified data (dict or object)
- OR ExecutionResult object with `data` property

```python
# Example transform function
def transform_data(data, context=None):
    """Transform data by adding timestamp and formatting."""
    return {
        'original': data,
        'timestamp': time.time(),
        'formatted': data.get('text', '').upper()
    }
```

#### Condition Functions (State Test)

Condition functions determine arc traversal and are called with:

**Input Parameters:**
- `data`: The current state data
- `context`: Optional execution context

**Return Value:**
- Boolean: `True` to take the arc, `False` to skip
- OR tuple of (boolean, reason_string)

```python
# Example condition function
def check_threshold(data, context=None):
    """Check if value exceeds threshold."""
    threshold = context.get('variables', {}).get('threshold', 100)
    return data.get('value', 0) > threshold
```

### Function Definition in States

```python
{
    "name": "process",
    "pre_validators": [              # Run before state entry
        {
            "type": "inline",
            "code": "lambda data: data.get('input') is not None"
        }
    ],
    "validators": [                  # Validate state data
        {
            "type": "registered",
            "name": "validate_format"
        }
    ],
    "transforms": [                  # Transform data on state entry
        {
            "type": "inline",
            "code": "lambda data: {'result': process(data['input'])}"
        }
    ]
}
```

### Function Execution Order

For a state, functions execute in this order:
1. **Pre-validators** - Check if state can be entered
2. **Validators** - Validate current data
3. **Transforms** - Modify data (executed sequentially)
4. **Arc conditions** - Determine next state

## Data Modes

Data modes control how data is handled during state transitions.

### Configuration

```python
{
    "data_mode": {
        "default": "copy",               # Default mode for all states
        "state_overrides": {             # Override for specific states
            "stream_state": "reference",
            "process_state": "direct"
        },
        "copy_config": {...},            # Configuration for COPY mode
        "reference_config": {...},       # Configuration for REFERENCE mode
        "direct_config": {...}           # Configuration for DIRECT mode
    }
}
```

### Available Modes

1. **COPY** (default)
   - Creates deep copies of data for each state
   - Safe for rollback and parallel processing
   - Higher memory usage
   - Best for: Transactional workflows, data integrity critical

2. **REFERENCE**
   - Passes references to the same data object
   - Memory efficient
   - Changes affect all states
   - Best for: Read-only workflows, streaming data

3. **DIRECT**
   - Direct manipulation without copying
   - Most efficient performance
   - No rollback capability
   - Best for: Simple transformations, performance critical

## Transaction Management

Configure how the FSM handles transactions across states.

### Configuration

```python
{
    "transaction": {
        "strategy": "batch",             # Transaction strategy
        "batch_size": 100,               # Batch size for BATCH strategy
        "commit_triggers": ["save"],     # State names that trigger commits
        "rollback_on_error": true,       # Rollback on error (default: true)
        "timeout_seconds": 30            # Transaction timeout
    }
}
```

### Transaction Strategies

1. **SINGLE** - One transaction per FSM execution
2. **BATCH** - Group operations in batches
3. **MANUAL** - Explicit transaction control in functions
4. **NONE** - No transaction management

## Resources

Resources are external dependencies that states can use.

### Resource Types

1. **database** - Database connections
2. **filesystem** - File system access
3. **http** - HTTP/REST API clients
4. **llm** - Language model interfaces
5. **vector_store** - Vector database connections
6. **custom** - User-defined resource types

### Resource Configuration

```python
{
    "resources": [
        {
            "name": "database",
            "type": "database",
            "config": {
                "connection_string": "postgresql://localhost/mydb",
                "pool_size": 10
            },
            "connection_pool_size": 10,  # Connection pool size
            "timeout_seconds": 30,        # Operation timeout
            "retry_attempts": 3,          # Retry count on failure
            "retry_delay_seconds": 1.0,   # Delay between retries
            "health_check_interval": 60   # Health check interval (seconds)
        },
        {
            "name": "api",
            "type": "http",
            "config": {
                "base_url": "https://api.example.com",
                "headers": {
                    "Authorization": "Bearer ${API_TOKEN}"
                },
                "timeout": 30
            }
        },
        {
            "name": "llm",
            "type": "llm",
            "config": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "${OPENAI_API_KEY}",
                "temperature": 0.7
            }
        },
        {
            "name": "custom_resource",
            "type": "custom",
            "config": {
                "class": "my_module.MyResourceClass",
                "param1": "value1"
            }
        }
    ]
}
```

### Using Resources in Functions

Resources are available in the function context:

```python
def fetch_data(data, context):
    """Fetch data using configured resources."""
    # Access database resource
    db = context['resources']['database']
    result = db.query("SELECT * FROM users WHERE id = ?", [data['user_id']])

    # Access API resource
    api = context['resources']['api']
    response = api.get(f"/users/{data['user_id']}")

    return {
        'db_data': result,
        'api_data': response.json()
    }
```

## Execution Strategies

Control how the FSM executes states and transitions.

### Available Strategies

```python
{
    "execution_strategy": "depth_first"  # Execution strategy
}
```

**Options:**
- `"depth_first"` - Depth-first traversal (default)
- `"breadth_first"` - Breadth-first traversal
- `"resource_optimized"` - Optimize for resource utilization
- `"stream_optimized"` - Optimize for streaming data

## Complete Examples

### Example 1: Simple Data Processing Pipeline

```python
simple_pipeline = {
    "name": "SimpleDataPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "copy"
    },
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "input",
                "is_start": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"}
                    },
                    "required": ["data"]
                },
                "arcs": [
                    {"target": "validate"}
                ]
            },
            {
                "name": "validate",
                "validators": [
                    {
                        "type": "inline",
                        "code": "lambda data: len(data.get('data', '')) > 0"
                    }
                ],
                "arcs": [
                    {
                        "target": "transform",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: data.get('valid', True)"
                        }
                    },
                    {
                        "target": "error",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: not data.get('valid', True)"
                        }
                    }
                ]
            },
            {
                "name": "transform",
                "transforms": [
                    {
                        "type": "inline",
                        "code": "lambda data: {'result': data['data'].upper(), 'length': len(data['data'])}"
                    }
                ],
                "arcs": [
                    {"target": "output"}
                ]
            },
            {
                "name": "output",
                "is_end": true
            },
            {
                "name": "error",
                "is_end": true,
                "transforms": [
                    {
                        "type": "inline",
                        "code": "lambda data: {'error': 'Validation failed', 'input': data}"
                    }
                ]
            }
        ]
    }]
}
```

### Example 2: ETL Pipeline with Resources and Error Handling

```python
etl_pipeline = {
    "name": "ETLPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "copy"
    },
    "transaction": {
        "strategy": "batch",
        "batch_size": 1000,
        "rollback_on_error": true
    },
    "resources": [
        {
            "name": "source_db",
            "type": "database",
            "config": {
                "connection_string": "${SOURCE_DB_URL}",
                "pool_size": 5
            }
        },
        {
            "name": "target_db",
            "type": "database",
            "config": {
                "connection_string": "${TARGET_DB_URL}",
                "pool_size": 10
            }
        }
    ],
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "start",
                "is_start": true,
                "arcs": [{"target": "extract"}]
            },
            {
                "name": "extract",
                "transforms": [
                    {
                        "type": "registered",
                        "name": "extract_from_source"
                    }
                ],
                "resources": ["source_db"],
                "arcs": [
                    {
                        "target": "validate",
                        "transform": {
                            "type": "inline",
                            "code": "lambda data: {'records': data['extracted'], 'count': len(data['extracted'])}"
                        }
                    }
                ]
            },
            {
                "name": "validate",
                "validators": [
                    {
                        "type": "registered",
                        "name": "validate_records"
                    }
                ],
                "arcs": [
                    {
                        "target": "transform",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: data.get('validation_passed', False)"
                        }
                    },
                    {
                        "target": "failure",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: not data.get('validation_passed', False)"
                        }
                    }
                ]
            },
            {
                "name": "transform",
                "transforms": [
                    {
                        "type": "registered",
                        "name": "transform_records"
                    }
                ],
                "arcs": [{"target": "load"}]
            },
            {
                "name": "load",
                "transforms": [
                    {
                        "type": "registered",
                        "name": "load_to_target"
                    }
                ],
                "resources": ["target_db"],
                "arcs": [
                    {
                        "target": "success",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: data.get('load_successful', False)"
                        }
                    },
                    {
                        "target": "failure",
                        "condition": {
                            "type": "inline",
                            "code": "lambda data: not data.get('load_successful', False)"
                        }
                    }
                ]
            },
            {
                "name": "success",
                "is_end": true,
                "transforms": [
                    {
                        "type": "inline",
                        "code": "lambda data: {'status': 'success', 'records_processed': data.get('count', 0)}"
                    }
                ]
            },
            {
                "name": "failure",
                "is_end": true,
                "transforms": [
                    {
                        "type": "registered",
                        "name": "rollback_changes"
                    }
                ]
            }
        ]
    }]
}
```

### Example 3: Multi-Network FSM with Subnetwork Calls

```python
multi_network_fsm = {
    "name": "OrderProcessingFSM",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {
                    "name": "receive_order",
                    "is_start": true,
                    "arcs": [
                        {
                            "target": "validate_order",
                            "target_network": "validation",
                            "return_state": "process_payment"
                        }
                    ]
                },
                {
                    "name": "process_payment",
                    "arcs": [
                        {
                            "target": "check_payment",
                            "target_network": "payment",
                            "return_state": "ship_order"
                        }
                    ]
                },
                {
                    "name": "ship_order",
                    "arcs": [{"target": "complete"}]
                },
                {
                    "name": "complete",
                    "is_end": true
                }
            ]
        },
        {
            "name": "validation",
            "states": [
                {
                    "name": "validate_order",
                    "is_start": true,
                    "validators": [
                        {
                            "type": "inline",
                            "code": "lambda data: all([data.get('customer_id'), data.get('items')])"
                        }
                    ],
                    "arcs": [{"target": "validation_complete"}]
                },
                {
                    "name": "validation_complete",
                    "is_end": true
                }
            ]
        },
        {
            "name": "payment",
            "states": [
                {
                    "name": "check_payment",
                    "is_start": true,
                    "transforms": [
                        {
                            "type": "registered",
                            "name": "process_payment_method"
                        }
                    ],
                    "arcs": [{"target": "payment_complete"}]
                },
                {
                    "name": "payment_complete",
                    "is_end": true
                }
            ]
        }
    ]
}
```

## Configuration Formats

### Simple Format

For simple FSMs, you can use a simplified format that gets transformed to the full format:

```python
{
    "name": "SimpleFSM",
    "states": {                         # Dict format for states
        "start": {
            "is_start": true,
            "on_complete": {"target": "process"}  # Inline transition
        },
        "process": {
            "transform": "lambda data: {'result': data['input'] * 2}",
            "on_complete": {"target": "end"}
        },
        "end": {
            "final": true                # Alternative to is_end
        }
    },
    "initial_state": "start"             # Alternative way to specify start state
}
```

### YAML Format

FSM configurations can be written in YAML for better readability:

```yaml
name: MyFSM
main_network: main
networks:
  - name: main
    states:
      - name: start
        is_start: true
        arcs:
          - target: process
      - name: process
        transforms:
          - type: inline
            code: "lambda data: {'result': data['value'] * 2}"
        arcs:
          - target: end
      - name: end
        is_end: true
```

## Environment Variables

Environment variables can be used in configurations:

### Syntax

- `${VAR_NAME}` - Required variable
- `${VAR_NAME:-default}` - Variable with default value
- `${VAR_NAME:?error message}` - Required with custom error message

### Example

```python
{
    "resources": [
        {
            "name": "database",
            "type": "database",
            "config": {
                "connection_string": "${DATABASE_URL}",
                "password": "${DB_PASSWORD:-defaultpass}",
                "api_key": "${API_KEY:?API key is required}"
            }
        }
    ]
}
```

The loader will also check for variables with the `FSM_` prefix automatically.

## Best Practices

### 1. State Design
- **Always define at least one initial state** (`is_start: true`)
- **Always define at least one final state** (`is_end: true`)
- **Use meaningful state names** that describe the operation
- **Keep states focused** on a single responsibility
- **Validate early** to catch errors sooner

### 2. Arc Configuration
- **Use priorities** when multiple conditions might be true
- **Provide clear conditions** with descriptive code
- **Avoid complex conditions** - use separate states if needed
- **Document arc purposes** in metadata

### 3. Function Implementation
- **Keep functions pure** when possible (no side effects)
- **Handle errors gracefully** - return appropriate values
- **Use registered functions** for complex logic
- **Test functions independently** before integration

### 4. Data Management
- **Use COPY mode** for transactional workflows
- **Use REFERENCE mode** for read-only or streaming workflows
- **Use DIRECT mode** only when performance is critical
- **Define schemas** for data validation

### 5. Resource Management
- **Configure appropriate pool sizes** based on load
- **Set reasonable timeouts** to prevent hanging
- **Use retry policies** for resilient operations
- **Implement health checks** for critical resources

### 6. Network Organization
- **Use subnetworks** for modular, reusable components
- **Keep networks focused** on specific workflows
- **Define clear interfaces** between networks
- **Document network purposes** and dependencies

### 7. Error Handling
- **Always have error states** for graceful failure
- **Use rollback capabilities** for data integrity
- **Log errors with context** for debugging
- **Provide meaningful error messages**

### 8. Performance Optimization
- **Use appropriate execution strategies** for your use case
- **Configure batch sizes** based on data volume
- **Set max_transitions** to prevent infinite loops
- **Use streaming** for large data sets

### 9. Configuration Management
- **Use environment variables** for sensitive data
- **Version your configurations** with the version field
- **Document configurations** with descriptions and metadata
- **Validate configurations** before deployment

### 10. Testing
- **Test each state independently**
- **Test all arc conditions**
- **Test error paths**
- **Test with production-like data volumes**
- **Monitor execution metrics**

## Common Patterns

### Pattern 1: Validation → Process → Output
```python
"states": [
    {"name": "input", "is_start": true, "arcs": [{"target": "validate"}]},
    {"name": "validate", "validators": [...], "arcs": [
        {"target": "process", "condition": {"type": "inline", "code": "lambda d: d.get('valid')"}},
        {"target": "error", "condition": {"type": "inline", "code": "lambda d: not d.get('valid')"}}
    ]},
    {"name": "process", "transforms": [...], "arcs": [{"target": "output"}]},
    {"name": "output", "is_end": true},
    {"name": "error", "is_end": true}
]
```

### Pattern 2: Retry with Exponential Backoff
```python
"states": [
    {"name": "attempt", "transforms": [...], "arcs": [
        {"target": "success", "condition": {"type": "inline", "code": "lambda d: d.get('succeeded')"}},
        {"target": "wait", "condition": {"type": "inline", "code": "lambda d: d.get('retries', 0) < 3"}},
        {"target": "failure", "condition": {"type": "inline", "code": "lambda d: d.get('retries', 0) >= 3"}}
    ]},
    {"name": "wait", "transforms": [
        {"type": "inline", "code": "lambda d: {**d, 'wait': 2 ** d.get('retries', 0), 'retries': d.get('retries', 0) + 1}"}
    ], "arcs": [{"target": "attempt"}]},
    {"name": "success", "is_end": true},
    {"name": "failure", "is_end": true}
]
```

### Pattern 3: Fan-Out/Fan-In
```python
"states": [
    {"name": "split", "transforms": [
        {"type": "inline", "code": "lambda d: {'chunks': split_data(d['data'])}"}
    ], "arcs": [
        {"target": "process_chunk", "target_network": "processor", "return_state": "merge"}
    ]},
    {"name": "merge", "transforms": [
        {"type": "inline", "code": "lambda d: {'result': combine_results(d['chunks'])}"}
    ], "arcs": [{"target": "complete"}]}
]
```

## Troubleshooting

### Common Errors

1. **"Network must have at least one start state"**
   - Solution: Add `"is_start": true` to at least one state

2. **"Arc target 'X' not found in network"**
   - Solution: Ensure all arc targets reference existing state names

3. **"Main network 'X' not found"**
   - Solution: Ensure `main_network` matches a network name in `networks`

4. **"Resource 'X' not found"**
   - Solution: Define the resource in the `resources` section

5. **Function execution errors**
   - Check lambda syntax for inline functions
   - Ensure registered functions are properly registered before FSM creation
   - Verify function parameters match expected signature
   - Check that state.data structure matches what functions expect

### Debugging Tips

1. **Enable logging** to track state transitions
2. **Add metadata** to states and arcs for debugging context
3. **Use simple test data** to verify configuration
4. **Test functions independently** before integrating
5. **Start with simple configurations** and gradually add complexity
6. **Use the FSM debugger** to step through execution
7. **Monitor resource usage** and execution metrics
8. **Validate configurations** before deployment

## Migration from Other Formats

If migrating from other FSM formats, note these key differences:

1. **States and arcs can be defined flexibly** - either nested in states or separately in networks
2. **Use `is_start` and `is_end` flags** instead of special state types
3. **Functions are defined with type specifications** (inline, registered, builtin, custom)
4. **Networks allow for modular FSM composition**
5. **Data modes provide fine-grained control** over data handling
6. **Resources are explicitly configured** and managed
7. **Push arcs enable subnetwork transitions** with return states
8. **Transaction management** is configurable per FSM