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
13. [History Storage](#history-storage)
14. [Best Practices](#best-practices)

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
    "run_on_failure": True,              # Optional: Run transforms despite a prior failure (default: False)
    "emit_output": False,                # Optional: End state whose records are excluded from output (default: True)
    "metadata": {...}                    # Optional: Additional metadata
}
```

**`run_on_failure`** — When a state transform raises, the engines record the
failure and skip the transforms of every subsequent state (and the failing
state's remaining transforms) so indeterminate, pre-failure data is not mutated
or persisted (e.g. a downstream load/upsert is not run). The record still
traverses to a final state and is reported as a failure. Set
`run_on_failure: true` to opt a state out of that skip — its transforms run even
after an upstream failure. Use it for recovery / compensation / cleanup /
dead-letter states. It re-enables the transforms only; the record is still
reported as a failure (`execute()` returns `success=False`).

**`emit_output`** — Defaults to `true`. Set `emit_output: false` on an **end**
state whose records should be excluded from the output (e.g. a `filtered`
terminal for dropped records, or an `error` terminal for rejected records). The
flag is consulted by output writers — the streaming sink and pattern writers
like `FileProcessor` — which apply the same exclusion in every processing mode.
Result-returning APIs (`process` / `process_batch`) do not drop records: they
return every result with its terminal state, so a caller can apply the flag
itself. (The `FileProcessor` pattern uses it for its `filtered` / `error`
states across STREAM, BATCH, and WHOLE.)

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
    "data_isolation": "copy",            # Optional: push-arc isolation (copy/reference/serialize)
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

3. **Built-in Functions** - Framework-provided functions from the FSM library
```python
{
    "type": "builtin",
    "name": "transformers.map_fields",
    "params": {"mapping": {"source_id": "id"}}
}
```

4. **Custom Functions** - Functions or classes from external modules
```python
{
    "type": "custom",
    "module": "my_package.my_module",
    "name": "MyTransform",
    "params": {"key": "value"}
}
```

#### Referencing built-in functions

The FSM ships a library of validators and transformers in
`dataknobs_fsm.functions.library`. They are registered automatically by
introspection under the names **`validators.<Name>`** and
**`transformers.<Name>`**, where `<Name>` is the exact public class or factory
name in that module. `params` are passed to the class constructor (or factory)
as keyword arguments to configure the function.

The library exposes each capability as **both a class and a keyword factory**:

| `name` | Kind | Example `params` |
|---|---|---|
| `transformers.map_fields` | factory → `FieldMapper` | `{"mapping": {"a": "b"}}` |
| `transformers.FieldMapper` | class | `{"field_map": {"a": "b"}}` |
| `transformers.normalize` | factory → `ValueNormalizer` | `{"email": "lowercase"}` |
| `validators.RequiredFieldsValidator` | class | `{"fields": ["a", "b"]}` |
| `validators.range_check` | factory → `RangeValidator` | `{"age": {"min": 0, "max": 150}}` |

> **`params` must be keyword arguments.** A factory whose signature is
> positional-only (e.g. `validators.required_fields(*fields)`) cannot be
> configured through the `params` dict — reference its class form instead
> (`validators.RequiredFieldsValidator` with `{"fields": [...]}`).

Built-in (and custom) **validators** gate state entry only when declared under
`pre_validators`; a `validators` (post-entry) entry whose result is a dict is
merged into the record. A built-in name that does not exist fails loudly at
build time with `ValueError: Built-in function not found: <name>`.

#### Referencing custom functions

A `custom` reference (`{"type": "custom", "module": ..., "name": ...}`) resolves
in one of two shapes:

- a **class** implementing an FSM function interface (`ITransformFunction`,
  `IValidationFunction`, `IStateTestFunction`) — constructed with `params` as
  constructor keyword arguments, exactly like a built-in class; or
- a plain **function** with the `(data, context)` signature — resolved through
  the standard wrapper path (it takes no `params`).

The interface method (`transform` / `validate` / `test`) may be **synchronous or
`async def`** — async custom methods are awaited. Unlike built-ins, a custom
*factory* function (one that returns a function-instance) is not supported; pass
the class directly to configure it with `params`. A missing module or attribute
fails loudly at build time with `ValueError: Custom function module not found:
<module>` or `ValueError: Custom function not found: <module>.<name>`.

#### The bare-string shorthand resolves to `registered`/`inline` only

In the state-sugar form, a bare string (e.g. `"transform": "data['x'] = 1"`)
resolves **only** to a pre-`registered` function (when the string matches a
registered name) or to `inline` code otherwise. It is **never** promoted to a
`builtin` or `custom` reference. To reference a built-in or external-module
function, use the full dict form shown above.

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

Environment variables can be used in configurations. The FSM
configuration loader delegates substitution to the canonical
`dataknobs_config.substitute_env_vars` helper, with a thin FSM-side
wrapper that preserves the documented `FSM_` prefix-fallback.

### Syntax

- `${VAR_NAME}` — Required variable; raises `ValueError` if unset
- `${VAR_NAME:default}` — DataKnobs legacy default form
- `${VAR_NAME:-default}` — Bash-style default (alias for `${VAR_NAME:default}`)
- `${VAR_NAME:?error message}` — Required with custom error message

Substitution is recursive (nested dicts and lists) and supports
embedded patterns such as `"http://${HOST}:${PORT}/path"` — the
references are replaced in-place and the surrounding string is
preserved.

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

### `FSM_` prefix-fallback

When a `${VAR}` reference is unset in the environment but
`${FSM_VAR}` is set, the prefixed value is used. This is convenient
for environments that namespace FSM-specific configuration but write
YAMLs against the unprefixed names.

## History Storage

Execution history is persisted via pluggable storage backends. All backends use
the `UnifiedDatabaseStorage` implementation, which works with any `dataknobs_data`
`AsyncDatabase` backend.

### Storage Backends

```python
from dataknobs_fsm.storage.base import StorageConfig, StorageBackend, StorageFactory

# Config-driven creation (factory selects the backend from the enum)
config = StorageConfig(backend=StorageBackend.MEMORY)
storage = StorageFactory.create(config)
await storage.initialize()

advanced_fsm.enable_history(storage=storage)
```

Available backends: `MEMORY`, `FILE`, `SQLITE`, `POSTGRES`, `MONGODB`,
`ELASTICSEARCH`, `S3`. Backend selection is driven by `StorageConfig.backend`
(the enum), so no redundant `'type'` key in `connection_params` is needed.

Convenience subclasses `InMemoryStorage` and `FileStorage` apply
backend-specific defaults automatically.

### Immutable, dict-loadable runtime configs

`StorageConfig` — along with the other FSM runtime configs (`PoolConfig`,
`IOConfig`, the streaming `StreamConfig`, and `ResourceConfig`) — is a frozen
`StructuredConfig` subclass. This means it is **dict-loadable** and
**immutable**:

```python
from dataclasses import replace

# Load from a config mapping (e.g. parsed YAML/JSON). Enum fields such as
# ``backend`` accept their raw string value ("memory", "file", ...).
config = StorageConfig.from_dict({"backend": "file", "compression": True})

# Serialize back. ``to_dict()`` keeps Enum members (in-process round-trip);
# ``to_json_dict()`` renders them as values for a JSON round-trip.
as_dict = config.to_json_dict()

# Immutable — derive a modified copy instead of mutating in place.
larger_batches = replace(config, batch_size=500)
```

These runtime config objects are the imperative construction layer. The
Pydantic FSM loader schema (`config/schema.py`) is the separate **declarative**
layer used when loading a full FSM definition from a file; it is unchanged.

### Constructing pattern consumers from config

The consumers built from these configs — `CircuitBreaker`, `Bulkhead`,
`ErrorRecoveryWorkflow`, `APIOrchestrator`, `DatabaseETL`, `FileProcessor`,
`StreamContext`, `AsyncStreamContext`, and `ResourcePool` — build through
`StructuredConfigConsumer`, so each accepts a config mapping directly via
`from_config(...)` in addition to a typed config instance:

```python
from dataknobs_fsm.patterns.error_recovery import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

# Typed config (unchanged):
cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

# Dict-dispatch — the mapping is projected onto CircuitBreakerConfig:
cb = CircuitBreaker.from_config({"failure_threshold": 3})

# All-default config (only for configs with no required fields):
cb = CircuitBreaker()

# ``self.config`` is the typed config (read-only):
assert cb.config.failure_threshold == 3
```

`ResourcePool` additionally carries a required `provider` collaborator (a live
resource provider, not config data). It keeps its back-compat
`ResourcePool(provider, config=None)` positional shortcut — the provider is
threaded through the mixin's collaborator channel while the config flows onto
`self.config` — and `ResourcePool.from_config(config, provider=provider)`
delivers the provider alongside the config:

```python
from dataknobs_fsm.resources.pool import ResourcePool, PoolConfig

# Back-compat positional (unchanged):
pool = ResourcePool(provider, PoolConfig(max_size=5))

# Dict/typed config with the provider as an injected collaborator:
pool = ResourcePool.from_config({"max_size": 5}, provider=provider)
```

### Database Injection

`UnifiedDatabaseStorage` (and its subclasses `InMemoryStorage`, `FileStorage`)
accept pre-built `AsyncDatabase` instances. This enables connection pool
sharing across components and simplifies testing.

```python
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_fsm.storage.database import UnifiedDatabaseStorage
from dataknobs_fsm.storage.base import StorageConfig, StorageBackend

# Share an existing database instance with FSM storage
shared_db = AsyncMemoryDatabase()  # or any AsyncDatabase from a pool

config = StorageConfig(backend=StorageBackend.MEMORY)
storage = UnifiedDatabaseStorage(config, database=shared_db)
await storage.initialize()  # skips factory creation, uses shared_db

advanced_fsm.enable_history(storage=storage)
```

The `database` and `steps_database` parameters are keyword-only:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `database` | `None` | Pre-built `AsyncDatabase` for history records. When provided, `_setup_backend()` skips factory creation. |
| `steps_database` | `None` | Separate `AsyncDatabase` for step records. Defaults to `database` when omitted. |

When both parameters are `None` (the default), the storage creates its own
database via the factory. When history and step records share a single
database instance, `UnifiedDatabaseStorage` automatically adds an `EXISTS`
filter on the type-specific field (`history_data` or `step_data`) to each
query, preventing record-type collisions. This works across all backends.

The factory also supports injection via keyword arguments:

```python
storage = StorageFactory.create(config, database=shared_db)
```

### Querying Histories

Use `query_histories()` to search execution histories with filters:

```python
# Filter by builtin keys
results = await storage.query_histories({"fsm_name": "my_fsm"})
results = await storage.query_histories({"status": "completed"})
results = await storage.query_histories({"failed": True})

# Time range filtering
results = await storage.query_histories({
    "start_time_after": 1710000000.0,
    "start_time_before": 1710100000.0,
})

# Pagination
results = await storage.query_histories({"fsm_name": "my_fsm"}, limit=10, offset=20)
```

#### Supported Filter Keys

| Key | Type | Description |
|-----|------|-------------|
| `fsm_name` | `str` | Exact match on FSM name |
| `data_mode` | `str` | Exact match on data handling mode |
| `status` | `str` | Exact match on status string |
| `start_time_after` | `float` | Histories started at or after this timestamp |
| `start_time_before` | `float` | Histories started at or before this timestamp |
| `failed` | `bool` | `True` for histories with failures, `False` for clean runs |
| `metadata.<key>` | `Any` | Exact match on a metadata field (see below) |

Unknown filter keys are logged as warnings and ignored.

#### Metadata Filtering

When saving a history with metadata, those metadata fields become queryable
using dot-notation filter keys:

```python
# Save histories with domain-specific metadata
await storage.save_history(history_a, metadata={"work_order_id": "WO-001", "scope_id": "S-A"})
await storage.save_history(history_b, metadata={"work_order_id": "WO-002", "scope_id": "S-B"})

# Filter by a single metadata field
results = await storage.query_histories({"metadata.work_order_id": "WO-001"})

# Filter by multiple metadata fields (AND semantics)
results = await storage.query_histories({
    "metadata.work_order_id": "WO-001",
    "metadata.scope_id": "S-A",
})

# Combine metadata filters with builtin filters
results = await storage.query_histories({
    "fsm_name": "order_processor",
    "metadata.work_order_id": "WO-001",
})
```

Metadata filtering uses query-level filtering (not post-filtering), so
pagination with `limit` and `offset` works correctly — the limit applies
to matched results, not to pre-filtered results.

> **Backend compatibility:** Metadata filtering is supported on **all** backends.
> SQL backends (PostgreSQL, SQLite, DuckDB) use native JSON path extraction
> in the query layer, while memory and file backends use `Record.get_value()`
> dot-notation traversal.

#### Symmetry kwargs: `filter_metadata=` and `sort=`

For callers composing FSM history with the `dataknobs-bots` registry layer
(`ArtifactRegistry.query`, `GeneratorRegistry.list_definitions`, etc.),
`query_histories` also accepts the same kw-only filter / sort surface so
the consumer sees one consistent shape:

```python
from dataknobs_data import SortSpec, SortOrder

# Symmetry form — equivalent to `filters={"metadata.tenant_id": "acme"}`.
results = await storage.query_histories(
    filter_metadata={"tenant_id": "acme"},
)

# Both routes AND-combine — additive over `metadata.X` entries in `filters`.
results = await storage.query_histories(
    filters={"metadata.work_order_id": "WO-001"},
    filter_metadata={"tenant_id": "acme"},
)

# `sort=` overrides the default `start_time DESC` ordering.
results = await storage.query_histories(
    filter_metadata={"tenant_id": "acme"},
    sort=[
        SortSpec(field="fsm_name", order=SortOrder.ASC),
        SortSpec(field="start_time", order=SortOrder.DESC),
    ],
)

# `filters=` is now optional (`None` ≡ `{}`).
all_histories = await storage.query_histories(limit=50)
```

`limit` / `offset` remain positional with defaults `100`/`0` for back-compat.

### Saving Steps with Metadata

`save_step` accepts a `metadata=` kwarg that lands in the underlying record's
`metadata` column.  Combined with `load_steps`'s symmetry kwargs, this makes
per-step cross-cutting context (tenant, correlation, audit) filterable
without mixing it into the structural step payload:

```python
# Save with metadata
await storage.save_step(
    "exec-1",
    step,
    metadata={"tenant_id": "acme", "correlation_id": "corr-42"},
)

# Filter on the metadata channel
acme_steps = await storage.load_steps(
    "exec-1", filter_metadata={"tenant_id": "acme"},
)

# AND-combines with data-column `filters=`
state_a_acme = await storage.load_steps(
    "exec-1",
    filters={"state_name": "state_a"},
    filter_metadata={"tenant_id": "acme"},
)

# Sort + pagination push down to the database query.
from dataknobs_data import SortSpec, SortOrder
asc = await storage.load_steps(
    "exec-1",
    sort=[SortSpec(field="timestamp", order=SortOrder.ASC)],
    limit=10,
    offset=0,
)
```

`limit=0` honors Python-slice semantics (empty result), aligning with the
post-fix pagination behavior in the underlying `dataknobs-data` layer.

> **Note:** `metadata=` on `save_step` and `save_history` is
> **consumer-supplied**.  The FSM engine itself does not populate it
> during execution — the engine has no opinion on which cross-cutting
> fields (`tenant_id`, `correlation_id`, audit, feature flags) the
> caller wants persisted, and forcing a default risks leaking
> caller-private context into history.  Wrap the storage call from
> your own execution path (or extend `UnifiedDatabaseStorage` in a
> subclass) to inject these fields uniformly.

#### Return Value

Each result dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Execution ID |
| `fsm_name` | `str` | FSM name |
| `data_mode` | `str` | Data handling mode |
| `status` | `str` | Execution status |
| `start_time` | `float \| None` | Start timestamp |
| `end_time` | `float \| None` | End timestamp (None if in progress) |
| `total_steps` | `int` | Total steps executed |
| `failed_steps` | `int` | Number of failed steps |
| `metadata` | `dict` | Metadata dict passed to `save_history()` |

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