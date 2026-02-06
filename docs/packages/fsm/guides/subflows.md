# FSM Subflows and Hierarchical Composition

## Overview

The DataKnobs FSM framework supports hierarchical state machine composition through **subflows**. A subflow allows one FSM network to delegate execution to another network, creating modular, reusable processing pipelines. This is implemented through `PushArc`, a specialized arc type that pushes execution onto a sub-network and returns control to the parent when the sub-network reaches a final state.

Key concepts:

- **PushArc**: An arc that transitions into a sub-network instead of a sibling state
- **Network stack**: A runtime stack (`ExecutionContext.network_stack`) that tracks nested network execution
- **Data mapping**: Field-level control over what data flows between parent and child networks
- **Data isolation**: Configurable isolation modes that determine how data is shared or copied
- **Multi-transform arcs**: Arcs with sequential transform pipelines (applicable to both regular arcs and subflows)

## PushArc Configuration

`PushArc` extends `ArcDefinition` with fields specific to sub-network invocation. It is defined in `dataknobs_fsm.core.arc`.

### Class Definition

```python
from dataknobs_fsm.core.arc import PushArc, DataIsolationMode

@dataclass
class PushArc(ArcDefinition):
    """Arc that pushes to a sub-network."""

    target_network: str = ""
    return_state: str | None = None
    isolation_mode: DataIsolationMode = DataIsolationMode.COPY
    pass_context: bool = True
    data_mapping: Dict[str, str] = field(default_factory=dict)
    result_mapping: Dict[str, str] = field(default_factory=dict)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_network` | `str` | `""` | Name of the sub-network to push to. Supports `"network:state"` syntax to specify a custom initial state. |
| `return_state` | `str \| None` | `None` | State in the parent network to return to after the sub-network completes. |
| `isolation_mode` | `DataIsolationMode` | `COPY` | How data is isolated between parent and child (see below). |
| `pass_context` | `bool` | `True` | Whether to propagate the execution context to the sub-network. |
| `data_mapping` | `Dict[str, str]` | `{}` | Maps parent data fields to child data fields: `{'parent_field': 'child_field'}`. |
| `result_mapping` | `Dict[str, str]` | `{}` | Maps child result fields back to parent fields: `{'child_result': 'parent_field'}`. |

Inherited from `ArcDefinition`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_state` | `str` | (required) | Target state for standard arcs. For push arcs, this is typically set but `target_network` drives the actual transition. |
| `pre_test` | `str \| None` | `None` | Condition function name. The arc is only taken if this returns `True`. |
| `transform` | `str \| list[str] \| None` | `None` | Transform function(s) to apply during the arc transition. |
| `priority` | `int` | `0` | Higher priority arcs are evaluated first. |
| `definition_order` | `int` | `0` | Tracks definition order for stable sorting when priorities are equal. |
| `metadata` | `Dict[str, Any]` | `{}` | Arbitrary metadata attached to the arc. |
| `required_resources` | `Dict[str, str]` | `{}` | Resources the arc needs, e.g., `{'database': 'main_db'}`. |

### Custom Initial State Syntax

The `target_network` field supports a colon-separated syntax to specify which state to enter in the sub-network:

```python
# Enter the sub-network at its default initial state
PushArc(target_network="validation", ...)

# Enter the sub-network at a specific state
PushArc(target_network="validation:deep_check", ...)
```

The execution engine parses this in `_execute_push_arc`:

```python
if ':' in push_arc.target_network:
    network_name, initial_state = push_arc.target_network.split(':', 1)
else:
    network_name = push_arc.target_network
    initial_state = None  # Uses network's default initial state
```

### Configuration Format

In JSON/dict configuration, push arcs are distinguished from regular arcs by the presence of the `target_network` key. The config schema validator (`StateConfig.validate_arcs`) automatically detects this:

```python
# Regular arc config
{"target": "next_state", "condition": {...}, "priority": 1}

# Push arc config (target_network triggers PushArcConfig)
{
    "target": "placeholder",
    "target_network": "validation",
    "return_state": "post_validation",
    "data_isolation": "copy"
}
```

## DataIsolationMode

`DataIsolationMode` controls how data is handled when pushing to a sub-network. It is defined as an enum in `dataknobs_fsm.core.arc`.

```python
class DataIsolationMode(Enum):
    COPY = "copy"           # Deep copy data when pushing
    REFERENCE = "reference" # Pass data by reference
    SERIALIZE = "serialize" # Serialize/deserialize for isolation
```

### Mode Comparison

| Mode | Safety | Performance | Use Case |
|------|--------|-------------|----------|
| **COPY** | High - full isolation via `copy.deepcopy()` | Medium - deep copy overhead | Default. Safe for concurrent or independent subflows. |
| **REFERENCE** | Low - parent and child share the same object | Fast - no copying | When subflow intentionally mutates parent data or data is read-only. |
| **SERIALIZE** | High - isolation via JSON round-trip | Slower - serialization overhead | When data must cross serialization boundaries or you need to verify serializability. |

### How Isolation Is Applied

The `ExecutionEngine._execute_push_arc` method applies isolation after data mapping but before entering the sub-network:

```python
# After data_mapping has been applied to produce mapped_data:

if push_arc.isolation_mode == DataIsolationMode.COPY:
    context.data = copy.deepcopy(mapped_data)
elif push_arc.isolation_mode == DataIsolationMode.SERIALIZE:
    serialized = dumps(mapped_data)
    context.data = loads(serialized)
else:
    # REFERENCE mode - use data directly
    context.data = mapped_data
```

## ExecutionEngine Subflow Support

The `ExecutionEngine` (in `dataknobs_fsm.execution.engine`) handles the full lifecycle of subflow execution through three methods: `_execute_push_arc`, `_check_subflow_completion`, and `_pop_subflow`.

### Network Stack

The `ExecutionContext` maintains a `network_stack` that tracks the hierarchy of active sub-networks:

```python
# ExecutionContext (execution/context.py)
network_stack: List[Tuple[str, str | None]] = []
```

Each entry is a tuple of `(network_name, return_state)`. The stack grows when push arcs are followed and shrinks when sub-networks complete.

```python
def push_network(self, network_name: str, return_state: str | None = None) -> None:
    self.network_stack.append((network_name, return_state))

def pop_network(self) -> Tuple[str, str | None]:
    if self.network_stack:
        return self.network_stack.pop()
    return ("", None)
```

### Push Flow

When the execution engine encounters a `PushArc` during `_execute_transition`, it delegates to `_execute_push_arc`:

1. **Depth check**: Validates `len(context.network_stack) < max_subflow_depth` (default 10) to prevent infinite recursion.
2. **Network resolution**: Looks up `target_network` in `self.fsm.networks`. Supports `"network:state"` syntax for custom initial states.
3. **Data mapping**: Applies `data_mapping` to project parent fields into child fields via `_apply_data_mapping`.
4. **Data isolation**: Applies the configured `DataIsolationMode`.
5. **Stack push**: Calls `context.push_network(network_name, push_arc.return_state)`.
6. **Resource inheritance**: Stores `context.parent_state_resources` so sub-network states can access parent resources.
7. **State entry**: Enters the sub-network's initial state (either specified or the network's default).

### Pop Flow (Subflow Completion)

After each transition, the execution loop checks `_check_subflow_completion`:

1. Verifies `context.network_stack` is non-empty (we are in a subflow).
2. Looks up the current network from the top of the stack.
3. Checks if the current state is in the network's `final_states`.
4. If so, calls `_pop_subflow` which:
   - Pops the network from the stack via `context.pop_network()`.
   - Applies `result_mapping` if a `PushArc` reference is available.
   - Enters the `return_state` in the parent network.

```python
# Simplified flow in _execute_single:
while transitions < max_transitions:
    if self._is_final_state(context.current_state):
        return True, context.data

    # ... evaluate arcs and execute transition ...

    # After transition, check if subflow completed
    if context.network_stack:
        if self._check_subflow_completion(context):
            logger.debug("Subflow completed, continuing in parent")
```

### Depth Limiting

The engine enforces a maximum subflow nesting depth (default 10) to prevent runaway recursion:

```python
if len(context.network_stack) >= max_subflow_depth:
    logger.error(
        "Maximum subflow depth %d exceeded when pushing to network '%s'",
        max_subflow_depth,
        push_arc.target_network
    )
    return False
```

## Multi-Transform Arcs

Both regular arcs and push arcs inherit the `transform` field from `ArcDefinition`, which supports sequential transform pipelines.

### Transform Field

```python
@dataclass
class ArcDefinition:
    transform: str | list[str] | None = None
```

- `None`: No transform; data passes through unchanged.
- `str`: A single transform function name.
- `list[str]`: A list of transform function names executed sequentially.

### Sequential Execution

When `transform` is a list, `ArcExecution.execute` normalizes it to a list and processes each transform in order. A single shared `FunctionContext` is created for all transforms in the pipeline, and each transform's output becomes the next transform's input:

```python
# From ArcExecution.execute (core/arc.py):

# Normalize to list for uniform handling
transform_names = (
    self.arc_def.transform
    if isinstance(self.arc_def.transform, list)
    else [self.arc_def.transform]
)

# Create function context with resources (shared across all transforms)
func_context = self._create_function_context(context, resources, stream_enabled)

result = data
for transform_name in transform_names:
    result = self._execute_single_transform(
        transform_name, result, func_context, stream_enabled
    )
```

### Single Transform Execution

Each individual transform is resolved and executed by `_execute_single_transform`:

1. Looks up the function in the `function_registry` (supports both `FunctionRegistry` objects and plain dicts).
2. Checks for streaming capability if `stream_enabled`.
3. Calls either `transform_func.transform(data, func_context)` (for objects with a `transform` method) or `transform_func(data, func_context)` (for plain callables).
4. Handles `ExecutionResult` return values, extracting `.data` on success or raising `FunctionError` on failure.

### Example: Multi-Transform Pipeline

```python
from dataknobs_fsm.core.arc import ArcDefinition

# Single transform
arc_single = ArcDefinition(
    target_state="validated",
    transform="validate_input"
)

# Multi-transform pipeline: normalize -> enrich -> validate
arc_pipeline = ArcDefinition(
    target_state="validated",
    transform=["normalize_text", "enrich_metadata", "validate_schema"]
)
```

With the pipeline above, data flows as:

```
input_data -> normalize_text -> enrich_metadata -> validate_schema -> result
```

Each function receives the output of the previous function and the same shared `FunctionContext`.

## Example: Parent and Child Network

Below is a complete example showing a parent network that delegates validation to a child sub-network using a `PushArc`.

### Network Configuration

```python
config = {
    "name": "OrderProcessing",
    "version": "1.0.0",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {
                    "name": "receive_order",
                    "is_start": True,
                    "arcs": [
                        {
                            "target": "validate_entry",
                            "target_network": "validation",
                            "return_state": "process_payment",
                            "data_isolation": "copy"
                        }
                    ]
                },
                {
                    "name": "process_payment",
                    "arcs": [
                        {"target": "complete"}
                    ]
                },
                {
                    "name": "complete",
                    "is_end": True
                }
            ]
        },
        {
            "name": "validation",
            "states": [
                {
                    "name": "check_inventory",
                    "is_start": True,
                    "arcs": [
                        {"target": "check_address"}
                    ]
                },
                {
                    "name": "check_address",
                    "arcs": [
                        {"target": "validation_done"}
                    ]
                },
                {
                    "name": "validation_done",
                    "is_end": True
                }
            ]
        }
    ]
}
```

### Execution Flow

```
1. Engine starts in main:receive_order
2. PushArc detected -> push to "validation" network
   - network_stack: [("validation", "process_payment")]
   - Data deep-copied (COPY isolation mode)
3. Engine enters validation:check_inventory
4. Transitions: check_inventory -> check_address -> validation_done
5. validation_done is a final state AND network_stack is non-empty
   -> _check_subflow_completion triggers _pop_subflow
   - Pops ("validation", "process_payment") from stack
   - Enters main:process_payment
6. Transitions: process_payment -> complete
7. complete is a final state, network_stack is empty -> execution complete
```

### Programmatic Construction with Data Mapping

```python
from dataknobs_fsm.core.arc import PushArc, DataIsolationMode

# Push arc with explicit data and result mapping
push_arc = PushArc(
    target_state="validate_entry",
    target_network="validation",
    return_state="process_payment",
    isolation_mode=DataIsolationMode.COPY,
    data_mapping={
        "order_items": "items",      # parent.order_items -> child.items
        "customer_id": "customer",   # parent.customer_id -> child.customer
    },
    result_mapping={
        "is_valid": "validation_passed",   # child.is_valid -> parent.validation_passed
        "errors": "validation_errors",     # child.errors -> parent.validation_errors
    },
)
```

With this configuration, when the subflow completes:
- The child network receives only the mapped fields (`items`, `customer`)
- The parent network gets back the mapped results (`validation_passed`, `validation_errors`)
- The original parent data is preserved (COPY mode) and augmented with the result mapping

### Nested Subflows

Subflows can nest to arbitrary depth (up to `max_subflow_depth`, default 10). Each push adds to the stack, and each completion pops from it:

```python
# main -> validation -> address_verification -> ... -> main
# Stack grows: [("validation", "s1"), ("address_verification", "s2")]
# Stack shrinks as each sub-network reaches its final state
```

The depth limit prevents accidental infinite recursion when networks reference each other cyclically.
