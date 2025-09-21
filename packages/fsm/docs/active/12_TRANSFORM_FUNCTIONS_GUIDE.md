# Transform Functions Reference Guide

## Overview

The DataKnobs FSM supports two distinct types of data transformation functions that serve different purposes in the execution pipeline:

## StateTransforms vs ArcTransforms

| Aspect | StateTransform | ArcTransform |
|--------|----------------|--------------|
| **When** | State entry | Arc traversal |
| **Purpose** | Prepare data for state processing | Transform data between states |
| **Input** | State object | Raw data + context |
| **Configuration** | `state.functions.transform` | `arc.transform` |
| **Execution Frequency** | Once per state entry | Once per arc traversal |
| **Function Signature** | `transform(state: State) -> Dict` | `transform(data: Any, context: FunctionContext) -> Any` |

## Execution Order

```
Input Data → StateTransform → State Processing → ArcTransform → Next State
```

## Configuration Examples

### StateTransform Configuration

```yaml
states:
  - name: process_data
    functions:
      transform: 
        type: inline
        code: "lambda state: {'normalized': state.data['raw'].upper()}"
    # OR
    functions:
      transform:
        class: "custom.transformers.DataNormalizer"
```

### ArcTransform Configuration

```yaml
states:
  - name: source_state
    arcs:
      - target: target_state
        transform:
          type: inline
          code: "lambda data, ctx: {'processed': data['value'] * 2}"
        # OR
        transform:
          class: "custom.transformers.DataProcessor"
```

## Common Use Cases

### StateTransforms
- Data normalization and validation
- Format conversion (JSON to internal format)
- Adding computed fields
- Data enrichment from external sources
- Preparing data for state-specific operations

### ArcTransforms
- Data filtering and projection
- Conditional data modification
- State-specific data preparation
- Data routing and splitting
- Inter-state data flow transformations

## Implementation Notes

### Function Registry
- Both transform types are stored in the `FunctionRegistry`
- StateTransforms and ArcTransforms can use the same interface (`ITransformFunction`)
- The registry's `get_function()` method searches across all function types

### Backward Compatibility
- ArcExecution supports both `FunctionRegistry` objects and plain dictionaries
- Existing tests using dictionary-based function registries continue to work

### Test Coverage
- Comprehensive tests ensure both types execute exactly once
- Tests verify correct execution order and data flow
- Monkey patching in tests properly handles function registration

## Architecture Decision

This separation was formalized in **ADR-011: StateTransform vs ArcTransform Separation** to address:
- Confusion between the two concepts
- Failed ArcTransform execution due to incorrect function lookup
- Duplicate StateTransform execution

The fix ensures clear separation of concerns and proper execution timing for both transform types.

## Related Documentation

- `08_ARCHITECTURE_DECISIONS.md` - ADR-011 for detailed decision rationale
- `01.FSM_DESIGN.md` - Core design concepts and interfaces
- `07_IMPLEMENTATION_STATUS.md` - Implementation status and recent fixes