# FSM Architecture Decision Records (ADRs)

## ADR-001: Dual Arc Configuration Format Support

### Status
Accepted

### Context
Users naturally want to define FSM arcs at the network level using 'from' and 'to' fields, but the internal implementation expects arcs to be attached to their source states.

### Decision
Support both configuration formats:
1. Network-level arcs with 'from'/'to' fields (user-friendly)
2. State-level arcs attached to source states (internal format)

The ConfigLoader transforms network-level arcs to state-level during loading.

### Consequences
**Positive:**
- More intuitive configuration format for users
- Maintains clean internal architecture
- Backward compatible with existing configs

**Negative:**
- Additional transformation logic required
- Two formats to document and maintain

### Implementation
- `ConfigLoader._transform_network_arcs()` handles the transformation
- Both formats are validated by the same schema

---

## ADR-002: Async-First Execution Engine

### Status
Accepted

### Context
The execution engine needs to support various execution modes (single, batch, stream) and I/O-bound operations like database queries and API calls.

### Decision
Design the execution engine to be async-first with synchronous wrappers where needed.

### Consequences
**Positive:**
- Better support for I/O-bound operations
- Natural fit for stream processing
- Improved scalability for concurrent execution

**Negative:**
- Added complexity for simple use cases
- Requires careful handling of sync/async boundaries
- Learning curve for developers unfamiliar with async

### Implementation
- `ExecutionEngine.execute_async()` as primary method
- `ExecutionEngine.execute()` wraps async for compatibility
- SimpleFSM handles async/sync conversion transparently

---

## ADR-003: Resource Provider Pattern

### Status
Accepted

### Context
FSMs need to interact with various external resources (databases, caches, queues) with different lifecycle requirements.

### Decision
Use a provider pattern where resources implement a common interface and are managed by a central ResourceManager.

### Consequences
**Positive:**
- Flexible resource type support
- Consistent lifecycle management
- Easy to add new resource types
- Resource pooling and sharing

**Negative:**
- Additional abstraction layer
- More complex than direct resource access

### Implementation
- `ResourceProvider` base class defines interface
- `ResourceManager` handles registration and lifecycle
- Built-in providers for common resources

---

## ADR-004: Pydantic for Configuration Validation

### Status
Accepted

### Context
FSM configurations are complex with many interdependent fields that need validation.

### Decision
Use Pydantic models for configuration definition and validation.

### Consequences
**Positive:**
- Strong typing with automatic validation
- Clear error messages for invalid configs
- JSON Schema generation for free
- IDE support with type hints

**Negative:**
- External dependency
- Slight performance overhead
- Learning curve for Pydantic features

### Implementation
- All configuration objects are Pydantic models
- Validation happens automatically on instantiation
- Custom validators for complex rules

---

## ADR-005: Real Implementation Testing Strategy

### Status
Accepted

### Context
Complex integration between FSM components makes it difficult to test with mocks without missing real issues.

### Decision
Use real implementations with simplified backends for testing instead of extensive mocking.

### Consequences
**Positive:**
- Tests catch real integration issues
- More confidence in test results
- Tests serve as usage examples
- Fewer mock maintenance issues

**Negative:**
- Tests may run slower
- More complex test setup
- Harder to isolate failures

### Implementation
- Create simple test configurations
- Use in-memory resources for tests
- Mock only external dependencies

---

## ADR-006: SimpleFSM Abstraction Layer

### Status
Accepted

### Context
The full FSM API is powerful but complex for simple use cases.

### Decision
Provide SimpleFSM as a high-level abstraction that hides complexity while retaining full functionality.

### Consequences
**Positive:**
- Easy to get started
- Common use cases are simple
- Full power available when needed
- Good for prototyping

**Negative:**
- Another API to maintain
- Potential for abstraction leaks
- May hide important details

### Implementation
- SimpleFSM wraps core FSM components
- Provides convenient methods for common operations
- Factory functions for even simpler usage

---

## ADR-007: State-Level and Network-Level Arc Registry

### Status
Accepted

### Context
Arcs need to be accessible both from their source states and at the network level for different operations.

### Decision
Register arcs in both locations:
1. Attached to source states for traversal
2. In network.arcs registry for global operations

### Consequences
**Positive:**
- Efficient state traversal
- Easy arc querying and analysis
- Supports both local and global operations

**Negative:**
- Dual registration complexity
- Potential for inconsistency
- More memory usage

### Implementation
- FSMBuilder registers arcs in both places
- Arc IDs use "source:target" format
- Validation ensures consistency

---

## ADR-008: Function Registry with Multiple Implementations

### Status
Accepted

### Context
FSMs need to execute various types of functions (Python, JavaScript, Lambda) with different execution models.

### Decision
Use a central function registry that supports multiple function implementations through a common interface.

### Consequences
**Positive:**
- Flexible function types
- Consistent execution interface
- Easy to add new function types
- Function reuse across states

**Negative:**
- Additional abstraction
- Type safety challenges
- Debugging complexity

### Implementation
- `FunctionRegistry` manages all functions
- `BaseFunction` interface for all implementations
- Type-specific executors (Lambda, Python, JS)

---

## ADR-009: Execution Context State Management

### Status
Accepted

### Context
Execution needs to track current state, history, data, and resources throughout FSM traversal.

### Decision
Use ExecutionContext as a mutable container that tracks all execution state.

### Consequences
**Positive:**
- Central place for execution state
- Easy to pass between components
- Supports nested contexts (batch, stream)
- Good for debugging and tracing

**Negative:**
- Mutable state complexity
- Potential for race conditions
- Memory overhead

### Implementation
- ExecutionContext tracks all execution state
- Child contexts for batch/stream processing
- Context merging for aggregation

---

## ADR-010: CLI Tool with Click Framework

### Status
Accepted

### Context
Need a command-line interface for FSM operations with complex command structure.

### Decision
Use Click framework for CLI implementation with Rich for enhanced terminal output.

### Consequences
**Positive:**
- Declarative command definition
- Automatic help generation
- Good testing support
- Rich terminal UI features

**Negative:**
- External dependencies
- Learning curve for Click patterns
- Limited customization in some areas

### Implementation
- Click groups for command organization
- Rich tables and progress bars
- JSON/YAML output options
- Interactive debug mode

---

## ADR-011: StateTransform vs ArcTransform Separation

### Status
Accepted

### Context
There was confusion in the implementation about two distinct types of data transformation functions in FSMs:
1. Transforms that occur when entering a state (StateTransforms)
2. Transforms that occur during arc traversal (ArcTransforms)

This confusion led to:
- Incorrect function registration patterns
- Failed ArcTransform execution
- Unclear separation of concerns

### Decision
Clearly separate StateTransforms and ArcTransforms as distinct concepts with different:
- **Execution timing**: StateTransforms execute when entering a state, ArcTransforms execute during arc traversal
- **Function signatures**: StateTransforms receive `State` objects, ArcTransforms receive data and context
- **Purpose**: StateTransforms prepare data for state processing, ArcTransforms modify data during transitions
- **Configuration**: StateTransforms via `functions.transform`, ArcTransforms via arc `transform` property

### Consequences
**Positive:**
- Clear separation of concerns between state entry and transition processing
- Proper data flow: input → StateTransform → state processing → ArcTransform → next state
- Correct function execution timing ensures transforms are applied exactly once
- Better debugging and reasoning about data transformations

**Negative:**
- More complex mental model for users
- Need to understand when to use each type
- Additional documentation and examples required

### Implementation
- **StateTransforms**: Configured via `state.functions.transform`, executed once when entering state
- **ArcTransforms**: Configured via `arc.transform`, executed during arc traversal
- **Function Registry**: Both types stored in FunctionRegistry with proper lookup via `get_function()`
- **Execution Engine**: Passes FunctionRegistry object to ArcExecution for unified function access
- **Backward Compatibility**: ArcExecution accepts both FunctionRegistry and dict for function lookup

### Examples

**StateTransform Configuration:**
```yaml
states:
  - name: process_data
    functions:
      transform: data_normalizer  # Executes when entering state
```

**ArcTransform Configuration:**
```yaml
states:
  - name: source_state
    arcs:
      - target: target_state
        transform:
          type: inline
          code: arc_transformer  # Executes during transition
```

### Validation
- Added comprehensive test coverage for both transform types
- Fixed monkey patching in tests to ensure proper function registration
- Verified correct execution order: StateTransform → ArcTransform
- Confirmed no duplicate execution of transforms

---

## Future Architecture Considerations

### Performance Optimization
- Consider JIT compilation for hot paths
- Implement caching strategies
- Profile and optimize memory usage

### Distributed Execution
- Design for cluster deployment
- Consider state synchronization
- Plan for network partitions

### State Persistence
- Define persistence interface
- Consider event sourcing
- Plan for recovery scenarios

### Dynamic Modification
- Design for runtime FSM changes
- Consider versioning strategy
- Plan for migration paths

## Design Principles

1. **Separation of Concerns**: Keep configuration, execution, and resources separate
2. **Interface Segregation**: Small, focused interfaces over large, general ones
3. **Dependency Inversion**: Depend on abstractions, not concrete implementations
4. **Open/Closed**: Open for extension, closed for modification
5. **DRY**: Don't repeat yourself - extract common patterns
6. **YAGNI**: Don't add complexity until actually needed
7. **Fail Fast**: Validate early and provide clear errors
8. **Explicit over Implicit**: Make behavior obvious and predictable