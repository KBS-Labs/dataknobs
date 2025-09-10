# FSM API Reference

The FSM package provides two main APIs for different use cases:

## SimpleFSM API

The [SimpleFSM](simple.md) class provides a high-level, user-friendly interface for creating and running finite state machines. It's perfect for:

- Quick prototyping
- Simple workflows
- Learning the FSM concepts
- Scripts and small applications

**Key Features:**
- Fluent API for building FSMs
- Automatic resource management
- Built-in error handling
- Support for both sync and async execution

[View SimpleFSM Documentation →](simple.md)

## AdvancedFSM API

The [AdvancedFSM](advanced.md) class offers fine-grained control over FSM execution and advanced features. Use it for:

- Production applications
- Complex workflows
- Performance-critical systems
- Custom execution strategies

**Key Features:**
- Full control over execution engines
- Custom resource providers
- Advanced data modes (COPY, REFERENCE, DIRECT)
- Network-based FSM composition
- Transaction support
- Streaming capabilities

[View AdvancedFSM Documentation →](advanced.md)

## Quick Comparison

| Feature | SimpleFSM | AdvancedFSM |
|---------|-----------|--------------|
| **Ease of Use** | Very Easy | Moderate |
| **Configuration Support** | Yes | Yes |
| **Async Support** | Yes | Yes |
| **Batch Processing** | Yes | Yes |
| **Stream Processing** | Limited | Full |
| **Resource Management** | Automatic | Manual/Custom |
| **Data Modes** | Automatic | Configurable |
| **Network Support** | No | Yes |
| **Transaction Support** | Basic | Advanced |
| **Performance** | Good | Excellent |
| **Customization** | Limited | Extensive |

## Core Components

Both APIs build on these core components:

### States
Represent discrete points in the FSM workflow:
- **Initial State**: Entry point of the FSM
- **Terminal State**: Exit points of the FSM
- **Intermediate States**: Processing stages

### Arcs (Transitions)
Define connections between states:
- **Source**: Starting state
- **Target**: Destination state
- **Function**: Processing logic
- **Condition**: Optional guard condition
- **Metadata**: Additional configuration

### Execution Context
Manages FSM execution state:
- Current state
- Data payload
- Execution history
- Resource references

### Resources
External services and connections:
- Databases
- File systems
- HTTP clients
- LLM providers
- Custom resources

## Usage Examples

### Simple Workflow (SimpleFSM)

```python
from dataknobs_fsm import SimpleFSM

fsm = SimpleFSM()
fsm.add_state("start", initial=True)
fsm.add_state("end", terminal=True)
fsm.add_transition("start", "end", 
    function=lambda x: {"result": x["value"] * 2})

result = fsm.run({"value": 5})
# {"result": 10}
```

### Advanced Workflow (AdvancedFSM)

```python
from dataknobs_fsm import AdvancedFSM
from dataknobs_fsm.core import DataMode

fsm = AdvancedFSM(
    name="advanced_workflow",
    data_mode=DataMode.REFERENCE  # Efficient for large data
)

# Configure resources
fsm.add_resource("db", {
    "type": "database",
    "provider": "postgresql",
    "config": {...}
})

# Build complex workflow
fsm.add_network("main", states=[...], arcs=[...])

# Execute with custom engine
from dataknobs_fsm.execution import AsyncEngine
engine = AsyncEngine(max_concurrent=10)
result = await fsm.execute({"input": "data"}, engine=engine)
```

## API Stability

The FSM package follows semantic versioning:

- **SimpleFSM**: Stable API, backward compatible within major versions
- **AdvancedFSM**: Stable core API, some advanced features may evolve
- **Core Components**: Very stable, minimal changes expected

## Getting Help

- Check the [Examples](../examples/index.md) for real-world usage
- Read the [Patterns Guide](../patterns/index.md) for common scenarios
- Review the [Guides](../guides/index.md) for specific topics
- See the [FAQ](../faq.md) for common questions