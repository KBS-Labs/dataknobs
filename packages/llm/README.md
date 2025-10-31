# dataknobs-llm

Comprehensive LLM utilities and integrations for DataKnobs.

## Installation

```bash
pip install dataknobs-llm
```

## Features

- **LLM Abstractions**: Unified interface for OpenAI, Anthropic, Ollama, HuggingFace
- **Prompt Library System**: Template-based prompt management with validation
- **Conversation Management**: Track and manage multi-turn LLM conversations
- **FSM Integration**: Complex LLM workflows using state machines
- **Template Rendering**: Advanced template system with conditional sections
- **RAG Support**: Retrieval-augmented generation patterns

## Quick Start

### Basic LLM Usage

```python
from dataknobs_llm.llm import create_llm_provider, LLMConfig

# Create LLM provider
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.7
)
provider = create_llm_provider(config)

# Generate completion
response = provider.complete("What is the capital of France?")
print(response.content)
```

### Prompt Templates

```python
from dataknobs_llm.prompts import PromptRenderer

renderer = PromptRenderer()
result = renderer.render(
    "Hello {{name}}((, you are {{age}} years old))",
    {"name": "Alice", "age": 30}
)
print(result.content)
# Output: "Hello Alice, you are 30 years old"
```

### Conversation Management

```python
from dataknobs_llm.conversations import ConversationManager

manager = ConversationManager(conversation_id="demo")
manager.add_message(role="user", content="Hello!")
manager.add_message(role="assistant", content="Hi there!")

# Get conversation history
history = manager.get_conversation_history()
```

## FSM Integration

The LLM package provides comprehensive integration with the dataknobs-fsm state machine package, enabling complex conversational workflows and LLM-based agent systems.

### Conversation Flows

Build structured conversation flows using FSM:

```python
from dataknobs_llm.conversations.flow import (
    ConversationFlow,
    FlowState,
    keyword_condition,
    always
)

# Define conversation flow
flow = ConversationFlow(
    name="customer_support",
    initial_state="greeting",
    states={
        "greeting": FlowState(
            prompt_name="support_greeting",
            transitions={
                "need_help": "collect_issue",
                "just_browsing": "end"
            },
            transition_conditions={
                "need_help": keyword_condition(["help", "issue", "problem"]),
                "just_browsing": keyword_condition(["browse", "look"])
            }
        ),
        "collect_issue": FlowState(
            prompt_name="issue_details",
            transitions={
                "technical": "tech_support",
                "billing": "billing_support"
            },
            transition_conditions={
                "technical": keyword_condition(["bug", "error", "broken"]),
                "billing": keyword_condition(["payment", "charge", "invoice"])
            },
            max_loops=3  # Prevent infinite loops
        ),
        "tech_support": FlowState(
            prompt_name="tech_support_response",
            transitions={"done": "end"},
            transition_conditions={"done": always()}
        ),
        "billing_support": FlowState(
            prompt_name="billing_support_response",
            transitions={"done": "end"},
            transition_conditions={"done": always()}
        ),
        "end": FlowState(
            prompt_name="goodbye",
            transitions={},
            transition_conditions={}
        )
    }
)

# Execute flow with ConversationManager
async for node in manager.execute_flow(flow):
    print(f"State: {node.metadata['state']}")
    print(f"Response: {node.content}")
```

**Features:**
- State-based conversation management
- Conditional transitions based on user input
- Loop detection and prevention
- Multiple conversation paths
- Integration with prompt library

### Workflow Patterns

Build complex LLM workflows using FSM:

```python
from dataknobs_llm.fsm_integration import (
    create_rag_workflow,
    create_chain_workflow,
    LLMWorkflow
)
from dataknobs_llm.llm import LLMConfig

# Create RAG workflow
config = LLMConfig(provider="openai", model="gpt-4")
rag_flow = create_rag_workflow(
    retriever=my_retriever,
    llm_config=config,
    top_k=5
)

# Create chain-of-thought workflow
chain_flow = create_chain_workflow(
    provider="openai",
    model="gpt-4",
    steps=[
        {"name": "analyze", "prompt": "Analyze: {{input}}"},
        {"name": "synthesize", "prompt": "Synthesize: {{analyze}}"}
    ]
)
```

### FSM Resources and Functions

Use LLM resources in FSM configurations:

```python
from dataknobs_llm.fsm_integration import LLMResource, LLMProvider

# Create LLM resource for FSM
resource = LLMResource(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4",
    temperature=0.7
)
```

### Choosing Between Conversation Flows and FSM Integration

The LLM package provides two approaches for FSM-based workflows:

| Aspect | **Conversation Flows** (`conversations/flow/`) | **FSM Integration** (`fsm_integration/`) |
|--------|-----------------------------------------------|------------------------------------------|
| **Level** | High-level conversation orchestration | Low-level FSM+LLM patterns |
| **Best For** | Dialog management, multi-turn conversations | RAG pipelines, agent systems, LLM workflows |
| **Integration** | FSM + ConversationManager | Direct FSM with LLM providers |
| **Input** | `ConversationFlow` definitions | FSM configs with LLM transforms |
| **Output** | `ConversationNode` messages in conversation tree | FSM execution results |
| **State Management** | Automatic conversation history tracking | Manual state management |
| **Use Cases** | - Customer support flows<br>- Sales qualification<br>- Multi-step interviews<br>- Guided conversations | - RAG document retrieval<br>- Chain-of-thought reasoning<br>- Multi-agent orchestration<br>- Complex LLM pipelines |
| **Ease of Use** | ✅ Simpler, declarative | ⚙️ More control, requires FSM knowledge |
| **Example** | `conversation_flow_example.py` | `fsm_conversation.py` |

**Quick Decision Guide:**

- **Use Conversation Flows** when you need:
  - Structured multi-turn conversations with users
  - Intent-based routing between conversation states
  - Automatic conversation history management
  - Integration with the ConversationManager

- **Use FSM Integration** when you need:
  - Complex LLM processing pipelines (RAG, chain-of-thought)
  - Fine-grained control over state transitions
  - Custom LLM workflow patterns
  - Agent-based systems with multiple LLM calls

Both approaches use the same underlying FSM engine, so you can even combine them in the same application!

## Examples

The `examples/` directory contains:

- **conversation_flow_example.py** - Conversation flow demonstrations
  - Customer support flow with intent routing
  - Sales qualification flow
  - State transitions and conditions
  - Loop detection
- **fsm_conversation.py** - Low-level FSM-based conversation system
  - Multi-stage conversation flow
  - Intent detection and routing
  - Context management
  - Error recovery

## Module Structure

```
dataknobs_llm/
├── llm/                    # Core LLM abstractions
│   ├── base.py            # LLM provider interfaces
│   ├── providers.py       # OpenAI, Anthropic, Ollama, HuggingFace
│   └── utils.py           # Utilities and helpers
├── prompts/               # Prompt library system
│   ├── base/              # Base types and interfaces
│   ├── builders/          # Prompt builders
│   └── rendering/         # Template rendering
├── conversations/         # Conversation management
│   ├── manager.py         # ConversationManager
│   ├── storage.py         # Conversation storage
│   └── flow/              # Conversation flows (NEW)
│       ├── flow.py        # ConversationFlow and FlowState
│       ├── adapter.py     # FSM adapter
│       └── conditions.py  # Transition conditions
└── fsm_integration/       # FSM integration (migrated from dataknobs-fsm)
    ├── workflows.py       # RAG, chain-of-thought, agent patterns
    ├── resources.py       # LLM resources for FSM
    └── functions.py       # LLM function library
```

## Migration Note

The `fsm_integration` module contains code that was previously in the `dataknobs-fsm` package. It has been moved here to consolidate all LLM functionality in one package and eliminate duplication.

**Old imports** (from dataknobs-fsm):
```python
from dataknobs_fsm.llm import LLMProvider, LLMConfig
from dataknobs_fsm.patterns.llm_workflow import RAGWorkflow
```

**New imports** (from dataknobs-llm):
```python
from dataknobs_llm.llm import LLMProvider, LLMConfig
from dataknobs_llm.fsm_integration import LLMWorkflow, create_rag_workflow
```

## Documentation

For more information, see:
- FSM Package: `packages/fsm/README.md` for general FSM functionality
- Examples: `packages/llm/examples/` for working code samples

## Testing

Run the tests with:

```bash
cd packages/llm
uv run pytest tests/ -v
```

## Dependencies

- dataknobs-common
- dataknobs-fsm (for FSM integration)
- anthropic (optional, for Anthropic provider)
- openai (optional, for OpenAI provider)

## License

See LICENSE file in the root repository.
