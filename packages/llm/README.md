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

Build FSM-based conversations using the conversation flow adapter:

```python
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm.fsm_integration import ConversationFlow, FlowState

# Note: Full ConversationFlow adapter coming soon
# For now, use workflow patterns directly
```

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

## Examples

The `examples/` directory contains:

- **fsm_conversation.py** - Complete FSM-based conversation system
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
│   └── storage.py         # Conversation storage
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
