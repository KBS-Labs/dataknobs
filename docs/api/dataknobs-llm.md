# dataknobs-llm API Reference

## Overview

The `dataknobs-llm` package provides a unified interface for working with different LLM providers (OpenAI, Anthropic, Ollama, HuggingFace), along with advanced prompt management, conversation tracking, and tool integration.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/llm.md) - Full auto-generated reference
> - [Detailed API Reference](../packages/llm/api/llm.md) - Curated API docs with examples
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/llm/src/dataknobs_llm) - Browse on GitHub
> - [Package Guide](../packages/llm/index.md) - Detailed documentation

## LLM Provider

### Creating an LLM Provider

**Source:** [`llm/base.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/llm/base.py)

```python
from dataknobs_llm import create_llm_provider

# Using factory function
llm = create_llm_provider(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Or with config dict
llm = create_llm_provider(config={
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 1000
})

# Using LLMConfig dataclass
from dataknobs_llm.llm import LLMConfig

config = LLMConfig(
    provider="ollama",
    model="llama3.2:3b",
    temperature=0.7,
    max_tokens=500
)
llm = create_llm_provider(config)
```

### Generating Completions

The primary method is `complete()`, which accepts either a string or a list of `LLMMessage` objects.

```python
from dataknobs_llm import create_llm_provider
from dataknobs_llm.llm import LLMMessage

llm = create_llm_provider(provider="openai", model="gpt-4")

# Simple string completion
response = await llm.complete("What is the capital of France?")
print(response.content)  # "Paris is the capital of France."
print(response.usage)    # {'prompt_tokens': 8, 'completion_tokens': 7, 'total_tokens': 15}

# Multi-turn conversation with messages
messages = [
    LLMMessage(role="system", content="You are a helpful physics tutor"),
    LLMMessage(role="user", content="Explain quantum computing"),
    LLMMessage(role="assistant", content="Quantum computing uses quantum mechanics..."),
    LLMMessage(role="user", content="Can you give a simple example?")
]
response = await llm.complete(messages)

# With parameters
response = await llm.complete(
    "Write a haiku about coding",
    temperature=0.9,
    max_tokens=100
)
```

### Streaming Completions

Use `stream_complete()` for streaming responses:

```python
# Stream a response
async for chunk in llm.stream_complete("Tell me a story about robots"):
    print(chunk.delta, end="", flush=True)

# Stream with messages
messages = [
    LLMMessage(role="system", content="You are a storyteller"),
    LLMMessage(role="user", content="Tell me about space exploration")
]
async for chunk in llm.stream_complete(messages):
    print(chunk.delta, end="", flush=True)
```

### Embeddings

Use `embed()` for generating vector embeddings:

```python
# Single text embedding
embedding = await llm.embed("This is a sample text")
print(f"Embedding dimension: {len(embedding)}")

# Multiple texts
texts = ["First document", "Second document", "Third document"]
embeddings = await llm.embed(texts)
```

## Prompt Management

The prompt system uses a builder pattern with libraries and resource adapters.

### PromptBuilder

**Source:** [`prompts/builders/`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/prompts/builders/)

The `PromptBuilder` and `AsyncPromptBuilder` classes provide a flexible system for rendering prompts with variable substitution.

```python
from dataknobs_llm.prompts import (
    PromptBuilder,
    AsyncPromptBuilder,
    FileSystemPromptLibrary,
    DictResourceAdapter
)
from pathlib import Path

# Create a filesystem prompt library
library = FileSystemPromptLibrary(prompt_dir=Path("prompts/"))

# Create resource adapters for variable substitution
config_adapter = DictResourceAdapter({
    "app_name": "DataKnobs",
    "version": "1.0.0"
})

# Create builder with adapters
builder = PromptBuilder(
    library=library,
    adapters={'config': config_adapter}
)

# Render a prompt
result = builder.render_user_prompt(
    'greeting_template',
    params={'user_name': 'Alice'}
)
print(result.content)
```

### Prompt Libraries

**Source:** [`prompts/implementations/`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/prompts/implementations/)

Different prompt storage backends:

```python
from dataknobs_llm.prompts import (
    FileSystemPromptLibrary,
    ConfigPromptLibrary,
    CompositePromptLibrary,
    VersionedPromptLibrary
)
from pathlib import Path

# Filesystem library - loads prompts from files
fs_library = FileSystemPromptLibrary(
    prompt_dir=Path("prompts/"),
    file_extension=".txt"
)

# Config library - loads from config dict
config_library = ConfigPromptLibrary(config={
    "prompts": {
        "greeting": {
            "system": "You are {{assistant_name}}",
            "user": "Hello, {{user_name}}!"
        }
    }
})

# Composite library - combines multiple libraries
composite = CompositePromptLibrary(
    libraries=[fs_library, config_library],
    priority_order=["filesystem", "config"]
)

# Versioned library - supports prompt versioning
versioned = VersionedPromptLibrary(base_library=fs_library)
```

### Template Syntax

Templates use `{{variables}}` for substitution and `((conditionals))` for conditional content:

```python
# In your prompt file (e.g., prompts/analyze_code.txt):
"""
System: You are a {{language}} code analyzer.

((if:has_context))
Context: {{context}}
((endif))

User: Analyze the following code:
{{code}}

((if:include_suggestions))
Provide improvement suggestions.
((endif))
"""

# Usage:
result = builder.render_user_prompt(
    'analyze_code',
    params={
        'language': 'Python',
        'code': 'def hello(): print("hi")',
        'has_context': True,
        'context': 'This is a greeting function',
        'include_suggestions': True
    }
)
```

### Versioning and A/B Testing

**Source:** [`prompts/versioning/`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/prompts/versioning/)

```python
from dataknobs_llm.prompts import (
    VersionManager,
    ABTestManager,
    PromptVersion,
    VersionStatus
)

# Version management
version_manager = VersionManager()

# Create versions
v1 = PromptVersion(
    name="summarize",
    version="1.0.0",
    template="Summarize: {{text}}",
    status=VersionStatus.ACTIVE
)

v2 = PromptVersion(
    name="summarize",
    version="2.0.0",
    template="Provide a concise summary:\n\n{{text}}",
    status=VersionStatus.TESTING
)

version_manager.register_version(v1)
version_manager.register_version(v2)

# A/B testing
ab_manager = ABTestManager()

experiment = ab_manager.create_experiment(
    name="summary_test",
    prompt_name="summarize",
    variants=["1.0.0", "2.0.0"],
    traffic_split=[0.5, 0.5]  # 50/50 split
)

# Get variant for user
variant = ab_manager.get_variant("summary_test", user_id="user-123")
```

## Conversation Management

The conversation system uses a tree-based structure for supporting branching and message history.

### ConversationState

**Source:** [`conversations/storage.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/conversations/storage.py)

```python
from dataknobs_llm.conversations import (
    ConversationState,
    ConversationNode,
    DataknobsConversationStorage
)
from dataknobs_llm.llm import LLMMessage
from dataknobs_structures.tree import Tree
from dataknobs_data.backends import AsyncMemoryDatabase

# Create conversation nodes
root_node = ConversationNode(
    message=LLMMessage(role="system", content="You are a helpful assistant"),
    node_id=""
)

# Create conversation tree
tree = Tree(root_node)

# Create conversation state
state = ConversationState(
    conversation_id="conv-123",
    message_tree=tree,
    current_node_id="",
    metadata={"user_id": "user-456", "session": "web-001"}
)

# Add messages to the tree
user_msg_node = ConversationNode(
    message=LLMMessage(role="user", content="What is Python?"),
    node_id="msg-1"
)
tree.add_child("", user_msg_node)

assistant_msg_node = ConversationNode(
    message=LLMMessage(role="assistant", content="Python is a programming language..."),
    node_id="msg-2"
)
tree.add_child("msg-1", assistant_msg_node)

# Update current node
state.current_node_id = "msg-2"
```

### ConversationStorage

**Source:** [`conversations/storage.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/conversations/storage.py)

```python
from dataknobs_llm.conversations import DataknobsConversationStorage
from dataknobs_data.backends import AsyncMemoryDatabase, AsyncPostgresDatabase

# Using memory backend (for testing/development)
storage = DataknobsConversationStorage(AsyncMemoryDatabase())

# Save conversation
await storage.save_conversation(state)

# Load conversation
loaded_state = await storage.load_conversation("conv-123")
if loaded_state:
    print(f"Loaded conversation with {len(loaded_state.message_tree)} nodes")

# Delete conversation
deleted = await storage.delete_conversation("conv-123")

# Using PostgreSQL backend (for production)
pg_backend = AsyncPostgresDatabase(
    connection_string="postgresql://localhost/conversations"
)
pg_storage = DataknobsConversationStorage(pg_backend)
```

### ConversationManager

**Source:** [`conversations/manager.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/conversations/manager.py)

```python
from dataknobs_llm.conversations import ConversationManager
from dataknobs_llm import create_llm_provider

# Create LLM provider
llm = create_llm_provider(provider="openai", model="gpt-4")

# Create conversation manager
manager = ConversationManager(
    llm=llm,
    storage=storage,
    system_message="You are a helpful coding assistant"
)

# Start or continue conversation
response = await manager.send_message(
    conversation_id="conv-123",
    user_message="How do I use async/await in Python?"
)
print(response.content)

# Get conversation history
messages = await manager.get_history("conv-123")
```

## Tools and Function Calling

**Source:** [`tools/base.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/tools/base.py)

```python
from dataknobs_llm.tools import Tool, ToolRegistry

# Define a tool function
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather for a location.

    Args:
        location: City name
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information dictionary
    """
    # Implementation would call weather API
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "conditions": "Sunny"
    }

# Create tool
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    func=get_weather,
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    }
)

# Use with LLM (provider-specific implementation)
# Note: Tool calling syntax varies by provider
response = await llm.complete(
    "What's the weather in Paris?",
    tools=[weather_tool]
)
```

### ToolRegistry

```python
from dataknobs_llm.tools import ToolRegistry

# Create registry
registry = ToolRegistry()

# Register tools
registry.register(weather_tool)
registry.register(Tool(
    name="calculate",
    description="Evaluate mathematical expressions",
    func=lambda expr: eval(expr),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate"
            }
        },
        "required": ["expression"]
    }
))

# Get tool by name
tool = registry.get("get_weather")

# List all tools
all_tools = registry.list_tools()
```

## Full Example

Here's a complete example combining LLM provider, prompts, conversations, and tools:

```python
import asyncio
from pathlib import Path
from dataknobs_llm import create_llm_provider
from dataknobs_llm.llm import LLMMessage
from dataknobs_llm.prompts import (
    PromptBuilder,
    FileSystemPromptLibrary,
    DictResourceAdapter
)
from dataknobs_llm.conversations import (
    ConversationState,
    ConversationNode,
    DataknobsConversationStorage,
    ConversationManager
)
from dataknobs_llm.tools import Tool
from dataknobs_structures.tree import Tree
from dataknobs_data.backends import AsyncMemoryDatabase


async def main():
    # Setup LLM provider
    llm = create_llm_provider(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    )

    # Setup prompt system
    prompt_library = FileSystemPromptLibrary(
        prompt_dir=Path("prompts/")
    )

    config_adapter = DictResourceAdapter({
        "assistant_name": "CodeHelper",
        "version": "1.0"
    })

    prompt_builder = PromptBuilder(
        library=prompt_library,
        adapters={'config': config_adapter}
    )

    # Setup conversation storage
    backend = AsyncMemoryDatabase()
    storage = DataknobsConversationStorage(backend)

    # Define tools
    def search_docs(query: str) -> str:
        """Search documentation."""
        return f"Documentation results for: {query}"

    def run_code(code: str, language: str = "python") -> dict:
        """Execute code safely."""
        return {"output": "Code executed successfully", "language": language}

    tools = [
        Tool(
            name="search_docs",
            description="Search documentation",
            func=search_docs,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="run_code",
            description="Execute code",
            func=run_code,
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "description": "Programming language"}
                },
                "required": ["code"]
            }
        )
    ]

    # Create conversation manager
    manager = ConversationManager(
        llm=llm,
        storage=storage,
        system_message="You are a helpful coding assistant with access to documentation and code execution."
    )

    # Interactive conversation
    conversation_id = "coding-session-001"

    print("Coding Assistant (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # Send message with tools
        response = await manager.send_message(
            conversation_id=conversation_id,
            user_message=user_input
        )

        print(f"\nAssistant: {response.content}")

        # Show token usage
        if response.usage:
            print(f"(Tokens: {response.usage.get('total_tokens', 'N/A')})")

    # Save final conversation state
    print("\nSaving conversation...")
    conversation_state = await storage.load_conversation(conversation_id)
    if conversation_state:
        print(f"Conversation saved with {len(conversation_state.message_tree)} messages")


if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features

### Resource Adapters

Resource adapters provide data for prompt variable substitution:

```python
from dataknobs_llm.prompts import (
    DictResourceAdapter,
    DataknobsBackendAdapter,
    InMemoryAdapter
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Dictionary adapter
dict_adapter = DictResourceAdapter({
    "key1": "value1",
    "key2": "value2"
})

# Dataknobs backend adapter (async)
backend = AsyncMemoryDatabase()
backend_adapter = DataknobsBackendAdapter(
    backend=backend,
    collection="config"
)

# In-memory adapter with search
memory_adapter = InMemoryAdapter()
await memory_adapter.set("setting1", "value1")
value = await memory_adapter.get("setting1")
```

### Conversation Middleware

Add processing layers to conversations:

```python
from dataknobs_llm.conversations import (
    LoggingMiddleware,
    ContentFilterMiddleware,
    ValidationMiddleware,
    MetadataMiddleware
)

# Create middleware stack
middlewares = [
    LoggingMiddleware(),
    ContentFilterMiddleware(banned_words=["spam", "abuse"]),
    ValidationMiddleware(max_message_length=5000),
    MetadataMiddleware(default_metadata={"app": "chatbot"})
]

# Use with conversation manager
manager = ConversationManager(
    llm=llm,
    storage=storage,
    middlewares=middlewares
)
```

## Usage Examples

For detailed usage examples, see the package documentation and examples directory.

## Provider Support

Supported LLM providers:

- **OpenAI**: GPT-4, GPT-3.5, embeddings
- **Anthropic**: Claude 3 family (Opus, Sonnet, Haiku)
- **Ollama**: Local models (Llama, Mistral, etc.)
- **HuggingFace**: Inference API models
- **Echo**: Testing provider that echoes input

Each provider implements the same interface (`complete`, `stream_complete`, `embed`) for consistent usage across different backends.
