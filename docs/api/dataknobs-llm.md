# dataknobs-llm API Reference

## Overview

The `dataknobs-llm` package provides a unified interface for working with different LLM providers (OpenAI, Anthropic, Ollama, HuggingFace), along with advanced prompt management, conversation tracking, and tool integration.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/llm.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/llm/src/dataknobs_llm) - Browse on GitHub
> - [Package Guide](../packages/llm/index.md) - Detailed documentation

## LLM Provider

### Creating an LLM Provider

**Source:** [`llm/base.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/llm/base.py)

```python
from dataknobs_llm import create_llm_provider

# The factory takes ONE config argument -- an LLMConfig, a Config, or a plain
# dict -- and does not accept the config's fields as keywords of its own.
llm = create_llm_provider({
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-api-key",
})

# Named, if you prefer
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

llm = create_llm_provider({"provider": "openai", "model": "gpt-4"})

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

# Filesystem library - loads prompts from files. The parameter is plural and
# takes a list, because a directory may hold more than one extension.
fs_library = FileSystemPromptLibrary(
    prompt_dir=Path("prompts/"),
    file_extensions=[".txt"]
)

# Config library - the dict is keyed by prompt TYPE at the top level
# ("system", "user", "messages", "rag"), then by prompt name.
config_library = ConfigPromptLibrary({
    "system": {
        "greeting": {"template": "You are {{assistant_name}}"}
    },
    "user": {
        "greeting": {"template": "Hello, {{user_name}}!"}
    }
})

# Composite library - the LIST ORDER is the priority: the first library
# holding a prompt wins. `names` only labels them for logging.
composite = CompositePromptLibrary(
    libraries=[config_library, fs_library],
    names=["config", "filesystem"]
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

Both managers are async, and both persist as they go -- so a version is
created through the manager rather than constructed and then registered.
`PromptVersion` itself requires a `version_id`, which is exactly what
`create_version` mints for you.

```python
from dataknobs_llm.prompts import (
    VersionManager,
    ABTestManager,
    VersionStatus,
)
from dataknobs_llm.prompts.versioning.types import PromptVariant

# Version management
version_manager = VersionManager()

# Create versions. `prompt_type` is required alongside the name, because a
# name is only unique within a type. It is a free-form label, not a checked
# vocabulary; the library's own accessors look for "system" and "user"
# (`list_system_prompts` / `list_user_prompts`), and `ConfigPromptLibrary`
# spells its message section "messages".
v1 = await version_manager.create_version(
    name="summarize",
    prompt_type="user",
    version="1.0.0",
    template="Summarize: {{text}}",
    status=VersionStatus.ACTIVE,
)

v2 = await version_manager.create_version(
    name="summarize",
    prompt_type="user",
    version="2.0.0",
    template="Provide a concise summary:\n\n{{text}}",
    # DRAFT | ACTIVE | PRODUCTION | DEPRECATED | ARCHIVED -- a version still
    # being trialled is a DRAFT; there is no TESTING.
    status=VersionStatus.DRAFT,
)

print(v1.version_id)    # a generated uuid4, not the semantic version

# A/B testing
ab_manager = ABTestManager()

# Variants are PromptVariant objects carrying their own weight, and the
# experiment is keyed by the prompt's name and type rather than a separate
# test name. traffic_split is derived from the weights when omitted.
experiment = await ab_manager.create_experiment(
    name="summarize",
    prompt_type="user",
    variants=[
        PromptVariant(version="1.0.0", weight=0.5),
        PromptVariant(version="2.0.0", weight=0.5),
    ],
)
print(experiment.traffic_split)   # {'1.0.0': 0.5, '2.0.0': 0.5}

# Get variant for user -- addressed by the minted experiment_id, and sticky:
# the same user keeps the same variant.
variant = await ab_manager.get_variant_for_user(experiment.experiment_id, "user-123")
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
from dataknobs_data.backends import AsyncMemoryDatabase
from dataknobs_data.backends.postgres import AsyncPostgresDatabase

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
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

# Create LLM provider
llm = create_llm_provider({"provider": "openai", "model": "gpt-4"})

# Prompt builder -- `storage` is the one built in the section above
library = ConfigPromptLibrary({
    "system": {"assistant": {"template": "You are a helpful {{topic}} assistant."}},
})
builder = AsyncPromptBuilder(library=library)

# Create conversation manager. `create()` is async because it loads the
# conversation if `conversation_id` names one that already exists; the three
# collaborators -- llm, prompt_builder, storage -- are all required.
#
# The system message comes from the prompt library by name rather than as a
# literal: that is what makes it versionable and parameterised.
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="assistant",
    system_params={"topic": "coding"},
    conversation_id="conv-123",
)

# A turn is two steps: append the user's message, then complete.
await manager.add_message(
    role="user",
    content="How do I use async/await in Python?",
)
response = await manager.complete()
print(response.content)

# Get conversation history -- the manager holds one conversation, so this
# takes no id. It answers LLMMessage objects along the current branch.
messages = await manager.get_history()
print([m.role for m in messages])   # ['system', 'user', 'assistant']
```

## Tools and Function Calling

**Source:** [`tools/base.py`](https://github.com/kbs-labs/dataknobs/blob/main/packages/llm/src/dataknobs_llm/tools/base.py)

`Tool` is abstract: a tool is a subclass supplying a `schema` property and an
async `execute`, with the name and description passed up to `super().__init__`.
There is no function-wrapping constructor -- the callable IS `execute`.

```python
from typing import Any, Dict

from dataknobs_llm.tools import Tool


class WeatherTool(Tool):
    """Get the current weather for a location."""

    def __init__(self) -> None:
        super().__init__(
            name="get_weather",
            description="Get current weather for a location",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        """JSON Schema for the parameters -- what `parameters` would have been."""
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        }

    async def execute(self, location: str, unit: str = "celsius") -> dict:
        # Implementation would call a weather API
        return {
            "location": location,
            "temperature": 22,
            "unit": unit,
            "conditions": "Sunny",
        }


weather_tool = WeatherTool()

# The provider wire format is derived from the schema, not hand-written.
print(weather_tool.to_function_definition()["name"])   # get_weather

# Use with LLM (provider-specific implementation)
# Note: Tool calling syntax varies by provider
response = await llm.complete(
    "What's the weather in Paris?",
    tools=[weather_tool]
)
```

### ToolRegistry

```python
from typing import Any, Dict

from dataknobs_llm.tools import Tool, ToolRegistry


class CalculateTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="calculate",
            description="Evaluate mathematical expressions",
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    async def execute(self, expression: str) -> float:
        # A real implementation parses rather than eval()s.
        return float(expression)


# Create registry
registry = ToolRegistry()

# Register tools. `register_tool` takes the tool alone and reads its name;
# the generic `register(key, item)` inherited from Registry wants both.
registry.register_tool(weather_tool)
registry.register_tool(CalculateTool())

# Get tool by name -- `get_tool`, again the tool-aware form of `get(key)`.
tool = registry.get_tool("get_weather")

# List all tools
all_tools = registry.list_tools()
print(registry.get_tool_names())        # ['get_weather', 'calculate']

# The registry also runs them, which is what the provider loop calls.
result = await registry.execute_tool("get_weather", location="Paris")
print(result["conditions"])             # Sunny
```

## Full Example

Here's a complete example combining LLM provider, prompts, conversations, and tools:

```python
import asyncio
from pathlib import Path
from dataknobs_llm import create_llm_provider
from dataknobs_llm.llm import LLMMessage
from dataknobs_llm.prompts import (
    AsyncPromptBuilder,
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
    llm = create_llm_provider({
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
    })

    # Setup prompt system
    prompt_library = FileSystemPromptLibrary(
        prompt_dir=Path("prompts/")
    )

    config_adapter = DictResourceAdapter({
        "assistant_name": "CodeHelper",
        "version": "1.0"
    })

    # ConversationManager takes the ASYNC builder; PromptBuilder is its
    # synchronous twin, for code that is not on an event loop.
    prompt_builder = AsyncPromptBuilder(
        library=prompt_library,
        adapters={'config': config_adapter}
    )

    # Setup conversation storage
    backend = AsyncMemoryDatabase()
    storage = DataknobsConversationStorage(backend)

    # Define tools -- each is a Tool subclass; the schema is a property and
    # the body goes in the async execute().
    class SearchDocsTool(Tool):
        def __init__(self) -> None:
            super().__init__(name="search_docs", description="Search documentation")

        @property
        def schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            }

        async def execute(self, query: str) -> str:
            return f"Documentation results for: {query}"

    class RunCodeTool(Tool):
        def __init__(self) -> None:
            super().__init__(name="run_code", description="Execute code")

        @property
        def schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                    },
                },
                "required": ["code"],
            }

        async def execute(self, code: str, language: str = "python") -> dict:
            return {"output": "Code executed successfully", "language": language}

    tools = [SearchDocsTool(), RunCodeTool()]

    # Create conversation manager. The system message is a named prompt from
    # the library, not a literal keyword.
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=prompt_builder,
        storage=storage,
        system_prompt_name="coding_assistant",
        conversation_id="coding-session-001",
    )

    print("Coding Assistant (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # A turn is append-then-complete; tools are passed to complete()
        await manager.add_message(role="user", content=user_input)
        response = await manager.complete(tools=tools)

        print(f"\nAssistant: {response.content}")

        # Show token usage
        if response.usage:
            print(f"(Tokens: {response.usage.get('total_tokens', 'N/A')})")

    # Save final conversation state
    print("\nSaving conversation...")
    conversation_state = await storage.load_conversation(manager.conversation_id)
    if conversation_state:
        # The state holds a tree; get_all_nodes() flattens it.
        print(f"Conversation saved with {len(conversation_state.get_all_nodes())} messages")


if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Features

### Resource Adapters

Resource adapters provide data for prompt variable substitution:

```python
from dataknobs_llm.prompts import (
    DictResourceAdapter,
    AsyncDataknobsBackendAdapter,
    InMemoryAdapter
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Dictionary adapter
dict_adapter = DictResourceAdapter({
    "key1": "value1",
    "key2": "value2"
})

# Dataknobs backend adapter. The parameter is `database`, and there are two
# classes rather than one flag: the async adapter takes an AsyncDatabase, the
# sync DataknobsBackendAdapter takes a SyncDatabase.
backend = AsyncMemoryDatabase()
backend_adapter = AsyncDataknobsBackendAdapter(
    database=backend,
    text_field="content",
)

# In-memory adapter -- its contents are fixed at construction; it is a source
# for templates to read, not a store to write into.
memory_adapter = InMemoryAdapter(
    data={"setting1": "value1"},
    search_results=[{"content": "a document the template can cite"}],
)
value = memory_adapter.get_value("setting1")   # 'value1'
hits = memory_adapter.search("anything", k=1)
```

### Conversation Middleware

Add processing layers to conversations:

```python
from dataknobs_llm.conversations import (
    ConversationManager,
    LoggingMiddleware,
    ContentFilterMiddleware,
    MetadataMiddleware,
    RateLimitMiddleware,
)

# Create middleware stack (onion order: first item wraps outermost)
middleware = [
    LoggingMiddleware(),
    RateLimitMiddleware(max_requests=60, window_seconds=60),
    ContentFilterMiddleware(filter_words=["spam", "abuse"]),
    MetadataMiddleware(request_metadata={"app": "chatbot"}),
]

# Use with conversation manager
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=middleware,
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
