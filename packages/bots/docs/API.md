# API Reference

Complete API documentation for DataKnobs Bots.

## Table of Contents

- [Core Classes](#core-classes)
  - [DynaBot](#dynabot)
  - [BotContext](#botcontext)
  - [BotRegistry](#botregistry)
- [Memory](#memory)
  - [Memory (Base)](#memory-base)
  - [BufferMemory](#buffermemory)
  - [VectorMemory](#vectormemory)
- [Knowledge Base](#knowledge-base)
  - [RAGKnowledgeBase](#ragknowledgebase)
- [Reasoning](#reasoning)
  - [ReasoningStrategy (Base)](#reasoningstrategy-base)
  - [SimpleReasoning](#simplereasoning)
  - [ReActReasoning](#reactreasoning)
- [Tools](#tools)
  - [KnowledgeSearchTool](#knowledgesearchtool)
- [Factory Functions](#factory-functions)

---

## Core Classes

### DynaBot

The main bot class that orchestrates all components.

#### Class Definition

```python
class DynaBot:
    """Configuration-driven chatbot leveraging the DataKnobs ecosystem."""
```

#### Constructor

```python
def __init__(
    self,
    llm: AsyncLLMProvider,
    prompt_builder: AsyncPromptBuilder,
    conversation_storage: DataknobsConversationStorage,
    tool_registry: ToolRegistry | None = None,
    memory: Memory | None = None,
    knowledge_base: Any | None = None,
    reasoning_strategy: Any | None = None,
    middleware: list[Any] | None = None,
    system_prompt_name: str | None = None,
    default_temperature: float = 0.7,
    default_max_tokens: int = 1000,
)
```

**Parameters:**
- `llm` (AsyncLLMProvider): LLM provider instance for generating responses
- `prompt_builder` (AsyncPromptBuilder): Prompt builder for managing prompts
- `conversation_storage` (DataknobsConversationStorage): Storage backend for conversations
- `tool_registry` (ToolRegistry, optional): Registry of available tools
- `memory` (Memory, optional): Memory implementation for context management
- `knowledge_base` (Any, optional): Knowledge base for RAG
- `reasoning_strategy` (Any, optional): Reasoning strategy for multi-step reasoning
- `middleware` (list[Any], optional): List of middleware for request/response processing
- `system_prompt_name` (str, optional): Name of the system prompt to use
- `default_temperature` (float, optional): Default temperature for generation (default: 0.7)
- `default_max_tokens` (int, optional): Default max tokens for generation (default: 1000)

**Example:**
```python
from dataknobs_bots import DynaBot
from dataknobs_llm.llm import LLMProviderFactory
from dataknobs_llm.prompts import AsyncPromptBuilder

# Create components
llm = LLMProviderFactory(is_async=True).create({"provider": "ollama", "model": "gemma3:3b"})
await llm.initialize()

prompt_builder = AsyncPromptBuilder()
storage = MemoryConversationStorage()

# Create bot
bot = DynaBot(
    llm=llm,
    prompt_builder=prompt_builder,
    conversation_storage=storage,
    default_temperature=0.7
)
```

#### Class Methods

##### `from_config`

Create a DynaBot from configuration dictionary.

```python
@classmethod
async def from_config(cls, config: dict[str, Any]) -> "DynaBot"
```

**Parameters:**
- `config` (dict): Configuration dictionary

**Returns:**
- `DynaBot`: Configured bot instance

**Configuration Schema:**
```python
{
    "llm": {
        "provider": str,      # LLM provider (e.g., "ollama", "openai")
        "model": str,         # Model name
        "temperature": float, # Optional, default: 0.7
        "max_tokens": int,    # Optional, default: 1000
    },
    "conversation_storage": {
        "backend": str,       # Storage backend (e.g., "memory", "postgres")
        # ... backend-specific options
    },
    "tools": list,            # Optional tool configurations
    "memory": dict,           # Optional memory configuration
    "knowledge_base": dict,   # Optional knowledge base configuration
    "reasoning": dict,        # Optional reasoning configuration
    "middleware": list,       # Optional middleware configurations
    "prompts": dict,          # Optional prompts library
    "system_prompt": dict | str,  # Optional system prompt configuration
}
```

**Example:**
```python
config = {
    "llm": {
        "provider": "ollama",
        "model": "gemma3:3b",
        "temperature": 0.7
    },
    "conversation_storage": {
        "backend": "memory"
    },
    "memory": {
        "type": "buffer",
        "max_messages": 10
    }
}

bot = await DynaBot.from_config(config)
```

#### Instance Methods

##### `chat`

Process a chat message and generate a response.

```python
async def chat(
    self,
    message: str,
    context: BotContext,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> str
```

**Parameters:**
- `message` (str): User message to process
- `context` (BotContext): Bot execution context
- `temperature` (float, optional): Temperature override for this request
- `max_tokens` (int, optional): Max tokens override for this request
- `stream` (bool, optional): Whether to stream the response (default: False)
- `**kwargs`: Additional arguments

**Returns:**
- `str`: Bot response

**Raises:**
- `Exception`: If LLM generation fails

**Example:**
```python
from dataknobs_bots import BotContext

context = BotContext(
    conversation_id="conv-123",
    client_id="client-456",
    user_id="user-789"
)

response = await bot.chat(
    message="What is the weather today?",
    context=context,
    temperature=0.8
)

print(response)
```

#### Attributes

- `llm` (AsyncLLMProvider): LLM provider instance
- `prompt_builder` (AsyncPromptBuilder): Prompt builder instance
- `conversation_storage` (DataknobsConversationStorage): Conversation storage
- `tool_registry` (ToolRegistry): Tool registry
- `memory` (Memory | None): Memory implementation
- `knowledge_base` (Any | None): Knowledge base
- `reasoning_strategy` (Any | None): Reasoning strategy
- `middleware` (list[Any]): Middleware list
- `system_prompt_name` (str | None): System prompt name
- `default_temperature` (float): Default temperature
- `default_max_tokens` (int): Default max tokens

---

### BotContext

Encapsulates the execution context for a bot interaction.

#### Class Definition

```python
@dataclass
class BotContext:
    """Context for bot execution."""
```

#### Constructor

```python
def __init__(
    self,
    conversation_id: str,
    client_id: str,
    user_id: str | None = None,
    session_metadata: dict[str, Any] | None = None,
    request_metadata: dict[str, Any] | None = None
)
```

**Parameters:**
- `conversation_id` (str): Unique identifier for the conversation
- `client_id` (str): Tenant/client identifier
- `user_id` (str, optional): User identifier
- `session_metadata` (dict, optional): Additional session metadata (persists across requests)
- `request_metadata` (dict, optional): Metadata specific to current request

**Example:**
```python
from dataknobs_bots import BotContext

# Minimal context
context = BotContext(
    conversation_id="conv-001",
    client_id="my-app"
)

# With optional fields
context = BotContext(
    conversation_id="conv-002",
    client_id="my-app",
    user_id="user-123",
    session_metadata={
        "source": "web",
        "subscription": "premium"
    },
    request_metadata={
        "ip_address": "192.168.1.1",
        "user_agent": "Mozilla/5.0"
    }
)
```

#### Attributes

- `conversation_id` (str): Conversation identifier
- `client_id` (str): Client/tenant identifier
- `user_id` (str | None): User identifier
- `session_metadata` (dict[str, Any]): Session metadata dictionary
- `request_metadata` (dict[str, Any]): Request-specific metadata dictionary

---

### BotRegistry

Multi-tenant bot registry with caching.

#### Class Definition

```python
class BotRegistry:
    """Multi-tenant bot registry with caching."""
```

#### Constructor

```python
def __init__(
    self,
    config: Config | dict[str, Any],
    cache_ttl: int = 300,
    max_cache_size: int = 1000
)
```

**Parameters:**
- `config` (Config | dict): Configuration object or dictionary
- `cache_ttl` (int): Cache time-to-live in seconds (default: 300)
- `max_cache_size` (int): Maximum cached bots (default: 1000)

**Example:**
```python
from dataknobs_bots import BotRegistry
from dataknobs_config import Config

# Load configuration
config = Config("config/bots.yaml")

# Create registry
registry = BotRegistry(config, cache_ttl=300)

# Get bot for a client
bot = await registry.get_bot("client-123")
```

#### Methods

##### `get_bot`

Get bot instance for a client.

```python
async def get_bot(
    self,
    client_id: str,
    force_refresh: bool = False
) -> DynaBot
```

**Parameters:**
- `client_id` (str): Client/tenant identifier
- `force_refresh` (bool): If True, bypass cache and create fresh bot (default: False)

**Returns:**
- `DynaBot`: Bot instance for the client

**Raises:**
- `KeyError`: If no configuration exists for the client
- `ValueError`: If bot configuration is invalid

##### `register_client`

Register or update a client's bot configuration.

```python
async def register_client(
    self,
    client_id: str,
    bot_config: dict[str, Any]
) -> None
```

**Parameters:**
- `client_id` (str): Client/tenant identifier
- `bot_config` (dict): Bot configuration dictionary

##### `remove_client`

Remove a client from the registry.

```python
async def remove_client(self, client_id: str) -> None
```

**Parameters:**
- `client_id` (str): Client/tenant identifier

##### `get_cached_clients`

Get list of currently cached client IDs.

```python
def get_cached_clients(self) -> list[str]
```

**Returns:**
- `list[str]`: List of client IDs with cached bots

##### `clear_cache`

Clear all cached bots.

```python
def clear_cache() -> None
```

#### Attributes

- `config` (Config | dict): Configuration object
- `cache_ttl` (int): Cache time-to-live in seconds
- `max_cache_size` (int): Maximum number of bots to cache

---

## Memory

### Memory (Base)

Abstract base class for memory implementations.

#### Class Definition

```python
class Memory(ABC):
    """Base class for memory systems."""
```

#### Abstract Methods

##### `add_message`

Add a message to memory.

```python
@abstractmethod
async def add_message(
    self,
    content: str,
    role: str,
    metadata: dict[str, Any] | None = None
) -> None
```

**Parameters:**
- `content` (str): Message content
- `role` (str): Message role (e.g., "user", "assistant", "system")
- `metadata` (dict, optional): Optional metadata for the message

##### `get_context`

Retrieve relevant context for a query.

```python
@abstractmethod
async def get_context(self, current_message: str) -> list[dict[str, Any]]
```

**Parameters:**
- `current_message` (str): The current message to get context for

**Returns:**
- `list[dict]`: List of relevant message dictionaries with keys:
  - `content` (str): Message content
  - `role` (str): Message role
  - `metadata` (dict): Message metadata

##### `clear`

Clear all memory.

```python
@abstractmethod
async def clear() -> None
```

---

### BufferMemory

Simple sliding window memory that keeps the last N messages.

#### Class Definition

```python
class BufferMemory(Memory):
    """Buffer memory implementation with sliding window."""
```

#### Constructor

```python
def __init__(self, max_messages: int = 10)
```

**Parameters:**
- `max_messages` (int): Maximum number of messages to keep (default: 10)

**Example:**
```python
from dataknobs_bots.memory import BufferMemory

memory = BufferMemory(max_messages=20)

# Add messages
await memory.add_message(content="Hello!", role="user")
await memory.add_message(content="Hi there!", role="assistant")

# Get context
context = await memory.get_context("What did we discuss?")
```

#### Methods

Implements all abstract methods from `Memory` base class.

---

### VectorMemory

Semantic search-based memory using vector embeddings.

#### Class Definition

```python
class VectorMemory(Memory):
    """Vector-based memory with semantic search."""
```

#### Constructor

```python
def __init__(
    self,
    embedding_provider: AsyncLLMProvider,
    vector_store: Any,
    max_messages: int = 100
)
```

**Parameters:**
- `embedding_provider` (AsyncLLMProvider): Provider for generating embeddings
- `vector_store` (Any): Vector store for similarity search
- `max_messages` (int): Maximum messages to store (default: 100)

**Example:**
```python
from dataknobs_bots.memory import VectorMemory
from dataknobs_llm.llm import LLMProviderFactory

# Create embedding provider
embedding_config = {
    "provider": "ollama",
    "model": "nomic-embed-text"
}
embedding_provider = LLMProviderFactory(is_async=True).create(embedding_config)
await embedding_provider.initialize()

# Create vector memory
memory = VectorMemory(
    embedding_provider=embedding_provider,
    vector_store=faiss_store,
    max_messages=100
)
```

---

## Knowledge Base

### RAGKnowledgeBase

RAG (Retrieval Augmented Generation) knowledge base implementation.

#### Class Definition

```python
class RAGKnowledgeBase:
    """RAG knowledge base using dataknobs-xization for chunking and vector search."""
```

#### Constructor

```python
def __init__(
    self,
    vector_store: Any,
    embedding_provider: Any,
    chunking_config: dict[str, Any] | None = None
)
```

**Parameters:**
- `vector_store` (Any): Vector store backend from dataknobs_data
- `embedding_provider` (Any): LLM provider with embed() method
- `chunking_config` (dict, optional): Document chunking configuration:
  - `max_chunk_size` (int): Maximum chunk size in characters (default: 500)
  - `chunk_overlap` (int): Overlap between chunks (default: 50)
  - `combine_under_heading` (bool): Combine text under same heading (default: True)

**Example:**
```python
from dataknobs_bots.knowledge import RAGKnowledgeBase

kb = RAGKnowledgeBase(
    vector_store=vector_store,
    embedding_provider=embedding_provider,
    chunking_config={
        "max_chunk_size": 500,
        "chunk_overlap": 50,
        "combine_under_heading": True
    }
)
```

#### Methods

##### `load_markdown_document`

Load and chunk a markdown document.

```python
async def load_markdown_document(
    self,
    filepath: str | Path,
    metadata: dict[str, Any] | None = None
) -> int
```

**Parameters:**
- `filepath` (str | Path): Path to markdown file
- `metadata` (dict, optional): Optional metadata to attach to all chunks

**Returns:**
- `int`: Number of chunks created

##### `load_documents_from_directory`

Load all markdown documents from a directory.

```python
async def load_documents_from_directory(
    self,
    directory: str | Path,
    pattern: str = "**/*.md"
) -> dict[str, Any]
```

**Parameters:**
- `directory` (str | Path): Directory path containing documents
- `pattern` (str): Glob pattern for files to load (default: "\*\*/\*.md")

**Returns:**
- `dict[str, Any]`: Dictionary with loading statistics:
  - `total_files` (int): Number of files processed
  - `total_chunks` (int): Total chunks created
  - `errors` (list): List of errors encountered

##### `query`

Query the knowledge base.

```python
async def query(
    self,
    query: str,
    k: int = 5,
    filter_metadata: dict[str, Any] | None = None,
    min_similarity: float = 0.0
) -> list[dict[str, Any]]
```

**Parameters:**
- `query` (str): Query text to search for
- `k` (int): Number of results to return (default: 5)
- `filter_metadata` (dict, optional): Optional metadata filters
- `min_similarity` (float): Minimum similarity score 0-1 (default: 0.0)

**Returns:**
- `list[dict]`: List of results with keys:
  - `text` (str): Document chunk text
  - `source` (str): Source file path
  - `heading_path` (str): Heading hierarchy
  - `similarity` (float): Similarity score
  - `metadata` (dict): Full chunk metadata

---

## Reasoning

### ReasoningStrategy (Base)

Abstract base class for reasoning strategies.

#### Class Definition

```python
class ReasoningStrategy(ABC):
    """Base class for reasoning strategies."""
```

#### Abstract Methods

##### `generate`

Generate a response using the reasoning strategy.

```python
@abstractmethod
async def generate(
    self,
    manager: ConversationManager,
    llm: AsyncLLMProvider,
    tools: list[Tool] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Any
```

**Parameters:**
- `manager` (ConversationManager): Conversation manager
- `llm` (AsyncLLMProvider): LLM provider
- `tools` (list[Tool], optional): Available tools
- `temperature` (float): Generation temperature
- `max_tokens` (int): Max tokens to generate

---

### SimpleReasoning

Direct LLM generation without multi-step reasoning.

#### Class Definition

```python
class SimpleReasoning(ReasoningStrategy):
    """Simple direct LLM response."""
```

#### Constructor

```python
def __init__(self)
```

**Example:**
```python
from dataknobs_bots.reasoning import SimpleReasoning

reasoning = SimpleReasoning()
```

---

### ReActReasoning

ReAct (Reasoning + Acting) strategy for tool-using agents.

#### Class Definition

```python
class ReActReasoning(ReasoningStrategy):
    """ReAct reasoning strategy."""
```

#### Constructor

```python
def __init__(
    self,
    max_iterations: int = 5,
    verbose: bool = False,
    store_trace: bool = False
)
```

**Parameters:**
- `max_iterations` (int): Maximum reasoning iterations (default: 5)
- `verbose` (bool): Enable verbose logging (default: False)
- `store_trace` (bool): Store reasoning trace (default: False)

**Example:**
```python
from dataknobs_bots.reasoning import ReActReasoning

reasoning = ReActReasoning(
    max_iterations=10,
    verbose=True,
    store_trace=True
)
```

#### Attributes

- `max_iterations` (int): Maximum iterations
- `verbose` (bool): Verbose mode flag
- `store_trace` (bool): Trace storage flag
- `trace` (list[dict]): Reasoning trace (if `store_trace=True`)

---

## Tools

### KnowledgeSearchTool

Built-in tool for searching the knowledge base.

#### Class Definition

```python
class KnowledgeSearchTool(Tool):
    """Tool for searching the knowledge base."""
```

#### Constructor

```python
def __init__(self, knowledge_base: KnowledgeBase, k: int = 3)
```

**Parameters:**
- `knowledge_base` (KnowledgeBase): Knowledge base to search
- `k` (int): Number of results to return (default: 3)

**Example:**
```python
from dataknobs_bots.tools import KnowledgeSearchTool

tool = KnowledgeSearchTool(knowledge_base=kb, k=5)
```

---

## Factory Functions

### `create_memory_from_config`

Create a memory instance from configuration.

```python
async def create_memory_from_config(config: dict[str, Any]) -> Memory
```

**Parameters:**
- `config` (dict): Memory configuration

**Returns:**
- `Memory`: Memory instance

**Configuration Schema:**
```python
# Buffer memory
{
    "type": "buffer",
    "max_messages": int  # default: 10
}

# Vector memory
{
    "type": "vector",
    "max_messages": int,  # default: 100
    "embedding_provider": str,
    "embedding_model": str,
    "backend": str,
    "dimension": int
}
```

**Example:**
```python
from dataknobs_bots.memory import create_memory_from_config

memory_config = {
    "type": "buffer",
    "max_messages": 20
}

memory = await create_memory_from_config(memory_config)
```

---

### `create_knowledge_base_from_config`

Create a knowledge base from configuration.

```python
async def create_knowledge_base_from_config(
    config: dict[str, Any]
) -> RAGKnowledgeBase
```

**Parameters:**
- `config` (dict): Knowledge base configuration

**Returns:**
- `RAGKnowledgeBase`: Knowledge base instance

**Configuration Schema:**
```python
{
    "enabled": bool,
    "documents_path": str,  # Path to documents directory
    "vector_store": {
        "backend": str,
        "dimension": int,
        "collection": str
    },
    "embedding_provider": str,
    "embedding_model": str,
    "chunking": {
        "max_chunk_size": int,  # default: 500
        "chunk_overlap": int    # default: 50
    }
}
```

**Example:**
```python
from dataknobs_bots.knowledge import create_knowledge_base_from_config

kb_config = {
    "enabled": True,
    "documents_path": "./docs",
    "vector_store": {
        "backend": "faiss",
        "dimension": 384
    },
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text"
}

kb = await create_knowledge_base_from_config(kb_config)
```

---

### `create_reasoning_from_config`

Create a reasoning strategy from configuration.

```python
def create_reasoning_from_config(
    config: dict[str, Any]
) -> ReasoningStrategy
```

**Parameters:**
- `config` (dict): Reasoning configuration

**Returns:**
- `ReasoningStrategy`: Reasoning strategy instance

**Configuration Schema:**
```python
# Simple reasoning
{
    "strategy": "simple"
}

# ReAct reasoning
{
    "strategy": "react",
    "max_iterations": int,  # default: 5
    "verbose": bool,        # default: False
    "store_trace": bool     # default: False
}
```

**Example:**
```python
from dataknobs_bots.reasoning import create_reasoning_from_config

reasoning_config = {
    "strategy": "react",
    "max_iterations": 10,
    "verbose": True,
    "store_trace": True
}

reasoning = create_reasoning_from_config(reasoning_config)
```

---

## Error Handling

### Common Exceptions

The library may raise the following exceptions:

- `ValueError`: Invalid configuration or parameters
- `ImportError`: Failed to import tool class from configuration
- `AttributeError`: Tool class not found in module
- `ConnectionError`: Database or LLM connection failures
- `TimeoutError`: LLM generation timeout

### Example Error Handling

```python
from dataknobs_bots import DynaBot, BotContext

try:
    bot = await DynaBot.from_config(config)
    context = BotContext(conversation_id="conv-123", client_id="client-456")
    response = await bot.chat("Hello!", context)
except ValueError as e:
    print(f"Configuration error: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Type Hints

The library uses Python type hints throughout. Key types:

```python
from typing import Any, Optional
from dataknobs_llm.llm import AsyncLLMProvider
from dataknobs_llm.prompts import AsyncPromptBuilder
from dataknobs_llm.conversations import DataknobsConversationStorage
from dataknobs_llm.tools import ToolRegistry, Tool

# Common type aliases
ConfigDict = dict[str, Any]
MessageDict = dict[str, Any]
ContextDict = dict[str, Any]
```

---

## See Also

- [User Guide](../guides/user-guide.md) - Tutorials and how-to guides
- [Configuration Reference](../guides/configuration.md) - Complete configuration options
- [Tools Development](../guides/tools.md) - Creating custom tools
- [Architecture](../guides/architecture.md) - System design
- [Examples](../examples/index.md) - Working code examples
