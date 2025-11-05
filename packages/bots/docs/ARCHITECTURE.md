# Architecture

System design and technical architecture of DataKnobs Bots.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Multi-Tenancy](#multi-tenancy)
- [Scaling Considerations](#scaling-considerations)
- [Design Patterns](#design-patterns)
- [Integration Points](#integration-points)
- [Performance Characteristics](#performance-characteristics)

---

## Overview

DynaBot is designed as a **stateless, configuration-driven framework** for building AI agents and chatbots. The architecture emphasizes:

- **Modularity**: Pluggable components for LLM, storage, memory, and reasoning
- **Scalability**: Stateless design enabling horizontal scaling
- **Flexibility**: Configuration-driven behavior without code changes
- **Extensibility**: Easy addition of custom tools, memory strategies, and middleware

### Key Architectural Principles

1. **Configuration First**: All behavior defined through configuration
2. **Stateless Execution**: No shared state between requests
3. **Async by Default**: Fully asynchronous for high concurrency
4. **Ecosystem Integration**: Leverages DataKnobs ecosystem components
5. **Clean Abstractions**: Clear interfaces for extensibility

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Application                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ API Calls
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                          DynaBot                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Message Processing Pipeline             │   │
│  │  1. Middleware (Pre)                                 │   │
│  │  2. Context Building (Memory + Knowledge)            │   │
│  │  3. LLM Generation (with Reasoning)                  │   │
│  │  4. Tool Execution (if needed)                       │   │
│  │  5. Middleware (Post)                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Memory     │  │   Knowledge  │  │   Reasoning  │     │
│  │              │  │     Base     │  │   Strategy   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     Tools    │  │  Middleware  │  │   Prompts    │     │
│  │   Registry   │  │              │  │   Builder    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────┬───────────────────────────────────┬─────────┘
               │                                   │
               ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│   Conversation Storage  │         │      LLM Provider       │
│   (PostgreSQL/Memory)   │         │  (OpenAI/Ollama/etc)    │
└─────────────────────────┘         └─────────────────────────┘
```

### Component Hierarchy

```
DynaBot (Orchestrator)
├── AsyncLLMProvider (LLM Interface)
├── AsyncPromptBuilder (Prompt Management)
├── DataknobsConversationStorage (Storage)
│   └── Database Backend (PostgreSQL/Memory)
├── ToolRegistry (Tool Management)
│   └── Tools[] (Individual Tools)
├── Memory (Context Management)
│   ├── BufferMemory
│   └── VectorMemory
├── KnowledgeBase (RAG)
│   ├── VectorStore
│   └── EmbeddingProvider
├── ReasoningStrategy (Multi-Step Reasoning)
│   ├── SimpleReasoning
│   └── ReActReasoning
└── Middleware[] (Request/Response Processing)
```

---

## Core Components

### 1. DynaBot (Orchestrator)

**Responsibility**: Orchestrates all components and manages the message processing pipeline.

**Key Methods**:
- `from_config()`: Creates bot from configuration
- `chat()`: Processes user messages
- `_get_or_create_conversation()`: Manages conversation lifecycle
- `_build_message_with_context()`: Augments messages with context

**State Management**:
- Stateless per request
- Caches ConversationManager instances per conversation_id
- No shared mutable state

**Concurrency**: Fully async, supports concurrent requests

### 2. AsyncLLMProvider

**Responsibility**: Abstraction over different LLM providers.

**Interface** (from dataknobs-llm):
```python
class AsyncLLMProvider(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Response:
        """Generate completion."""
```

**Implementations**:
- OllamaProvider
- OpenAIProvider
- AnthropicProvider
- AzureOpenAIProvider

### 3. ConversationStorage

**Responsibility**: Persistent storage for conversation history.

**Interface** (from dataknobs-llm):
```python
class DataknobsConversationStorage(ABC):
    @abstractmethod
    async def save_message(self, conversation_id: str, message: Dict) -> None:
        """Save a message."""

    @abstractmethod
    async def load_conversation(self, conversation_id: str) -> List[Dict]:
        """Load conversation history."""
```

**Backends**:
- Memory (in-process dictionary)
- PostgreSQL (persistent database)

### 4. Memory

**Responsibility**: Manage conversation context beyond raw history.

**Types**:

**BufferMemory** (Sliding Window):
```
Messages: [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10]
          └────────────── Window (max=10) ────────────┘
New message comes in → M1 is evicted
```

**VectorMemory** (Semantic Search):
```
Query: "What did we discuss about pricing?"
          ↓ Embedding
    [0.23, 0.41, ..., 0.87] (384-dim vector)
          ↓ Similarity Search
    Top K similar messages from history
```

### 5. KnowledgeBase (RAG)

**Responsibility**: Retrieval Augmented Generation with document search.

**Architecture**:
```
Documents
    ↓ Chunking
Document Chunks
    ↓ Embedding
Vectors → VectorStore
              ↓ Query
          Retrieved Context
              ↓
          LLM + Context
```

**Components**:
- Document loader
- Text chunker
- Embedding provider
- Vector store
- Retrieval mechanism

### 6. ReasoningStrategy

**Responsibility**: Multi-step reasoning for complex tasks.

**ReAct Loop**:
```
1. Thought: What should I do?
2. Action: Use a tool
3. Observation: Tool result
4. [Repeat or Final Answer]
```

**Flow**:
```python
for iteration in range(max_iterations):
    # 1. Generate reasoning step
    response = await llm.complete(messages + tools_prompt)

    # 2. Parse thought and action
    thought, action, action_input = parse_response(response)

    # 3. Execute tool if action specified
    if action:
        observation = await tool_registry.execute(action, action_input)
        messages.append({"role": "tool", "content": observation})
    else:
        # Final answer reached
        break
```

### 7. ToolRegistry

**Responsibility**: Manage available tools and route tool calls.

**Operations**:
- Register tools
- Get tool by name
- List available tools
- Generate tool schemas for LLM

**Tool Loading**:
```python
# Direct instantiation
tool = CalculatorTool(precision=2)
registry.register(tool)

# From configuration
tool = _resolve_tool(config)
registry.register(tool)
```

### 8. Middleware

**Responsibility**: Cross-cutting concerns (logging, auth, metrics).

**Pipeline**:
```
Request
  ↓
Middleware 1 (before)
  ↓
Middleware 2 (before)
  ↓
Core Processing
  ↓
Middleware 2 (after)
  ↓
Middleware 1 (after)
  ↓
Response
```

---

## Data Flow

### Message Processing Flow

```
1. Client sends message
       ↓
2. Create/Resume BotContext
       ↓
3. Middleware (before_message)
       ↓
4. Build context from Memory + Knowledge Base
       ↓
5. Add augmented message to conversation
       ↓
6. Generate response (with or without reasoning)
       ├─ Without reasoning: Direct LLM call
       └─ With reasoning: ReAct loop with tools
       ↓
7. Update Memory with response
       ↓
8. Middleware (after_message)
       ↓
9. Return response to client
```

### Detailed Flow with Components

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ message, context
     ▼
┌─────────────────────────────────────┐
│ DynaBot.chat()                      │
├─────────────────────────────────────┤
│ 1. Apply middleware (before)        │
│    • Logging                        │
│    • Authentication                 │
│    • Rate limiting                  │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ _build_message_with_context()       │
├─────────────────────────────────────┤
│ 2. Query KnowledgeBase              │
│    message → [relevant docs]        │
│                                     │
│ 3. Query Memory                     │
│    message → [relevant history]     │
│                                     │
│ 4. Augment message                  │
│    Context + History + Message      │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ _get_or_create_conversation()       │
├─────────────────────────────────────┤
│ 5. Resume or create conversation    │
│    • Check cache                    │
│    • Load from storage              │
│    • Create new if needed           │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ manager.add_message()               │
├─────────────────────────────────────┤
│ 6. Add user message to history      │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Generate Response                   │
├─────────────────────────────────────┤
│ If reasoning_strategy:              │
│   ┌──────────────────────────────┐ │
│   │ ReActReasoning.generate()    │ │
│   │  • Thought loop              │ │
│   │  • Tool execution            │ │
│   │  • Observation               │ │
│   │  • Final answer              │ │
│   └──────────────────────────────┘ │
│ Else:                               │
│   ┌──────────────────────────────┐ │
│   │ manager.complete()           │ │
│   │  • Direct LLM call           │ │
│   └──────────────────────────────┘ │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ memory.add_message()                │
├─────────────────────────────────────┤
│ 7. Update memory with response      │
└────┬────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Middleware (after)                  │
├─────────────────────────────────────┤
│ 8. Post-processing                  │
│    • Logging                        │
│    • Metrics                        │
└────┬────────────────────────────────┘
     │
     ▼
┌──────────┐
│  Client  │
└──────────┘
```

---

## Multi-Tenancy

### Design

DynaBot supports multi-tenancy through **client_id** in BotContext:

```python
context = BotContext(
    conversation_id="conv-123",
    client_id="tenant-A",  # Tenant identifier
    user_id="user-456"
)
```

### Isolation

**Conversation Isolation**:
- Each conversation has unique `conversation_id`
- Conversations are isolated per `client_id`
- No data leakage between tenants

**Storage Partitioning**:
```sql
-- PostgreSQL schema
CREATE TABLE conversations (
    id VARCHAR PRIMARY KEY,
    client_id VARCHAR NOT NULL,  -- Tenant
    user_id VARCHAR,
    created_at TIMESTAMP,
    -- ... other fields
    INDEX idx_client_id (client_id)
);

CREATE TABLE messages (
    id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR REFERENCES conversations(id),
    -- ... message fields
);
```

### Scaling Strategy

```
┌─────────────────────────────────────────────┐
│         Load Balancer                        │
└───┬────────────────┬────────────────┬────────┘
    │                │                │
    ▼                ▼                ▼
┌─────────┐      ┌─────────┐     ┌─────────┐
│ Bot     │      │ Bot     │     │ Bot     │
│ Instance│      │ Instance│     │ Instance│
│   #1    │      │   #2    │     │   #3    │
└────┬────┘      └────┬────┘     └────┬─────┘
     │                │                │
     └────────────────┴────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Shared PostgreSQL    │
         │   Conversation Storage │
         └────────────────────────┘
```

**Characteristics**:
- Stateless bot instances
- Shared conversation storage
- Horizontal scaling
- No sticky sessions needed

---

## Scaling Considerations

### Vertical Scaling

**Memory Considerations**:
- ConversationManager cache grows with active conversations
- Vector memory requires more RAM than buffer memory
- Knowledge base vectors stored in memory (FAISS) or external (Pinecone)

**Recommendations**:
- Implement cache eviction for inactive conversations
- Use external vector stores for large knowledge bases
- Monitor memory usage and set limits

### Horizontal Scaling

**Stateless Design**:
- No shared state between bot instances
- Each request is independent
- Easy to add more instances

**Connection Pooling**:
```python
conversation_storage:
  backend: postgres
  pool_size: 20          # Connections per instance
  max_overflow: 10       # Extra connections
```

**Load Distribution**:
- Round-robin or least-connections
- No session affinity needed
- Geographic distribution possible

### Database Scaling

**PostgreSQL Optimization**:
- Index on `client_id`, `conversation_id`
- Partition by `client_id` for large tenants
- Read replicas for high read loads
- Connection pooling

**Schema Design**:
```sql
-- Partitioning example
CREATE TABLE messages (
    id VARCHAR PRIMARY KEY,
    conversation_id VARCHAR,
    client_id VARCHAR,
    created_at TIMESTAMP,
    -- ... fields
) PARTITION BY HASH (client_id);

CREATE TABLE messages_p0 PARTITION OF messages
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
-- ... create p1, p2, p3
```

---

## Design Patterns

### 1. Factory Pattern

**Used for**: Creating components from configuration

```python
# LLM Provider Factory
llm = LLMProviderFactory(is_async=True).create(llm_config)

# Database Factory
backend = AsyncDatabaseFactory().create(**storage_config)

# Memory Factory
memory = await create_memory_from_config(memory_config)
```

### 2. Strategy Pattern

**Used for**: Reasoning strategies

```python
class ReasoningStrategy(ABC):
    @abstractmethod
    async def generate(...) -> Any:
        pass

class SimpleReasoning(ReasoningStrategy):
    async def generate(...):
        # Simple strategy

class ReActReasoning(ReasoningStrategy):
    async def generate(...):
        # ReAct strategy
```

### 3. Registry Pattern

**Used for**: Tool management

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]
```

### 4. Builder Pattern

**Used for**: Prompt construction

```python
prompt_builder = AsyncPromptBuilder(library)
prompt = await prompt_builder.build(
    prompt_name="system_prompt",
    variables={"user_name": "Alice"}
)
```

### 5. Middleware Pattern

**Used for**: Cross-cutting concerns

```python
for middleware in self.middleware:
    if hasattr(middleware, "before_message"):
        await middleware.before_message(message, context)

# ... core processing ...

for middleware in self.middleware:
    if hasattr(middleware, "after_message"):
        await middleware.after_message(response, context)
```

### 6. Dependency Injection

**Used for**: Component composition

```python
bot = DynaBot(
    llm=llm_provider,
    prompt_builder=prompt_builder,
    conversation_storage=storage,
    tool_registry=tools,
    memory=memory,
    knowledge_base=kb,
    reasoning_strategy=reasoning
)
```

---

## Integration Points

### DataKnobs Ecosystem

```
dataknobs-bots (This Package)
    ↓ depends on
┌─────────────────────────┬──────────────────┬──────────────────┐
│   dataknobs-llm         │ dataknobs-data   │ dataknobs-config │
│   • LLM providers       │ • DB backends    │ • Config system  │
│   • Tools interface     │ • Storage        │ • XRef resolution│
│   • Conversations       │ • Async DB       │                  │
└─────────────────────────┴──────────────────┴──────────────────┘
                    ↓
         dataknobs-xization
         • Type conversions
         • Data transformations
```

### External Services

**LLM Providers**:
- OpenAI API
- Anthropic API
- Azure OpenAI
- Ollama (local)

**Vector Stores**:
- FAISS (local)
- Pinecone (cloud)
- Chroma (local/cloud)
- Weaviate (cloud)

**Databases**:
- PostgreSQL
- In-memory (development)

---

## Performance Characteristics

### Latency Breakdown

**Typical Request Latency**:
```
Total: ~500-2000ms
├── Memory query: 10-50ms
├── Knowledge base query: 50-200ms
├── LLM generation: 400-1500ms
└── Storage operations: 20-100ms
```

**Optimization Strategies**:
1. **Parallel Queries**: Memory and KB queries in parallel
2. **Caching**: Cache conversation managers
3. **Connection Pooling**: Reduce DB connection overhead
4. **Local LLM**: Use Ollama for lower latency

### Throughput

**Factors**:
- LLM provider rate limits
- Database connection pool size
- Memory usage per conversation
- Vector search performance

**Typical Throughput** (with OpenAI GPT-4):
- ~10-20 requests/second per instance
- Limited by LLM API rate limits
- Horizontal scaling increases total throughput

### Resource Usage

**Memory** (per active conversation):
- Minimal: ~1-5 MB (buffer memory)
- Moderate: ~10-50 MB (vector memory)
- High: ~100+ MB (with large KB)

**CPU**:
- Low during idle
- Moderate during LLM calls (async waiting)
- High during local embeddings or vector search

**Network**:
- Dependent on LLM provider
- ~1-10 KB request + ~1-50 KB response

---

## See Also

- [API Reference](../api/reference.md) - Complete API documentation
- [Configuration Reference](configuration.md) - Configuration options
- [User Guide](user-guide.md) - Usage tutorials
- [Tools Development](tools.md) - Creating custom tools
- [Examples](../examples/index.md) - Working examples
