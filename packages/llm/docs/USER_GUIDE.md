# Dataknobs LLM - User Guide

**Package**: `dataknobs_llm`
**Version**: 0.1.0
**Last Updated**: 2025-10-29

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Prompt Library System](#prompt-library-system)
5. [Conversation Management](#conversation-management)
6. [Middleware](#middleware)
7. [Advanced Features](#advanced-features)
8. [Complete Examples](#complete-examples)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

The Dataknobs LLM package provides a comprehensive system for managing prompts and multi-turn conversations with Large Language Models. It includes:

- **Prompt Library**: Template-based prompt management with validation and RAG
- **Resource Adapters**: Plug in any data source for dynamic content
- **Conversation Management**: Multi-turn conversations with branching and persistence
- **Middleware System**: Logging, validation, content filtering, and custom processing
- **LLM Integration**: Works with OpenAI, Anthropic, Ollama, and more

---

## Installation

```bash
# Install the package
uv add dataknobs-llm

# With optional dependencies for file-based prompts
uv add dataknobs-llm[yaml]
```

**Dependencies**:
- `dataknobs-common` - Utilities
- `dataknobs-config` - Configuration management
- `dataknobs-data` - Database backends (for conversation storage)
- `pyyaml` - YAML support (optional)

---

## Quick Start

### Simple Prompt Rendering

```python
from dataknobs_llm.prompts import render_template

# Basic template
result = render_template(
    "Hello {{name}}!",
    {"name": "Alice"}
)
print(result.content)  # "Hello Alice!"

# With conditional sections
result = render_template(
    "Hello {{name}}((, you are {{age}} years old))",
    {"name": "Alice", "age": 30}
)
print(result.content)  # "Hello Alice, you are 30 years old"

# Missing optional parameter
result = render_template(
    "Hello {{name}}((, you are {{age}} years old))",
    {"name": "Alice"}
)
print(result.content)  # "Hello Alice"
```

### One-Off LLM Interaction

```python
from dataknobs_llm.llm import OpenAIProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary

# Create prompt library
config = {
    "user": {
        "analyze_code": {
            0: {"template": "Analyze this {{language}} code:\n{{code}}"}
        }
    }
}
library = ConfigPromptLibrary(config)

# Create builder and LLM
builder = AsyncPromptBuilder(library=library)
llm = OpenAIProvider(
    config={"api_key": "your-key"},
    prompt_builder=builder
)

# Render and execute in one call
result = await llm.render_and_complete(
    "analyze_code",
    params={"code": "def hello(): print('hi')", "language": "python"}
)
print(result.content)
```

---

## Prompt Library System

### Template Syntax

The prompt library uses a simple, powerful template syntax:

```python
# Variable substitution
"{{variable}}"  # Required variable

# Conditional sections (removed if variables missing)
"((optional content with {{variable}}))"

# Nested conditionals
"Hello {{name}}((, age {{age}}((, from {{city}})))"
```

### Creating Prompts

#### Option 1: Config-Based Library (In-Memory)

```python
from dataknobs_llm.prompts import ConfigPromptLibrary

config = {
    "system": {
        "helpful_assistant": {
            "template": "You are a helpful assistant specializing in {{domain}}.",
            "defaults": {"domain": "general knowledge"},
            "validation": {
                "level": "warn",
                "required_params": []
            }
        }
    },
    "user": {
        "ask_question": {
            0: {"template": "{{question}}"},
            1: {"template": "{{question}}\n\nPlease be concise."}
        }
    }
}

library = ConfigPromptLibrary(config)
```

#### Option 2: Filesystem Library (File-Based)

Create directory structure:
```
prompts/
├── system/
│   └── helpful_assistant.yaml
├── user/
│   └── ask_question.yaml
└── rag/
    └── docs_search.yaml
```

**system/helpful_assistant.yaml**:
```yaml
template: "You are a helpful assistant specializing in {{domain}}."
defaults:
  domain: "general knowledge"
validation:
  level: "warn"
  required_params: []
```

**user/ask_question.yaml**:
```yaml
template: "{{question}}"
validation:
  level: "error"
  required_params: ["question"]
```

**Load the library**:
```python
from dataknobs_llm.prompts import FileSystemPromptLibrary
from pathlib import Path

library = FileSystemPromptLibrary(Path("prompts/"))
```

#### Option 3: Composite Library (Layered Overrides)

```python
from dataknobs_llm.prompts import CompositePromptLibrary

# Custom overrides + base defaults
composite = CompositePromptLibrary(
    libraries=[custom_library, base_library],
    names=["custom", "base"]
)

# First library wins (custom overrides base)
prompt = composite.get_system_prompt("helpful_assistant")
```

### Using Prompt Builder

```python
from dataknobs_llm.prompts import AsyncPromptBuilder

builder = AsyncPromptBuilder(library=library)

# Render system prompt from library
result = await builder.render_system_prompt(
    "helpful_assistant",
    params={"domain": "Python programming"}
)
print(result.content)

# Render user prompt from library
result = await builder.render_user_prompt(
    "ask_question",
    params={"question": "What is async/await?"}
)
print(result.content)
```

### Inline Prompt Rendering

For quick prototyping or dynamic prompts, use inline rendering without defining templates in the library:

```python
# Render inline system prompt
result = await builder.render_inline_system_prompt(
    content="You are a helpful {{role}} assistant.",
    params={"role": "coding"}
)
print(result.content)  # "You are a helpful coding assistant."

# Render inline user prompt
result = await builder.render_inline_user_prompt(
    content="Help me understand {{topic}}",
    params={"topic": "decorators"}
)
```

Inline prompts also support RAG enhancement:

```python
# Inline system prompt with RAG
result = await builder.render_inline_system_prompt(
    content="You are a helpful assistant.\n\nContext:\n{{CONTEXT}}",
    rag_configs=[{
        "adapter_name": "docs",
        "query": "assistant guidelines",
        "placeholder": "CONTEXT",
        "k": 3
    }]
)
```

This is useful for:
- Prototyping prompts before adding to a library
- Creating one-off prompts that don't need versioning
- Dynamically constructing prompts at runtime

### Validation Levels

```python
from dataknobs_llm.prompts import ValidationLevel

# Three levels available:
ValidationLevel.ERROR   # Raise exception for missing required params
ValidationLevel.WARN    # Log warning but continue
ValidationLevel.IGNORE  # No validation

# Validation hierarchy (highest priority first):
# 1. Runtime override
# 2. Template config
# 3. Builder default
# 4. Global default (WARN)

# Runtime override
result = await builder.render_user_prompt(
    "ask_question",
    params={},
    validation_level=ValidationLevel.ERROR  # Override
)
```

### Resource Adapters (RAG)

Resource adapters provide dynamic content injection:

```python
from dataknobs_llm.prompts import (
    AsyncDictResourceAdapter,
    AsyncDataknobsBackendAdapter
)
from dataknobs_data.backends import AsyncMemoryDatabase

# Dictionary adapter
config_data = {
    "coding_standards": {
        "python": "Use PEP 8 style guide",
        "javascript": "Use ESLint recommended"
    }
}
config_adapter = AsyncDictResourceAdapter(config_data)

# Database adapter
docs_db = AsyncMemoryDatabase()
# ... populate database with documents ...
docs_adapter = AsyncDataknobsBackendAdapter(
    docs_db,
    text_field="content",
    metadata_fields=["title", "category"]
)

# Create builder with adapters
builder = AsyncPromptBuilder(
    library=library,
    adapters={
        "config": config_adapter,
        "docs": docs_adapter
    }
)
```

### RAG Configuration

**rag/docs_search.yaml**:
```yaml
adapter_name: "docs"
query: "{{topic}} {{language}}"
k: 5
filters:
  category: "documentation"
score_threshold: 0.7
```

**Reference RAG from prompts**:
```yaml
# system/analyze_code.yaml
template: |
  You are a code analyzer.

  Relevant documentation:
  {{RAG_CONTENT}}

  Analyze this code: {{code}}

rag_config_refs: ["docs_search"]  # Reference RAG config
defaults:
  language: "python"
```

**Using RAG**:
```python
# RAG executes automatically with include_rag=True (default)
result = await builder.render_system_prompt(
    "analyze_code",
    params={"code": code_snippet, "topic": "async programming"},
    include_rag=True  # Default
)

# Result includes RAG content injected into {{RAG_CONTENT}}
```

### Template Composition

Templates can inherit from other templates:

```yaml
# system/base_assistant.yaml
template: |
  {{HEADER}}

  {{INSTRUCTIONS}}

  {{FOOTER}}

sections:
  HEADER: "You are a helpful AI assistant."
  INSTRUCTIONS: "Answer questions accurately."
  FOOTER: "Be concise and clear."

# system/code_assistant.yaml
extends: "base_assistant"  # Inherit from base

sections:
  HEADER: "You are an expert code reviewer."  # Override
  INSTRUCTIONS: |
    Review code for:
    - Bugs and errors
    - Best practices
    - Security issues
```

**Result**: The code_assistant inherits the template structure but overrides specific sections.

---

## Conversation Management

### Creating Conversations

```python
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage
)
from dataknobs_llm.llm import OpenAIProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_data.backends import AsyncMemoryDatabase

# Setup components
library = FileSystemPromptLibrary("prompts/")
builder = AsyncPromptBuilder(library=library)
llm = OpenAIProvider(config={"api_key": "your-key"})
storage = DataknobsConversationStorage(AsyncMemoryDatabase())

# Create new conversation
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="helpful_assistant",  # Initial system prompt
    system_params={"domain": "Python programming"}
)

print(f"Conversation ID: {manager.conversation_id}")
```

### Adding Messages

```python
# Add user message from prompt template
await manager.add_message(
    role="user",
    prompt_name="ask_question",
    params={"question": "What is async/await?"}
)

# Add user message with direct content
await manager.add_message(
    role="user",
    content="Can you explain with examples?"
)

# Add inline content with RAG enhancement
await manager.add_message(
    role="system",
    content="You are a helpful assistant.\n\nContext:\n{{CONTEXT}}",
    rag_configs=[{
        "adapter_name": "docs",
        "query": "assistant guidelines",
        "placeholder": "CONTEXT",
        "k": 3
    }]
)

# Get LLM response
response = await manager.complete()
print(response.content)
```

### Multi-Turn Conversations

```python
# Turn 1
await manager.add_message(
    role="user",
    content="Explain Python decorators"
)
response1 = await manager.complete()

# Turn 2
await manager.add_message(
    role="user",
    content="Show me an example"
)
response2 = await manager.complete()

# Turn 3
await manager.add_message(
    role="user",
    content="How about with arguments?"
)
response3 = await manager.complete()

# Get full history
messages = manager.state.get_current_messages()
for msg in messages:
    print(f"{msg.role}: {msg.content[:50]}...")
```

### Streaming Responses

```python
await manager.add_message(
    role="user",
    content="Write a long explanation of generators"
)

# Stream response chunks
async for chunk in manager.stream_complete():
    print(chunk.delta, end="", flush=True)

print("\n")  # New line after streaming
```

### Conversation Branching

The conversation system uses a tree structure, allowing multiple alternative paths:

```python
# Initial conversation
await manager.add_message(role="user", content="Explain lists")
response1 = await manager.complete()  # Node "0.0"

# Continue main branch
await manager.add_message(role="user", content="What about tuples?")
response2 = await manager.complete()  # Node "0.0.0"

# Go back and create alternative response
await manager.switch_to_node("0")  # Back to "Explain lists"
response1_alt = await manager.complete(branch_name="alternative")  # Node "0.1"

# Explore branches
branches = manager.get_branches()
for branch in branches:
    print(f"Branch {branch['node_id']}: {branch['branch_name']}")
    print(f"  Content: {branch['message'].content[:50]}...")
```

### RAG Caching

When using prompts with RAG (Retrieval-Augmented Generation), you can enable caching to improve performance and reduce costs:

```python
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="assistant",
    cache_rag_results=True,        # Store RAG metadata in conversation nodes
    reuse_rag_on_branch=True       # Reuse cached RAG when branching
)

# First message executes RAG search
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"language": "python", "topic": "decorators"}
)
await manager.complete()

# Branch back to earlier point
await manager.switch_to_node("0")

# This reuses cached RAG results (same query parameters!)
await manager.add_message(
    role="user",
    prompt_name="code_question",
    params={"language": "python", "topic": "decorators"}  # Same params = cache hit
)
```

**How it works:**
- `cache_rag_results=True` stores RAG metadata (queries, results, hashes) in conversation nodes
- `reuse_rag_on_branch=True` searches the conversation tree for cached results before executing new RAG searches
- Cache matching uses query hashes to ensure identical queries reuse results

**Inspecting RAG metadata:**

```python
# Get RAG metadata from current node
metadata = manager.get_rag_metadata()

if metadata:
    for placeholder, rag_data in metadata.items():
        print(f"Query: {rag_data['query']}")
        print(f"Results: {len(rag_data['results'])}")
        print(f"Timestamp: {rag_data['timestamp']}")
```

For detailed information about RAG caching, including cache matching logic, best practices, and troubleshooting, see [RAG_CACHING.md](RAG_CACHING.md).

### Persistence and Resumption

```python
# Create conversation
manager1 = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    system_prompt_name="helpful_assistant"
)

# Have some conversation
await manager1.add_message(role="user", content="Hello")
await manager1.complete()

# Save conversation ID
conv_id = manager1.conversation_id

# ... later, possibly in different session ...

# Resume conversation
manager2 = await ConversationManager.resume(
    conversation_id=conv_id,
    llm=llm,
    prompt_builder=builder,
    storage=storage
)

# Continue where we left off
await manager2.add_message(role="user", content="Tell me more")
response = await manager2.complete()
```

### Storage Backends

The conversation system works with any dataknobs backend:

```python
from dataknobs_data.backends import (
    AsyncMemoryDatabase,    # In-memory (testing)
    AsyncFileDatabase,      # Local files
    AsyncPostgresDatabase,  # PostgreSQL
    AsyncSQLiteDatabase     # SQLite
)

# In-memory storage (testing)
storage = DataknobsConversationStorage(AsyncMemoryDatabase())

# File-based storage (local development)
storage = DataknobsConversationStorage(
    AsyncFileDatabase(base_path="conversations/")
)

# PostgreSQL (production)
storage = DataknobsConversationStorage(
    AsyncPostgresDatabase(connection_string="postgresql://...")
)

# SQLite (simple persistence)
storage = DataknobsConversationStorage(
    AsyncSQLiteDatabase(db_path="conversations.db")
)
```

### Querying Conversations

```python
# List all conversations
conversations = await storage.list_conversations(limit=10)

# Filter by metadata
customer_convs = await storage.list_conversations(
    filter_metadata={"customer_id": "12345"},
    limit=50
)

# Load specific conversation
state = await storage.load_conversation(conversation_id)
if state:
    print(f"Conversation created: {state.created_at}")
    print(f"Last updated: {state.updated_at}")
    print(f"Message count: {len(state.get_current_messages())}")
```

---

## Middleware

Middleware processes requests before LLM calls and responses after LLM calls.

### Built-in Middleware

#### LoggingMiddleware

```python
from dataknobs_llm.conversations import LoggingMiddleware
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging_mw = LoggingMiddleware(logger)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[logging_mw]
)

# All interactions will be logged
await manager.add_message(role="user", content="Hello")
await manager.complete()
# Output: "Conversation <id> - Sending 2 messages to LLM"
# Output: "Conversation <id> - Received response: 150 chars, model=gpt-4, finish_reason=stop"
```

#### ContentFilterMiddleware

```python
from dataknobs_llm.conversations import ContentFilterMiddleware

# Define filter words
filter_mw = ContentFilterMiddleware(
    filter_words=["inappropriate", "offensive"],
    replacement="[FILTERED]",
    case_sensitive=False
)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[filter_mw]
)

# Filtered words will be replaced in responses
response = await manager.complete()
# If LLM returns "This is inappropriate", it becomes "This is [FILTERED]"
```

#### ValidationMiddleware

```python
from dataknobs_llm.conversations import ValidationMiddleware

# Create validation LLM (can be different from main LLM)
validation_llm = OpenAIProvider(config={"api_key": "your-key"})

validation_mw = ValidationMiddleware(
    llm=validation_llm,
    prompt_builder=builder,
    validation_prompt="validate_response",  # Prompt that checks if response is valid
    auto_retry=False  # Raise error on validation failure
)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[validation_mw]
)

# Responses will be validated
# If validation fails, ValueError is raised (auto_retry=False)
# If auto_retry=True, response.metadata["retry_requested"] = True
```

**Validation prompt example** (user/validate_response.yaml):
```yaml
template: |
  Check if this response is appropriate and helpful:

  {{response}}

  Respond with "VALID" if appropriate, "INVALID" otherwise.
```

#### MetadataMiddleware

```python
from dataknobs_llm.conversations import MetadataMiddleware
from datetime import datetime

# Static metadata
metadata_mw = MetadataMiddleware(
    request_metadata={"source": "web_app", "version": "1.0"},
    response_metadata={"processed": True}
)

# Dynamic metadata with function
def get_timestamp():
    return {"timestamp": datetime.now().isoformat()}

dynamic_mw = MetadataMiddleware(
    response_metadata_fn=get_timestamp
)

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[metadata_mw, dynamic_mw]
)
```

### Middleware Execution Order

Middleware executes in "onion model":
- **Requests**: Forward order (first to last)
- **Responses**: Reverse order (last to first)

```python
middleware = [
    LoggingMiddleware(logger),      # 1st for requests, 3rd for responses
    ValidationMiddleware(...),       # 2nd for requests, 2nd for responses
    MetadataMiddleware(...)          # 3rd for requests, 1st for responses
]

# Request flow:  Logging → Validation → Metadata → LLM
# Response flow: LLM → Metadata → Validation → Logging
```

### Custom Middleware

```python
from dataknobs_llm.conversations import ConversationMiddleware

class TokenCounterMiddleware(ConversationMiddleware):
    """Count tokens in requests and responses."""

    def __init__(self):
        self.request_tokens = 0
        self.response_tokens = 0

    async def process_request(self, messages, state):
        # Count tokens in messages
        for msg in messages:
            self.request_tokens += len(msg.content.split())
        return messages

    async def process_response(self, response, state):
        # Count tokens in response
        self.response_tokens += len(response.content.split())

        # Add to metadata
        if not response.metadata:
            response.metadata = {}
        response.metadata["total_tokens"] = (
            self.request_tokens + self.response_tokens
        )

        return response

# Use custom middleware
counter_mw = TokenCounterMiddleware()

manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    middleware=[counter_mw]
)

response = await manager.complete()
print(f"Total tokens: {response.metadata['total_tokens']}")
```

---

## Advanced Features

### Synchronous vs Asynchronous

The library provides both sync and async versions:

```python
# Async version (recommended)
from dataknobs_llm.prompts import AsyncPromptBuilder, AsyncDictResourceAdapter

builder = AsyncPromptBuilder(
    library=library,
    adapters={"data": AsyncDictResourceAdapter(data)}
)
result = await builder.render_system_prompt("prompt_name", params={...})

# Sync version (for non-async code)
from dataknobs_llm.prompts import PromptBuilder, DictResourceAdapter

builder = PromptBuilder(
    library=library,
    adapters={"data": DictResourceAdapter(data)}
)
result = builder.render_system_prompt("prompt_name", params={...})
```

### Error Handling

```python
from dataknobs_llm.prompts import PromptNotFoundError

try:
    result = await builder.render_user_prompt(
        "non_existent_prompt",
        params={}
    )
except PromptNotFoundError as e:
    print(f"Prompt not found: {e}")

# Validation errors
from dataknobs_llm.prompts import ValidationLevel

try:
    result = await builder.render_user_prompt(
        "prompt_name",
        params={},  # Missing required params
        validation_level=ValidationLevel.ERROR
    )
except ValueError as e:
    print(f"Validation failed: {e}")
```

### Metadata Tracking

```python
# Conversation metadata
manager = await ConversationManager.create(
    llm=llm,
    prompt_builder=builder,
    storage=storage,
    metadata={
        "user_id": "alice",
        "session_id": "abc123",
        "app_version": "1.0"
    }
)

# Add metadata during conversation
await manager.state.metadata.update({"topic": "python"})

# Node metadata (per message)
await manager.add_message(
    role="user",
    content="Hello",
    metadata={"intent": "greeting"}
)
```

---

## Complete Examples

### Example 1: Code Review Assistant

```python
"""
Complete code review assistant with validation and logging.
"""
import logging
from pathlib import Path
from dataknobs_llm.llm import OpenAIProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
    LoggingMiddleware,
    ValidationMiddleware
)
from dataknobs_data.backends import AsyncPostgresDatabase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_code_review_assistant():
    # Load prompts
    library = FileSystemPromptLibrary(Path("prompts/code_review/"))
    builder = AsyncPromptBuilder(library=library)

    # Create LLM
    llm = OpenAIProvider(config={
        "api_key": "your-key",
        "model": "gpt-4"
    })

    # Create validation LLM
    validation_llm = OpenAIProvider(config={
        "api_key": "your-key",
        "model": "gpt-3.5-turbo"  # Cheaper for validation
    })

    # Setup storage
    storage = DataknobsConversationStorage(
        AsyncPostgresDatabase(connection_string="postgresql://...")
    )

    # Setup middleware
    middleware = [
        LoggingMiddleware(logger),
        ValidationMiddleware(
            llm=validation_llm,
            prompt_builder=builder,
            validation_prompt="validate_code_review",
            auto_retry=True
        )
    ]

    # Create conversation
    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="code_reviewer",
        system_params={"language": "python"},
        middleware=middleware,
        metadata={"type": "code_review"}
    )

    return manager

async def review_code(manager, code: str):
    # Submit code for review
    await manager.add_message(
        role="user",
        prompt_name="submit_code",
        params={"code": code}
    )

    # Get review
    review = await manager.complete()
    print("Review:", review.content)

    # Ask follow-up
    await manager.add_message(
        role="user",
        content="What about performance?"
    )

    performance_review = await manager.complete()
    print("Performance notes:", performance_review.content)

    return manager.conversation_id

# Usage
async def main():
    manager = await create_code_review_assistant()

    code = """
    def process_data(items):
        result = []
        for item in items:
            result.append(item * 2)
        return result
    """

    conv_id = await review_code(manager, code)
    print(f"Review saved as conversation: {conv_id}")

# Run
import asyncio
asyncio.run(main())
```

### Example 2: Customer Support Bot with Branching

```python
"""
Customer support bot that explores multiple solution paths.
"""
from dataknobs_llm.llm import AnthropicProvider
from dataknobs_llm.prompts import AsyncPromptBuilder, ConfigPromptLibrary
from dataknobs_llm.conversations import (
    ConversationManager,
    DataknobsConversationStorage,
    ContentFilterMiddleware
)
from dataknobs_data.backends import AsyncSQLiteDatabase

async def create_support_bot():
    # Define prompts
    config = {
        "system": {
            "support_agent": {
                "template": "You are a helpful {{company}} support agent."
            }
        },
        "user": {
            "customer_issue": {
                0: {"template": "Customer issue: {{issue}}"}
            }
        }
    }

    library = ConfigPromptLibrary(config)
    builder = AsyncPromptBuilder(library=library)

    llm = AnthropicProvider(config={
        "api_key": "your-key",
        "model": "claude-3-sonnet-20240229"
    })

    storage = DataknobsConversationStorage(
        AsyncSQLiteDatabase(db_path="support.db")
    )

    # Filter inappropriate content
    middleware = [
        ContentFilterMiddleware(
            filter_words=["spam", "abuse"],
            replacement="[filtered]",
            case_sensitive=False
        )
    ]

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="support_agent",
        system_params={"company": "TechCorp"},
        middleware=middleware,
        metadata={"channel": "chat"}
    )

    return manager

async def handle_support_request(manager, issue: str):
    # Initial issue
    await manager.add_message(
        role="user",
        prompt_name="customer_issue",
        params={"issue": issue}
    )

    # Get initial solution
    solution1 = await manager.complete(branch_name="solution-A")
    print("Solution A:", solution1.content)

    # Try alternative approach
    await manager.switch_to_node("0")  # Back to initial issue
    solution2 = await manager.complete(branch_name="solution-B")
    print("Solution B:", solution2.content)

    # Get all branches
    branches = manager.get_branches()
    print(f"\nExplored {len(branches)} solution paths")

    return manager.conversation_id

async def main():
    manager = await create_support_bot()

    conv_id = await handle_support_request(
        manager,
        "My login isn't working"
    )

    print(f"Support ticket: {conv_id}")

import asyncio
asyncio.run(main())
```

### Example 3: Document Analysis with RAG

```python
"""
Document analysis assistant with RAG integration.
"""
from dataknobs_llm.llm import OpenAIProvider
from dataknobs_llm.prompts import (
    AsyncPromptBuilder,
    FileSystemPromptLibrary,
    AsyncDataknobsBackendAdapter
)
from dataknobs_llm.conversations import ConversationManager, DataknobsConversationStorage
from dataknobs_data.backends import AsyncMemoryDatabase
from dataknobs_data.records import Record

async def setup_document_database():
    """Create database with sample documents."""
    db = AsyncMemoryDatabase()

    # Add documents
    docs = [
        {"id": "1", "content": "Python asyncio enables concurrent programming", "category": "python"},
        {"id": "2", "content": "React hooks simplify state management", "category": "javascript"},
        {"id": "3", "content": "PostgreSQL supports JSONB for flexible schemas", "category": "database"}
    ]

    for doc in docs:
        record = Record(data=doc, storage_id=doc["id"])
        await db.upsert(doc["id"], record)

    return db

async def create_analysis_assistant():
    # Setup document database
    docs_db = await setup_document_database()

    # Create adapter
    docs_adapter = AsyncDataknobsBackendAdapter(
        docs_db,
        text_field="content",
        metadata_fields=["category"]
    )

    # Load prompts (with RAG config)
    library = FileSystemPromptLibrary(Path("prompts/analysis/"))

    # Create builder with adapter
    builder = AsyncPromptBuilder(
        library=library,
        adapters={"docs": docs_adapter}
    )

    llm = OpenAIProvider(config={"api_key": "your-key"})
    storage = DataknobsConversationStorage(AsyncMemoryDatabase())

    manager = await ConversationManager.create(
        llm=llm,
        prompt_builder=builder,
        storage=storage,
        system_prompt_name="document_analyzer"  # Uses RAG
    )

    return manager

async def analyze_topic(manager, topic: str):
    await manager.add_message(
        role="user",
        prompt_name="analyze_topic",
        params={"topic": topic}
        # RAG automatically retrieves relevant documents
    )

    # Stream analysis
    print(f"Analysis of '{topic}':\n")
    async for chunk in manager.stream_complete():
        print(chunk.delta, end="", flush=True)
    print("\n")

async def main():
    manager = await create_analysis_assistant()

    # Analyze different topics
    await analyze_topic(manager, "async programming")
    await analyze_topic(manager, "state management")

import asyncio
asyncio.run(main())
```

---

## Troubleshooting

### Common Issues

#### Issue: "Prompt not found"

```python
# Check if prompt exists
prompts = library.list_system_prompts()
print("Available system prompts:", prompts)

prompts = library.list_user_prompts()
print("Available user prompts:", prompts)
```

#### Issue: "Validation failed - missing required parameter"

```python
# Check what parameters are required
prompt = library.get_user_prompt("prompt_name")
if prompt and "validation" in prompt:
    print("Required params:", prompt["validation"].get("required_params", []))

# Provide all required parameters
result = await builder.render_user_prompt(
    "prompt_name",
    params={"required_param": "value"}
)
```

#### Issue: "Wrong adapter type for builder"

```python
# AsyncPromptBuilder requires async adapters
from dataknobs_llm.prompts import AsyncDictResourceAdapter

# NOT DictResourceAdapter (that's for sync PromptBuilder)
adapter = AsyncDictResourceAdapter(data)

builder = AsyncPromptBuilder(
    library=library,
    adapters={"data": adapter}
)
```

#### Issue: "Conversation not found when resuming"

```python
# Check if conversation exists
state = await storage.load_conversation(conversation_id)
if state is None:
    print(f"Conversation {conversation_id} not found")
    # Create new conversation instead
    manager = await ConversationManager.create(...)
else:
    # Resume existing conversation
    manager = await ConversationManager.resume(
        conversation_id=conversation_id,
        llm=llm,
        prompt_builder=builder,
        storage=storage
    )
```

#### Issue: "RAG not working"

```python
# Ensure RAG is enabled
result = await builder.render_system_prompt(
    "prompt_name",
    params={...},
    include_rag=True  # Must be True
)

# Check RAG config exists
rag_config = library.get_rag_config("config_name")
print("RAG config:", rag_config)

# Verify adapter is registered
print("Available adapters:", builder.adapters.keys())
```

### Performance Tips

1. **Use async for better concurrency**
   ```python
   # Good - parallel RAG searches
   builder = AsyncPromptBuilder(library, adapters)

   # Slower - sequential operations
   builder = PromptBuilder(library, adapters)
   ```

2. **Reuse LLM and builder instances**
   ```python
   # Create once, use many times
   llm = OpenAIProvider(config)
   builder = AsyncPromptBuilder(library)

   # Don't recreate for each request
   ```

3. **Choose appropriate storage backend**
   ```python
   # Development: Memory or SQLite
   storage = DataknobsConversationStorage(AsyncMemoryDatabase())

   # Production: PostgreSQL
   storage = DataknobsConversationStorage(AsyncPostgresDatabase(...))
   ```

4. **Limit conversation history**
   ```python
   # For very long conversations, consider summarization
   messages = manager.state.get_current_messages()
   if len(messages) > 50:
       # Implement summarization or pruning
       pass
   ```

### Getting Help

- **Documentation**: Check this guide and API docstrings
- **Examples**: See complete examples above and in test files
- **Issues**: Report bugs at https://github.com/kbs-labs/dataknobs/issues
- **Tests**: Review test files for additional usage patterns

---

## Next Steps

1. **Read Best Practices Guide** - Learn patterns for production use
2. **Review Examples** - Study complete examples above
3. **Experiment** - Try building your own prompts and conversations
4. **Explore Tests** - See `tests/prompts/` and `tests/conversations/` for more examples

---

**Happy prompting!**
