# RAG Query Processing

This module provides query transformation and expansion utilities for improved RAG retrieval.

## Overview

User queries often don't match well with embedded content:
- Ambiguous follow-ups: "Show me an example"
- Literal content to analyze: "Analyze this prompt: Write a poem"
- Context-dependent questions: "What about the second option?"

Query processing transforms these into effective search queries.

## Quick Start

```python
from dataknobs_bots.knowledge.query import (
    QueryTransformer,
    TransformerConfig,
    ContextualExpander,
    is_ambiguous_query,
)

# Option 1: LLM-based transformation
transformer = QueryTransformer(TransformerConfig(
    enabled=True,
    llm_provider="ollama",
    llm_model="llama3.2",
    domain_context="prompt engineering"
))
await transformer.initialize()
queries = await transformer.transform("Analyze this: Write a poem")

# Option 2: No-LLM expansion using conversation context
expander = ContextualExpander(max_context_turns=3)
if is_ambiguous_query("Show me an example"):
    expanded = expander.expand("Show me an example", conversation_history)
```

## Two Approaches

### QueryTransformer (LLM-based)

Uses an LLM to generate optimized search queries. Best for:
- Complex queries needing interpretation
- Domain-specific terminology
- Multi-faceted questions
- When latency is acceptable

### ContextualExpander (No LLM)

Expands queries using conversation history keywords. Best for:
- Ambiguous follow-up questions
- Context-dependent queries
- Low-latency requirements
- When LLM calls should be minimized

## QueryTransformer

### Configuration

```python
from dataknobs_bots.knowledge.query import TransformerConfig

config = TransformerConfig(
    enabled=True,              # Enable transformation
    llm_provider="ollama",     # LLM provider
    llm_model="llama3.2",      # Model to use
    num_queries=3,             # Number of queries to generate
    domain_context="prompt engineering"  # Domain hint
)
```

### Basic Usage

```python
from dataknobs_bots.knowledge.query import QueryTransformer, TransformerConfig

# Create and initialize
transformer = QueryTransformer(TransformerConfig(
    enabled=True,
    llm_provider="ollama",
    llm_model="llama3.2",
    num_queries=3,
    domain_context="prompt engineering"
))
await transformer.initialize()

# Transform user input
queries = await transformer.transform(
    "Analyze this prompt: Write a Python function to sort a list"
)
# Returns: [
#   "prompt analysis techniques",
#   "evaluating prompt quality",
#   "code generation prompt patterns"
# ]

# Search with each query
all_results = []
for query in queries:
    results = await kb.search(query)
    all_results.extend(results)
```

### With Conversation Context

```python
# Transform with conversation history
queries = await transformer.transform_with_context(
    user_input="What about improving it?",
    conversation_context="User: How do I write prompts for code generation?\n"
                        "Assistant: Here are the key techniques...",
    num_queries=3
)
```

### Cleanup

```python
# Close when done
await transformer.close()
```

## ContextualExpander

### Basic Usage

```python
from dataknobs_bots.knowledge.query import ContextualExpander, Message

expander = ContextualExpander(
    max_context_turns=3,      # How many recent messages to use
    include_assistant=False,   # Include assistant messages
    keyword_weight=2           # Keyword repetition factor
)

# Conversation history
history = [
    Message(role="user", content="Tell me about chain-of-thought prompting"),
    Message(role="assistant", content="Chain-of-thought is a technique..."),
    Message(role="user", content="How does it compare to few-shot?"),
]

# Expand ambiguous query
expanded = expander.expand(
    "Show me an example",
    history
)
# Returns: "chain-of-thought prompting compare few-shot Show me an example"
```

### With Dict History

```python
# Also accepts dict format
history = [
    {"role": "user", "content": "Tell me about chain-of-thought"},
    {"role": "assistant", "content": "It's a technique..."},
]

expanded = expander.expand("Show me more", history)
```

### Custom Topic Extraction

```python
def extract_technical_terms(text):
    """Custom topic extractor."""
    # Your extraction logic
    return ["chain-of-thought", "prompting", "few-shot"]

expanded = expander.expand_with_topics(
    "Show me an example",
    history,
    topic_extractor=extract_technical_terms
)
```

## Ambiguity Detection

### is_ambiguous_query

Check if a query needs expansion:

```python
from dataknobs_bots.knowledge.query import is_ambiguous_query

is_ambiguous_query("Show me an example")     # True (short + "example")
is_ambiguous_query("What about this one?")   # True (demonstrative)
is_ambiguous_query("How do I configure OAuth?")  # False (specific)
is_ambiguous_query("More")                   # True (very short)
```

### Integration Pattern

```python
from dataknobs_bots.knowledge.query import ContextualExpander, is_ambiguous_query

class SmartKnowledgeBase:
    def __init__(self):
        self.expander = ContextualExpander()

    async def query(self, message, history=None, k=10):
        # Expand if ambiguous and we have history
        if history and is_ambiguous_query(message):
            search_query = self.expander.expand(message, history)
        else:
            search_query = message

        return await self.vector_store.search(search_query, k=k)
```

## Complete Integration

### RAG Pipeline with Query Processing

```python
from dataknobs_bots.knowledge.query import (
    ContextualExpander,
    is_ambiguous_query,
)
from dataknobs_bots.knowledge.retrieval import ChunkMerger, ContextFormatter

class EnhancedRAGKnowledgeBase:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.expander = ContextualExpander(max_context_turns=3)
        self.merger = ChunkMerger()
        self.formatter = ContextFormatter()

    async def query(
        self,
        message: str,
        history: list | None = None,
        k: int = 10
    ):
        # 1. Expand ambiguous queries
        if history and is_ambiguous_query(message):
            search_query = self.expander.expand(message, history)
        else:
            search_query = message

        # 2. Search
        results = await self.vector_store.search(search_query, k=k)

        # 3. Merge adjacent chunks
        merged = self.merger.merge(results)

        return merged

    def format_context(self, merged_chunks):
        context = self.formatter.format_merged(merged_chunks)
        return self.formatter.wrap_for_prompt(context)
```

### With LLM Query Transformation

```python
from dataknobs_bots.knowledge.query import QueryTransformer, TransformerConfig

class TransformingRAGKnowledgeBase:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.transformer = QueryTransformer(TransformerConfig(
            enabled=True,
            llm_provider="ollama",
            llm_model="llama3.2",
            num_queries=3,
            domain_context="documentation"
        ))

    async def initialize(self):
        await self.transformer.initialize()

    async def query(self, message: str, k: int = 10):
        # Generate multiple search queries
        queries = await self.transformer.transform(message)

        # Search with each query
        all_results = []
        seen_ids = set()

        for query in queries:
            results = await self.vector_store.search(query, k=k)
            for result in results:
                # Deduplicate
                result_id = result.get("id") or hash(result.get("text", ""))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    all_results.append(result)

        # Sort by similarity and return top k
        all_results.sort(key=lambda r: r.get("similarity", 0), reverse=True)
        return all_results[:k]

    async def close(self):
        await self.transformer.close()
```

### In DynaBot

```python
class EnhancedBot(DynaBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_expander = ContextualExpander()

    async def chat(
        self,
        message: str,
        context: BotContext,
        rag_query: str | None = None,
        **kwargs
    ):
        # Use explicit rag_query if provided
        if rag_query:
            search_query = rag_query
        # Otherwise, expand if ambiguous
        elif is_ambiguous_query(message):
            history = await self.get_conversation_history(context)
            search_query = self.query_expander.expand(message, history)
        else:
            search_query = message

        # Continue with retrieval
        return await super().chat(
            message,
            context,
            rag_query=search_query,
            **kwargs
        )
```

## API Reference

### TransformerConfig

```python
@dataclass
class TransformerConfig:
    enabled: bool = False           # Enable transformation
    llm_provider: str = "ollama"    # LLM provider
    llm_model: str = "llama3.2"     # Model name
    num_queries: int = 3            # Queries to generate
    domain_context: str = ""        # Domain hint
```

### QueryTransformer

```python
class QueryTransformer:
    def __init__(self, config: TransformerConfig | None = None):
        """Initialize with optional configuration."""

    async def initialize(self) -> None:
        """Initialize the LLM provider."""

    async def close(self) -> None:
        """Close the LLM provider."""

    async def transform(
        self,
        user_input: str,
        num_queries: int | None = None,
    ) -> list[str]:
        """Transform user input into optimized search queries."""

    async def transform_with_context(
        self,
        user_input: str,
        conversation_context: str,
        num_queries: int | None = None,
    ) -> list[str]:
        """Transform with additional conversation context."""
```

### Message

```python
@dataclass
class Message:
    role: str      # "user", "assistant", "system"
    content: str   # Message content
```

### ContextualExpander

```python
class ContextualExpander:
    def __init__(
        self,
        max_context_turns: int = 3,
        include_assistant: bool = False,
        keyword_weight: int = 2,
    ):
        """Initialize the contextual expander."""

    def expand(
        self,
        user_input: str,
        conversation_history: list[Message] | list[dict],
    ) -> str:
        """Expand query with conversation context."""

    def expand_with_topics(
        self,
        user_input: str,
        conversation_history: list[Message] | list[dict],
        topic_extractor: callable = None,
    ) -> str:
        """Expand with custom topic extraction."""
```

### Utility Functions

```python
def is_ambiguous_query(query: str) -> bool:
    """Check if a query is likely ambiguous and needs expansion."""

async def create_transformer(config: dict) -> QueryTransformer:
    """Create and initialize a QueryTransformer from config dict."""
```

## Related

- [Retrieval Utilities](rag-retrieval.md) - Chunk merging and formatting
- [User Guide](user-guide.md) - Complete bot usage guide
