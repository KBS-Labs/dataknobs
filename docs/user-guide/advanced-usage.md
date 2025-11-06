# Advanced Usage

This guide covers advanced features and patterns for power users, including the heavier packages for AI, workflows, and data processing.

## Configuration Management

### Environment-Aware Configuration

```python
from dataknobs_config import Config
from dataknobs_data import database_factory

# config.yaml with environment variables
# databases:
#   primary:
#     backend: ${DB_BACKEND:memory}
#     host: ${DB_HOST:localhost}
#     port: ${DB_PORT:5432}

config = Config("config.yaml")
config.register_factory("database", database_factory)

# Backend chosen based on environment
db = config.get_instance("databases", "primary")
```

### Factory Pattern for Dynamic Objects

```python
from dataknobs_config import Config

def custom_processor_factory(config_dict):
    processor_type = config_dict.get("type")
    if processor_type == "fast":
        return FastProcessor(**config_dict)
    elif processor_type == "accurate":
        return AccurateProcessor(**config_dict)

config = Config({"processors": {"main": {"type": "fast"}}})
config.register_factory("processor", custom_processor_factory)
processor = config.get_instance("processors", "main")
```

[Learn more →](../packages/config/index.md)

## Data Abstraction

### Multi-Backend Applications

```python
from dataknobs_data import database_factory, Record, Query
from dataknobs_config import Config

config = Config({
    "databases": {
        "cache": {"backend": "memory"},
        "storage": {"backend": "postgres", "connection": "..."},
        "search": {"backend": "elasticsearch", "host": "..."}
    }
})
config.register_factory("database", database_factory)

# Use different backends for different purposes
cache = config.get_instance("databases", "cache")
storage = config.get_instance("databases", "storage")
search = config.get_instance("databases", "search")

# Same API across all backends
record = Record({"id": "123", "content": "data"})
cache.create(record)
storage.create(record)
search.create(record)

# Query with same interface
results = search.search(Query().filter("content", "contains", "data"))
```

### Async High-Performance Operations

```python
from dataknobs_data import async_database_factory, Record, Query

async def process_large_dataset():
    # Create async database using factory
    db = async_database_factory.create({
        "backend": "postgres",
        "connection": "postgresql://...",
        "pool_size": 20
    })

    # Batch create with pooling
    records = [Record({"id": i, "data": f"item{i}"}) for i in range(10000)]
    await db.bulk_create(records, batch_size=100)

    # Async iteration over large result sets
    async for record in db.stream(Query()):
        await process_record(record)

    await db.close()
```

[Learn more →](../packages/data/index.md)

## Workflow Orchestration with FSM

### Complex Multi-Stage Pipelines

```python
from dataknobs_fsm import SimpleFSM, DataHandlingMode

config = {
    "name": "etl_pipeline",
    "states": [
        {"name": "extract", "is_start": True},
        {"name": "validate"},
        {"name": "transform"},
        {"name": "enrich"},
        {"name": "load", "is_end": True},
        {"name": "error"}
    ],
    "arcs": [
        {
            "from": "extract",
            "to": "validate",
            "transform": {
                "type": "builtin",
                "name": "extract_from_api",
                "params": {"url": "https://api.example.com/data"}
            }
        },
        {
            "from": "validate",
            "to": "transform",
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: data.get('valid', False)"
            }
        },
        {
            "from": "validate",
            "to": "error",
            "pre_test": {
                "type": "inline",
                "code": "lambda data, ctx: not data.get('valid', False)"
            }
        },
        {"from": "transform", "to": "enrich"},
        {"from": "enrich", "to": "load"}
    ]
}

fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"source": "api"})
```

### FSM with Resource Management

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_data import database_factory

# Create database using factory for FSM context
db = database_factory.create({
    "backend": "postgres",
    "connection": "postgresql://..."
})

config = {
    "name": "db_processor",
    "states": [
        {"name": "load", "is_start": True},
        {"name": "process", "is_end": True}
    ],
    "arcs": [
        {
            "from": "load",
            "to": "process",
            "transform": {
                "type": "inline",
                "code": "lambda data, ctx: ctx['database'].search(Query())"
            }
        }
    ]
}

fsm = SimpleFSM(config)
fsm.context["database"] = db
result = fsm.process(data)
```

[Learn more →](../packages/fsm/index.md)

## LLM Integration

### Prompt Template Management

```python
from dataknobs_llm import create_llm_provider, MessageTemplate, MessageBuilder, LLMMessage

# Create message templates for reusable prompts
summarize_template_v1 = MessageTemplate(
    "Summarize the following in {max_words} words:\n\n{text}"
)
summarize_template_v2 = MessageTemplate(
    "Provide a {max_words}-word summary of:\n{text}\n\nFocus on key points."
)

# Use LLM with templates
llm = create_llm_provider({"provider": "openai", "model": "gpt-4"})

# Build messages from template
builder = MessageBuilder()
builder.add_user_message(summarize_template_v2.format(
    text="Long article content...",
    max_words=50
))

response = await llm.generate(builder.messages)
```

### Tool Calling with LLMs

```python
from dataknobs_llm import create_llm_provider, Tool, ToolRegistry, LLMMessage
from typing import Dict, Any

# Define custom tools by subclassing Tool
class DatabaseSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="search_database",
            description="Search the database for relevant information"
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> list:
        # Implementation
        return results

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculate",
            description="Evaluate a mathematical expression"
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }

    async def execute(self, expression: str) -> float:
        # Implementation
        return result

# Register tools
registry = ToolRegistry()
registry.register(DatabaseSearchTool())
registry.register(CalculatorTool())

# Use LLM with tools
llm = create_llm_provider({
    "provider": "openai",
    "model": "gpt-4"
})

messages = [LLMMessage(
    role="user",
    content="What's 15% of the revenue from last quarter?"
)]
response = await llm.generate(messages, tools=registry.get_all())
```

[Learn more →](../packages/llm/index.md)

## AI Agents and Chatbots

### Multi-Tenant Bot System

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext
from dataknobs_data import database_factory

async def main():
    # Persistent storage using factory
    db = database_factory.create({
        "backend": "postgres",
        "connection": "postgresql://..."
    })

    # Create bots for different tenants
    support_bot = await DynaBot.from_config({
        "llm": {"provider": "openai", "model": "gpt-4"},
        "conversation_storage": {"backend": "postgres", "connection": "postgresql://..."},
        "memory": {"type": "buffer", "max_messages": 20},
        "system_prompt": "You are a helpful support agent."
    })

    sales_bot = await DynaBot.from_config({
        "llm": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        "conversation_storage": {"backend": "postgres", "connection": "postgresql://..."},
        "memory": {"type": "buffer", "max_messages": 20},
        "system_prompt": "You are a sales assistant."
    })

    # Use bots with context isolation
    support_context = BotContext(
        conversation_id="support-001",
        client_id="tenant1",
        user_id="user1"
    )
    support_response = await support_bot.chat("Help me reset password", support_context)

    sales_context = BotContext(
        conversation_id="sales-001",
        client_id="tenant2",
        user_id="user2"
    )
    sales_response = await sales_bot.chat("Tell me about pricing", sales_context)

asyncio.run(main())
```

### RAG-Enabled Chatbot

```python
import asyncio
from dataknobs_bots import DynaBot, BotContext
from dataknobs_data import database_factory

async def main():
    # Knowledge base using factory
    knowledge_base = database_factory.create({
        "backend": "elasticsearch",
        "host": "localhost:9200",
        "index": "documentation"
    })

    bot_config = {
        "llm": {"provider": "openai", "model": "gpt-4"},
        "conversation_storage": {"backend": "memory"},
        "memory": {"type": "buffer", "max_messages": 10},
        "rag": {
            "enabled": True,
            "knowledge_base": knowledge_base,
            "top_k": 5,
            "score_threshold": 0.7
        },
        "system_prompt": "Answer questions using the provided documentation."
    }

    bot = await DynaBot.from_config(bot_config)

    # Bot retrieves relevant docs before answering
    context = BotContext(
        conversation_id="docs-001",
        client_id="my-app",
        user_id="user1"
    )
    response = await bot.chat("How do I configure the database?", context)
    print(response)

asyncio.run(main())
```

[Learn more →](../packages/bots/index.md)

## Advanced Tree Operations

### Tree Merging and Splitting

```python
from dataknobs_structures import Tree, build_tree_from_string

# Merge multiple trees
tree1 = build_tree_from_string("root -> a, b")
tree2 = build_tree_from_string("root -> c, d")

merged = Tree.merge(tree1, tree2)
# Result: root -> a, b, c, d

# Split tree at a node
subtree = tree1.extract_subtree("a")
```

### Tree Serialization

```python
import json
from dataknobs_structures import Tree

# Serialize tree to JSON
tree = build_tree_from_string("root -> child1, child2")
tree_json = tree.to_json()

# Deserialize from JSON
restored_tree = Tree.from_json(tree_json)

# Custom serialization format
def custom_serializer(node):
    return {
        "id": node.id,
        "value": node.value,
        "children": [custom_serializer(c) for c in node.children]
    }

serialized = custom_serializer(tree.root)
```

## Advanced Text Processing

### Custom Tokenizers

```python
from dataknobs_xization import masking_tokenizer

class CustomTokenizer(masking_tokenizer.MaskingTokenizer):
    def __init__(self):
        super().__init__()
        self.add_pattern(r'\b[A-Z]{2,}\b', 'ACRONYM')
        self.add_pattern(r'\$\d+\.\d{2}', 'CURRENCY')
    
    def mask_token(self, token, token_type):
        if token_type == 'ACRONYM':
            return '[ACRONYM]'
        elif token_type == 'CURRENCY':
            return '[MONEY]'
        return super().mask_token(token, token_type)

tokenizer = CustomTokenizer()
text = "IBM costs $150.99 per share"
tokens = tokenizer.tokenize(text)
# Output: ["[ACRONYM]", "costs", "[MONEY]", "per", "share"]
```

### Text Annotation Pipeline

```python
from dataknobs_xization import annotations
from dataknobs_structures import Text, TextMetaData

class AnnotationPipeline:
    def __init__(self):
        self.annotators = []
    
    def add_annotator(self, annotator):
        self.annotators.append(annotator)
    
    def process(self, text):
        metadata = TextMetaData()
        doc = Text(text, metadata)
        
        for annotator in self.annotators:
            doc = annotator.annotate(doc)
        
        return doc

# Create pipeline
pipeline = AnnotationPipeline()
pipeline.add_annotator(NamedEntityAnnotator())
pipeline.add_annotator(SentimentAnnotator())
pipeline.add_annotator(LanguageDetector())

# Process text
result = pipeline.process("John Smith loves Python programming.")
```

## Advanced Elasticsearch Integration

### Bulk Operations

```python
from dataknobs_utils import elasticsearch_utils

class BulkIndexer:
    def __init__(self, es_client, index_name):
        self.es = es_client
        self.index = index_name
        self.buffer = []
        self.buffer_size = 1000
    
    def add(self, doc):
        self.buffer.append(doc)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        
        actions = []
        for doc in self.buffer:
            actions.append({
                "_index": self.index,
                "_source": doc
            })
        
        elasticsearch_utils.bulk_index(self.es, actions)
        self.buffer.clear()

# Usage
indexer = BulkIndexer(es_client, "my_index")
for i in range(10000):
    indexer.add({"id": i, "data": f"Document {i}"})
indexer.flush()
```

### Custom Query Builders

```python
from dataknobs_utils import elasticsearch_utils

class QueryBuilder:
    def __init__(self):
        self.query = {"bool": {}}
    
    def must(self, clause):
        if "must" not in self.query["bool"]:
            self.query["bool"]["must"] = []
        self.query["bool"]["must"].append(clause)
        return self
    
    def should(self, clause):
        if "should" not in self.query["bool"]:
            self.query["bool"]["should"] = []
        self.query["bool"]["should"].append(clause)
        return self
    
    def filter(self, clause):
        if "filter" not in self.query["bool"]:
            self.query["bool"]["filter"] = []
        self.query["bool"]["filter"].append(clause)
        return self
    
    def build(self):
        return {"query": self.query}

# Build complex query
query = (QueryBuilder()
    .must({"match": {"title": "python"}})
    .filter({"range": {"date": {"gte": "2024-01-01"}}})
    .should({"match": {"tags": "tutorial"}})
    .build())

results = es_client.search(index="docs", body=query)
```

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
from dataknobs_structures import Tree

class CachedTreeProcessor:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def process_node(self, node_id, operation):
        # Expensive operation
        result = self._compute(node_id, operation)
        return result
    
    def process_tree(self, tree):
        results = []
        for node in tree.traverse():
            result = self.process_node(node.id, "analyze")
            results.append(result)
        return results
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from dataknobs_xization import basic_normalization_fn

def process_batch(texts):
    return [basic_normalization_fn(text) for text in texts]

def parallel_normalize(all_texts, workers=4):
    batch_size = len(all_texts) // workers
    batches = [all_texts[i:i+batch_size] 
               for i in range(0, len(all_texts), batch_size)]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(process_batch, batches)
    
    return [item for batch in results for item in batch]

# Process large dataset
texts = [f"Text {i}" for i in range(10000)]
normalized = parallel_normalize(texts)
```

## Custom Extensions

### Plugin System

```python
from abc import ABC, abstractmethod

class DataknobsPlugin(ABC):
    @abstractmethod
    def initialize(self, config):
        pass
    
    @abstractmethod
    def process(self, data):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register(self, name, plugin_class):
        self.plugins[name] = plugin_class
    
    def load(self, name, config=None):
        if name not in self.plugins:
            raise ValueError(f"Plugin {name} not found")
        
        plugin = self.plugins[name]()
        plugin.initialize(config or {})
        return plugin

# Create custom plugin
class SentimentPlugin(DataknobsPlugin):
    def initialize(self, config):
        self.model = config.get("model", "default")
    
    def process(self, data):
        # Sentiment analysis logic
        return {"sentiment": "positive", "score": 0.8}
    
    def cleanup(self):
        pass

# Use plugin system
manager = PluginManager()
manager.register("sentiment", SentimentPlugin)

plugin = manager.load("sentiment", {"model": "advanced"})
result = plugin.process("I love this product!")
```

## Integration Patterns

### Service Integration

```python
from dataknobs_utils import json_utils
import requests

class DataknobsService:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def process_document(self, doc):
        # Send to external service
        response = requests.post(
            f"{self.base_url}/process",
            json={"document": doc}
        )
        
        # Parse response
        result = response.json()
        
        # Extract using json_utils
        entities = json_utils.get_value(result, "analysis.entities", [])
        sentiment = json_utils.get_value(result, "analysis.sentiment")
        
        return {
            "entities": entities,
            "sentiment": sentiment
        }
```

## Next Steps

- Review [Best Practices](best-practices.md) for production use
- Explore the [API Reference](../api/index.md) for complete details
- See [Examples](../examples/index.md) for real-world implementations
- Read about [Performance Tuning](../development/architecture.md)
