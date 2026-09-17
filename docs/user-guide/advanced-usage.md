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
        "storage": {"backend": "postgres", "connection_string": "..."},
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
results = search.search(Query().filter("content", "like", "%data%"))
```

### Async High-Performance Operations

```python
from dataknobs_data import async_database_factory, Record, Query

async def process_large_dataset():
    # Create async database using factory
    db = async_database_factory.create(
        backend="postgres",
        connection_string="postgresql://...",
        max_pool_size=20,
    )

    # Batch create with pooling
    records = [Record({"id": i, "data": f"item{i}"}) for i in range(10000)]
    await db.create_batch(records)

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
db = database_factory.create(
    backend="postgres",
    connection_string="postgresql://...",
)

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

# Resources are passed in at construction, or registered before processing --
# there is no fsm.context to assign into.
fsm = SimpleFSM(config, resources={"database": db})
# equivalently: fsm.register_resource("database", db)
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
    db = database_factory.create(
        backend="postgres",
        connection_string="postgresql://...",
    )

    # Create bots for different tenants
    support_bot = await DynaBot.from_config({
        "llm": {"provider": "openai", "model": "gpt-4"},
        "conversation_storage": {"backend": "postgres", "connection_string": "postgresql://..."},
        "memory": {"type": "buffer", "max_messages": 20},
        "system_prompt": "You are a helpful support agent."
    })

    sales_bot = await DynaBot.from_config({
        "llm": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        "conversation_storage": {"backend": "postgres", "connection_string": "postgresql://..."},
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
    knowledge_base = database_factory.create(
        backend="elasticsearch",
        host="localhost:9200",
        index="documentation",
    )

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

# The string form is parenthesized, not arrow-separated. Anything that does
# not start with "(" is taken as a single node's data, so an arrow string
# parses without error into a childless node holding the whole string.
tree1 = build_tree_from_string("(root a b)")
tree2 = build_tree_from_string("(other c d)")

# There is no Tree.merge. Grafting is add_child, which detaches each node
# from its current parent on the way -- so iterate over a snapshot of the
# source's children rather than the list you are emptying.
for node in tree2.children or ():
    tree1.add_child(node)
print(tree1.as_string())   # (root a b c d)
print(tree2.as_string())   # other

# There is no extract_subtree either. prune() detaches a node and returns its
# former parent; the subtree below it stays intact and is usable on its own.
a = tree1.find_nodes(lambda n: n.data == "a", only_first=True)[0]
a.prune()
print(a.as_string())       # a
print(tree1.as_string())   # (root b c d)
```

### Tree Serialization

```python
import json

from dataknobs_structures import build_tree_from_string

# There is no to_json/from_json. The built-in round trip is the parenthesized
# string form: as_string() writes it and build_tree_from_string() reads it.
tree = build_tree_from_string("(root child1 child2)")
restored = build_tree_from_string(tree.as_string())
print(restored.as_string() == tree.as_string())   # True

# Nodes are rebuilt from their *string* form, so a tree carrying non-string
# data does not survive that round trip unchanged. Write your own walk when
# the payload matters -- a node exposes `data` and `children`, and `children`
# is None until a node first holds a child.
def to_record(node):
    return {
        "data": node.data,
        "children": [to_record(child) for child in node.children or ()],
    }

serialized = json.dumps(to_record(tree))
# {"data": "root", "children": [{"data": "child1", "children": []}, ...]}
```

## Advanced Text Processing

### Custom Tokenizers

```python
import re

from dataknobs_xization import masking_tokenizer

# There is no MaskingTokenizer base class to subclass and no add_pattern hook.
# Tokenizing is TextFeatures; classifying the tokens it yields is your own pass
# over them, which keeps the pattern set in your code rather than in a subclass.
PATTERNS = (
    ("ACRONYM", re.compile(r"^[A-Z]{2,}$"), "[ACRONYM]"),
    ("NUMBER", re.compile(r"^\d+$"), "[NUM]"),
)

def classify(token_text):
    for name, pattern, replacement in PATTERNS:
        if pattern.match(token_text):
            return name, replacement
    return None, token_text

text = "IBM costs 150.99 per share"
features = masking_tokenizer.TextFeatures(
    text, mark_alpha=True, mark_digit=True, emoji_data=None
)

masked = [classify(token.token_text)[1] for token in features.get_tokens()]
print(masked)
# ['[ACRONYM]', 'costs', '[NUM]', '[NUM]', 'per', 'share']

# Note the two [NUM]s: the period is a delimiter, so "150.99" tokenizes as
# "150" and "99". A pattern spanning a delimiter cannot match a single token,
# which is why multi-token entities are matched over the token *sequence*
# rather than over token text -- see the lexicon and annotations modules.
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
    
    def process(self, text, text_id):
        # TextMetaData requires a text_id
        doc = Text(text, TextMetaData(text_id))

        for annotator in self.annotators:
            annotator.annotate_input(doc)

        return doc

# There is no NamedEntityAnnotator, SentimentAnnotator or LanguageDetector.
# What the package ships is the contract: Annotator is abstract on
# annotate_input, and BasicAnnotator and EntityAnnotator are abstract on more
# still, so each of them is a base you subclass rather than a class you
# instantiate. CompoundAnnotator is the one concrete type, and it runs a
# series of annotators through an AnnotatorKernel.
class KeywordAnnotator(annotations.Annotator):
    def __init__(self, name, keywords):
        super().__init__(name)
        self.keywords = keywords

    def annotate_input(self, text_obj, **kwargs):
        # Return the annotations added; the real signature takes an
        # AnnotatedText and returns an Annotations.
        return [word for word in text_obj.text.split() if word in self.keywords]

pipeline = AnnotationPipeline()
pipeline.add_annotator(KeywordAnnotator("person", {"John", "Smith"}))
pipeline.add_annotator(KeywordAnnotator("language", {"Python"}))

# Process text
result = pipeline.process("John Smith loves Python programming.", "doc_001")
```

## Advanced Elasticsearch Integration

### Bulk Operations

```python
import tempfile
from pathlib import Path

from dataknobs_utils import elasticsearch_utils

class BulkIndexer:
    def __init__(self, batchfile_path, index_name):
        self.batchfile_path = batchfile_path
        self.index = index_name
        self.buffer = []
        self.buffer_size = 1000
        self.next_id = 1
    
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
        
        # There is no bulk_index(). This module WRITES the NDJSON bulk file
        # that Elasticsearch's bulk API consumes; add_batch_data takes an open
        # file handle and a generator of source records, and returns the next
        # id to use, so successive flushes can continue the numbering.
        with open(self.batchfile_path, "a") as batchfile:
            self.next_id = elasticsearch_utils.add_batch_data(
                batchfile, iter(self.buffer), self.index, cur_id=self.next_id
            )
        self.buffer.clear()

# Usage. The file is opened for append so successive flushes accumulate, which
# also means a fresh path per run rather than one in the working directory.
scratch = Path(tempfile.mkdtemp()) / "bulk_payload.ndjson"
indexer = BulkIndexer(scratch, "my_index")
for i in range(10000):
    indexer.add({"id": i, "data": f"Document {i}"})
indexer.flush()

print(indexer.next_id)                              # 10001
print(sum(1 for _ in scratch.open()))               # 20000
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

# Hand the built body to the index; search() answers a ServerResponse, so
# read .result rather than subscripting it.
index = elasticsearch_utils.SimplifiedElasticsearchIndex("docs")
response = index.search(query)
results = response.result["hits"]["hits"] if response.succeeded else []
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
        # find_nodes returns a materialised list, and `data` is the payload --
        # there is no traverse() and no node.id
        for node in tree.find_nodes(lambda n: True):
            result = self.process_node(node.data, "analyze")
            results.append(result)
        return results
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from dataknobs_xization.normalize import basic_normalization_fn

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
