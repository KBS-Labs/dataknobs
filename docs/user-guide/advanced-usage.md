# Advanced Usage

This guide covers advanced features and patterns for power users.

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
