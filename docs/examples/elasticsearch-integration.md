# Elasticsearch Integration Examples

This guide demonstrates how to integrate Dataknobs with Elasticsearch for indexing and searching documents.

## Basic Setup

### Creating an Elasticsearch Connection

```python
from dataknobs_utils.elasticsearch_utils import ElasticsearchIndex, TableSettings
from dataknobs_utils.requests_utils import RequestHelper

# Configure Elasticsearch connection
es_config = {
    "host": "localhost",
    "port": 9200,
    "scheme": "http"
}

# Create request helper
request_helper = RequestHelper(
    host=es_config["host"],
    port=es_config["port"],
    scheme=es_config["scheme"]
)

# Define index settings
table_settings = TableSettings(
    table_name="documents",
    mappings={
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "author": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "tags": {"type": "keyword"},
            "score": {"type": "float"}
        }
    },
    settings={
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
)

# Create index
index = ElasticsearchIndex(request_helper, [table_settings])
```

## Document Indexing

### Indexing Single Documents

```python
from dataknobs_structures import Document
from dataknobs_utils import elasticsearch_utils
import json

def index_document(index, doc, doc_id=None):
    """Index a single document."""
    # Prepare document for indexing
    doc_data = {
        "content": doc.text,
        "metadata": doc.metadata,
        "timestamp": doc.metadata.get("timestamp", "2024-01-01T00:00:00Z")
    }
    
    # Index the document
    response = index.index_document(
        index_name="documents",
        doc_type="_doc",
        doc_id=doc_id,
        body=doc_data
    )
    
    return response

# Example usage
doc = Document(
    text="This is a sample document about Elasticsearch integration.",
    metadata={
        "title": "Elasticsearch Guide",
        "author": "John Doe",
        "tags": ["elasticsearch", "search", "indexing"]
    }
)

response = index_document(index, doc, doc_id="doc_001")
print(f"Indexed document: {response}")
```

### Bulk Indexing

```python
from dataknobs_structures import Document
import json

def bulk_index_documents(index, documents):
    """Index multiple documents in bulk."""
    bulk_data = []
    
    for i, doc in enumerate(documents):
        # Action metadata
        action = {
            "index": {
                "_index": "documents",
                "_id": f"doc_{i:04d}"
            }
        }
        bulk_data.append(json.dumps(action))
        
        # Document data
        doc_data = {
            "content": doc.text,
            "metadata": doc.metadata
        }
        bulk_data.append(json.dumps(doc_data))
    
    # Join with newlines for bulk API
    bulk_body = "\n".join(bulk_data) + "\n"
    
    # Perform bulk indexing
    response = index.bulk_operation(bulk_body)
    
    return response

# Example usage
documents = [
    Document(f"Document {i} content", metadata={"id": i})
    for i in range(100)
]

response = bulk_index_documents(index, documents)
print(f"Indexed {len(documents)} documents")
```

## Document Searching

### Basic Search Queries

```python
def search_documents(index, query_text, size=10):
    """Search for documents matching query."""
    query = {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["content", "metadata.title^2"],  # Boost title field
                "type": "best_fields"
            }
        },
        "size": size,
        "_source": ["content", "metadata"]
    }
    
    response = index.search(
        index_name="documents",
        body=query
    )
    
    # Parse results
    hits = response.get("hits", {}).get("hits", [])
    results = []
    for hit in hits:
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"]
        })
    
    return results

# Example usage
results = search_documents(index, "elasticsearch integration")
for result in results:
    print(f"Score: {result['score']:.2f} - {result['metadata'].get('title', 'Untitled')}")
```

### Advanced Search with Filters

```python
def advanced_search(index, query_text, filters=None, aggregations=None):
    """Advanced search with filters and aggregations."""
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["content", "metadata.title"]
                        }
                    }
                ]
            }
        }
    }
    
    # Add filters
    if filters:
        query["query"]["bool"]["filter"] = filters
    
    # Add aggregations
    if aggregations:
        query["aggs"] = aggregations
    
    # Example with author filter and tag aggregation
    query["query"]["bool"]["filter"] = [
        {"term": {"metadata.author": "John Doe"}}
    ]
    
    query["aggs"] = {
        "tags": {
            "terms": {
                "field": "metadata.tags",
                "size": 10
            }
        }
    }
    
    response = index.search(
        index_name="documents",
        body=query
    )
    
    return response

# Example usage
results = advanced_search(
    index,
    "search",
    filters=[{"range": {"timestamp": {"gte": "2024-01-01"}}}]
)
```

## Text Processing Pipeline

### Index Normalized Documents

```python
from dataknobs_structures import Document
from dataknobs_xization import normalize

class ElasticsearchPipeline:
    """Pipeline for processing and indexing documents."""
    
    def __init__(self, index):
        self.index = index
    
    def process_and_index(self, doc, doc_id=None):
        """Process document and index to Elasticsearch."""
        # Normalize text
        normalized_text = normalize.basic_normalization_fn(doc.text)
        
        # Prepare document
        doc_data = {
            "original_text": doc.text,
            "normalized_text": normalized_text,
            "metadata": doc.metadata,
            "word_count": len(normalized_text.split()),
            "char_count": len(normalized_text)
        }
        
        # Index document
        response = self.index.index_document(
            index_name="processed_documents",
            doc_type="_doc",
            doc_id=doc_id,
            body=doc_data
        )
        
        return response
    
    def search_normalized(self, query_text):
        """Search using normalized query."""
        # Normalize query
        normalized_query = normalize.basic_normalization_fn(query_text)
        
        query = {
            "query": {
                "match": {
                    "normalized_text": normalized_query
                }
            }
        }
        
        return self.index.search(
            index_name="processed_documents",
            body=query
        )

# Example usage
pipeline = ElasticsearchPipeline(index)

doc = Document(
    "getUserData&ProcessInput",
    metadata={"type": "code"}
)

pipeline.process_and_index(doc, "doc_001")
results = pipeline.search_normalized("get user data")
```

## Document Analysis

### Term Frequency Analysis

```python
def analyze_term_frequency(index, field="content", size=20):
    """Analyze term frequency across all documents."""
    query = {
        "size": 0,
        "aggs": {
            "term_frequency": {
                "terms": {
                    "field": f"{field}.keyword",
                    "size": size
                }
            }
        }
    }
    
    response = index.search(
        index_name="documents",
        body=query
    )
    
    # Extract term frequencies
    buckets = response["aggregations"]["term_frequency"]["buckets"]
    terms = [(b["key"], b["doc_count"]) for b in buckets]
    
    return terms

# Example usage
top_terms = analyze_term_frequency(index)
print("Top terms:")
for term, count in top_terms:
    print(f"  {term}: {count}")
```

### Document Similarity Search

```python
def find_similar_documents(index, doc_id, size=5):
    """Find documents similar to a given document."""
    # First, get the document
    source_doc = index.get_document(
        index_name="documents",
        doc_id=doc_id
    )
    
    # Use more_like_this query
    query = {
        "query": {
            "more_like_this": {
                "fields": ["content"],
                "like": [
                    {
                        "_index": "documents",
                        "_id": doc_id
                    }
                ],
                "min_term_freq": 1,
                "max_query_terms": 12
            }
        },
        "size": size
    }
    
    response = index.search(
        index_name="documents",
        body=query
    )
    
    return response["hits"]["hits"]

# Example usage
similar_docs = find_similar_documents(index, "doc_001")
print("Similar documents:")
for doc in similar_docs:
    print(f"  {doc['_id']}: Score {doc['_score']:.2f}")
```

## Index Management

### Index Operations

```python
class IndexManager:
    """Manage Elasticsearch indices."""
    
    def __init__(self, index):
        self.index = index
    
    def create_index(self, index_name, settings=None, mappings=None):
        """Create a new index."""
        body = {}
        if settings:
            body["settings"] = settings
        if mappings:
            body["mappings"] = mappings
        
        return self.index.create_index(index_name, body)
    
    def delete_index(self, index_name):
        """Delete an index."""
        return self.index.delete_index(index_name)
    
    def reindex(self, source_index, target_index):
        """Reindex documents from source to target."""
        body = {
            "source": {"index": source_index},
            "dest": {"index": target_index}
        }
        
        return self.index.reindex(body)
    
    def get_index_stats(self, index_name):
        """Get index statistics."""
        return self.index.get_index_stats(index_name)
    
    def optimize_index(self, index_name):
        """Optimize index for search performance."""
        # Force merge to reduce segments
        return self.index.force_merge(index_name, max_num_segments=1)

# Example usage
manager = IndexManager(index)

# Create new index
manager.create_index(
    "documents_v2",
    settings={"number_of_shards": 2},
    mappings={"properties": {"content": {"type": "text"}}}
)

# Reindex data
manager.reindex("documents", "documents_v2")

# Get stats
stats = manager.get_index_stats("documents")
print(f"Index size: {stats['indices']['documents']['total']['store']['size_in_bytes']} bytes")
```

### Index Aliases

```python
def manage_aliases(index):
    """Manage index aliases for zero-downtime reindexing."""
    
    # Create alias
    index.add_alias("documents_v1", "documents_current")
    
    # Switch alias atomically
    actions = [
        {"remove": {"index": "documents_v1", "alias": "documents_current"}},
        {"add": {"index": "documents_v2", "alias": "documents_current"}}
    ]
    
    index.update_aliases({"actions": actions})
    
    # Now "documents_current" points to v2
    print("Alias switched to new index")

# Example usage
manage_aliases(index)
```

## Performance Optimization

### Batch Processing with Scroll API

```python
def process_all_documents(index, batch_size=100):
    """Process all documents using scroll API."""
    # Initial search
    response = index.search(
        index_name="documents",
        body={
            "size": batch_size,
            "query": {"match_all": {}}
        },
        scroll="2m"  # Keep scroll context for 2 minutes
    )
    
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    
    all_documents = []
    
    while hits:
        # Process current batch
        for hit in hits:
            all_documents.append(hit["_source"])
        
        # Get next batch
        response = index.scroll(
            scroll_id=scroll_id,
            scroll="2m"
        )
        
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
    
    # Clear scroll context
    index.clear_scroll(scroll_id=scroll_id)
    
    return all_documents

# Example usage
all_docs = process_all_documents(index)
print(f"Processed {len(all_docs)} documents")
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib
import json

class CachedElasticsearchClient:
    """Elasticsearch client with caching."""
    
    def __init__(self, index):
        self.index = index
    
    @lru_cache(maxsize=128)
    def cached_search(self, query_hash):
        """Cached search using query hash."""
        # This is called with the hash, need to store queries separately
        pass
    
    def search_with_cache(self, query):
        """Search with caching support."""
        # Create hash of query
        query_str = json.dumps(query, sort_keys=True)
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        
        # Check cache (simplified - in production use proper cache)
        if hasattr(self, '_cache') and query_hash in self._cache:
            return self._cache[query_hash]
        
        # Perform search
        result = self.index.search(
            index_name="documents",
            body=query
        )
        
        # Store in cache
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[query_hash] = result
        
        return result

# Example usage
cached_client = CachedElasticsearchClient(index)

# First search - hits Elasticsearch
result1 = cached_client.search_with_cache({"query": {"match_all": {}}})

# Second identical search - returns from cache
result2 = cached_client.search_with_cache({"query": {"match_all": {}}})
```

## Error Handling

### Robust Indexing with Retry

```python
import time
from typing import Dict, Any

def index_with_retry(index, doc_data: Dict[str, Any], max_retries=3):
    """Index document with retry logic."""
    for attempt in range(max_retries):
        try:
            response = index.index_document(
                index_name="documents",
                doc_type="_doc",
                body=doc_data
            )
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff
            wait_time = 2 ** attempt
            print(f"Indexing failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    return None

# Example usage
doc_data = {
    "content": "Important document",
    "metadata": {"priority": "high"}
}

try:
    result = index_with_retry(index, doc_data)
    print("Document indexed successfully")
except Exception as e:
    print(f"Failed to index document: {e}")
```

## Best Practices

1. **Use appropriate mappings**: Define mappings before indexing for optimal performance
2. **Batch operations**: Use bulk API for indexing multiple documents
3. **Implement pagination**: Use scroll API or search_after for large result sets
4. **Handle errors gracefully**: Implement retry logic and error handling
5. **Monitor performance**: Track indexing rate and search latency
6. **Use aliases**: Implement aliases for zero-downtime reindexing
7. **Optimize queries**: Use filters instead of queries when possible
8. **Cache frequently used queries**: Implement caching for repeated searches

## Related Examples

- [Document Processing](document-processing.md)
- [Text Normalization](text-normalization.md)
- [Basic Tree Operations](basic-tree.md)