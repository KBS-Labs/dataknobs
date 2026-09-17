# Elasticsearch Integration Examples

This guide demonstrates how to integrate Dataknobs with Elasticsearch for indexing and searching documents.

## Two Doors

`dataknobs_utils.elasticsearch_utils` offers two index classes, and picking the
wrong one accounts for most of the confusion:

| | `SimplifiedElasticsearchIndex` | `ElasticsearchIndex` |
|---|---|---|
| Scope | one index | several, described by `TableSettings` |
| Addressing | host and port directly | through a `RequestHelper` |
| Documents | `index`, `get`, `update`, `delete`, `count`, `exists` | none -- searching only |
| Searching | `search(body)` | `search(query, table=None)` |
| Also | `create`, `refresh`, `delete_by_query` | `analyze`, `sql`, `purge`, `inspect_indices`, `get_cluster_health` |

Most examples below use `SimplifiedElasticsearchIndex`, because writing
documents is what most integrations start with. Neither class wraps the whole
Elasticsearch API: there is no alias management, no reindex, no scroll and no
force-merge here. For those, talk to the cluster directly.

## Basic Setup

### A Single Index

```python
from dataknobs_utils.elasticsearch_utils import SimplifiedElasticsearchIndex

# Settings and mappings are given to the index, not to a separate object.
index = SimplifiedElasticsearchIndex(
    "documents",
    host="localhost",
    port=9200,
    settings={
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    mappings={
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "author": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "tags": {"type": "keyword"},
            "score": {"type": "float"},
        }
    },
)

# create() applies those settings and mappings; it answers False if the index
# already exists rather than raising.
index.create()
print(index.exists())
```

### Several Indices

`ElasticsearchIndex` manages a set of indices described by `TableSettings`,
and reaches the server through a `RequestHelper`. Note the parameter names --
the helper takes a server address, and the settings take three positional
arguments in this order:

```python
from dataknobs_utils.elasticsearch_utils import ElasticsearchIndex, TableSettings
from dataknobs_utils.requests_utils import RequestHelper

request_helper = RequestHelper("localhost", 9200)

table_settings = TableSettings(
    "documents",                       # table_name
    {                                  # data_settings
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    {                                  # data_mapping
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "author": {"type": "keyword"},
        }
    },
)

indices = ElasticsearchIndex(request_helper, [table_settings])
print(indices.is_up())
```

## Document Indexing

### Indexing Single Documents

```python
from dataknobs_structures import Text, TextMetaData

def index_document(index, doc, doc_id=None):
    """Index a single document."""
    doc_data = {
        "content": doc.text,
        "metadata": doc.metadata.data,
        "timestamp": doc.metadata.get_value("timestamp", "2024-01-01T00:00:00Z"),
    }

    # The index knows its own name, so `index()` takes the body and, at most,
    # a document id. Retries on transient 5xx errors are built in.
    return index.index(doc_data, doc_id=doc_id, refresh=True)

# Example usage. TextMetaData takes text_id positionally; anything else is
# free-form and lands in .data.
doc = Text(
    "This is a sample document about Elasticsearch integration.",
    TextMetaData(
        "doc_001",
        title="Elasticsearch Guide",
        author="John Doe",
        tags=["elasticsearch", "search", "indexing"],
    ),
)

response = index_document(index, doc, doc_id="doc_001")
print(f"Indexed document: {response['_id']}")
```

Pass `op_type="create"` for an insert that must not overwrite: an existing id
then raises `ElasticsearchConflictError` rather than replacing the document.

```python
from dataknobs_utils.elasticsearch_utils import ElasticsearchConflictError

try:
    index.index(doc_data, doc_id="doc_001", op_type="create")
except ElasticsearchConflictError:
    print("doc_001 already exists")
```

### Bulk Indexing

Bulk loading goes through a file rather than a method: `add_batch_data` writes
the alternating action/source NDJSON that Elasticsearch's bulk API expects,
and you hand that file to the cluster.

```python
from pathlib import Path
import tempfile

from dataknobs_structures import Text, TextMetaData
from dataknobs_utils.elasticsearch_utils import add_batch_data

documents = [
    Text(f"Document {i} content", TextMetaData(str(i)))
    for i in range(100)
]

def source_records(docs):
    for doc in docs:
        yield {"content": doc.text, "metadata": doc.metadata.data}

batchfile_path = Path(tempfile.mkdtemp()) / "documents.ndjson"

with batchfile_path.open("w") as batchfile:
    # Answers the next free id, so successive calls can append without
    # colliding. `source_id_fieldname` writes that id back into each record.
    next_id = add_batch_data(
        batchfile,
        source_records(documents),
        "documents",
        source_id_fieldname="id",
        cur_id=1,
    )

print(next_id)                                      # 101
print(sum(1 for _ in batchfile_path.open()))        # 200 -- two lines per doc
```

Read the file back with `batchfile_record_generator` (source lines only) or
`collect_batchfile_records` (as a DataFrame):

```python
from dataknobs_utils.elasticsearch_utils import collect_batchfile_records

df = collect_batchfile_records(str(batchfile_path))
print(len(df), list(df.columns))     # 100 ['content', 'metadata', 'id']
```

## Document Searching

### Basic Search Queries

`search` takes the query body alone and answers a `ServerResponse`, not a raw
dict: read `.result` for the payload and check `.succeeded` rather than
assuming the call reached the server.

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

    response = index.search(query)
    if not response.succeeded:
        return []

    hits = response.result.get("hits", {}).get("hits", [])
    return [
        {
            "id": hit["_id"],
            "score": hit["_score"],
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"],
        }
        for hit in hits
    ]

# Example usage
results = search_documents(index, "elasticsearch integration")
for result in results:
    print(f"Score: {result['score']:.2f} - {result['metadata'].get('title', 'Untitled')}")
```

For the two common shapes there are builders, so the DSL does not have to be
written out by hand:

```python
from dataknobs_utils.elasticsearch_utils import (
    build_field_query_dict,
    build_phrase_query_dict,
)

# One field, or several -- match vs multi_match is chosen for you
query = build_field_query_dict("content", "elasticsearch integration", operator="AND")
query = build_field_query_dict(["content", "title"], "elasticsearch integration")

# An exact phrase, with slop tolerance
query = build_phrase_query_dict("content", "sample document", slop=1)

response = index.search(query)
```

### Advanced Search with Filters

```python
def advanced_search(index, query_text, filters=None, aggregations=None):
    """Advanced search with filters and aggregations."""
    bool_query = {
        "must": [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": ["content", "metadata.title"]
                }
            }
        ]
    }

    if filters:
        bool_query["filter"] = filters

    query = {"query": {"bool": bool_query}}

    if aggregations:
        query["aggs"] = aggregations

    return index.search(query)

# Example usage
response = advanced_search(
    index,
    "search",
    filters=[
        {"term": {"metadata.author": "John Doe"}},
        {"range": {"timestamp": {"gte": "2024-01-01"}}},
    ],
    aggregations={"tags": {"terms": {"field": "metadata.tags", "size": 10}}},
)
```

### Results as DataFrames

`build_hits_dataframe` turns the `_source` of each hit into a row. It takes
the raw result payload, so unwrap the `ServerResponse` first:

```python
from dataknobs_utils.elasticsearch_utils import build_hits_dataframe

response = index.search({"query": {"match_all": {}}})
hits_df = build_hits_dataframe(response.result)
print(hits_df.head())
```

`decode_results` does the same for a whole result, answering a dict that may
hold `hits_df`. Its aggregation half (`build_aggs_dataframe`) is a documented
placeholder that returns `None` -- read aggregations off `response.result`
directly.

## Text Processing Pipeline

### Index Normalized Documents

```python
from dataknobs_structures import Text, TextMetaData
from dataknobs_xization import normalize

class ElasticsearchPipeline:
    """Pipeline for processing and indexing documents."""

    def __init__(self, index):
        self.index = index

    def process_and_index(self, doc, doc_id=None):
        """Process document and index to Elasticsearch."""
        # basic_normalization_fn lowercases, expands camelCase and simplifies
        # quotes; it does not strip or collapse whitespace.
        normalized_text = normalize.basic_normalization_fn(doc.text)

        doc_data = {
            "original_text": doc.text,
            "normalized_text": normalized_text,
            "metadata": doc.metadata.data,
            "word_count": len(normalized_text.split()),
            "char_count": len(normalized_text),
        }

        return self.index.index(doc_data, doc_id=doc_id)

    def search_normalized(self, query_text):
        """Search using normalized query."""
        normalized_query = normalize.basic_normalization_fn(query_text)
        return self.index.search(
            {"query": {"match": {"normalized_text": normalized_query}}}
        )

# One index per pipeline, since the index carries its own name.
processed = SimplifiedElasticsearchIndex("processed_documents")
processed.create()
pipeline = ElasticsearchPipeline(processed)

doc = Text(
    "getUserData&ProcessInput",
    TextMetaData("doc_001", text_label="code"),
)

pipeline.process_and_index(doc, "doc_001")
processed.refresh()          # make it searchable immediately
results = pipeline.search_normalized("get user data")
```

## Document Analysis

### Term Frequency Analysis

```python
def analyze_term_frequency(index, field="content", size=20):
    """Analyze term frequency across all documents."""
    response = index.search({
        "size": 0,
        "aggs": {
            "term_frequency": {
                "terms": {
                    "field": f"{field}.keyword",
                    "size": size
                }
            }
        }
    })

    buckets = response.result["aggregations"]["term_frequency"]["buckets"]
    return [(b["key"], b["doc_count"]) for b in buckets]

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
    # get() takes the id alone and answers the document, or None if absent.
    source_doc = index.get(doc_id)
    if source_doc is None:
        return []

    response = index.search({
        "query": {
            "more_like_this": {
                "fields": ["content"],
                "like": [{"_index": "documents", "_id": doc_id}],
                "min_term_freq": 1,
                "max_query_terms": 12
            }
        },
        "size": size
    })

    return response.result["hits"]["hits"]

# Example usage
similar_docs = find_similar_documents(index, "doc_001")
print("Similar documents:")
for doc in similar_docs:
    print(f"  {doc['_id']}: Score {doc['_score']:.2f}")
```

## Index Management

What is covered is the lifecycle of one index -- create it, count it, empty
it, drop it:

```python
index = SimplifiedElasticsearchIndex(
    "documents_v2",
    settings={"number_of_shards": 2},
    mappings={"properties": {"content": {"type": "text"}}},
)

index.create()                  # False if it already exists
print(index.exists())           # True
print(index.count())            # 0

# Remove matching documents, keeping the index
index.delete_by_query({"query": {"term": {"metadata.author": "John Doe"}}})

# Remove one document, or the whole index
index.delete(doc_id="doc_001")
index.delete()
```

Across several indices, `ElasticsearchIndex` reports and empties:

```python
indices = ElasticsearchIndex(request_helper, [table_settings])

print(indices.is_up())
print(indices.inspect_indices())
print(indices.get_cluster_health())

indices.purge("documents")          # empty it
indices.delete_table("documents")   # drop it
```

### What Is Not Here

Aliases, reindex, scroll, force-merge and index statistics have no wrapper in
this package. They are ordinary Elasticsearch REST calls; issue them against
the cluster directly, with `RequestHelper` if you want the same connection
handling:

```python
from dataknobs_utils.requests_utils import RequestHelper

helper = RequestHelper("localhost", 9200)
response = helper.post("_aliases", payload={
    "actions": [
        {"remove": {"index": "documents_v1", "alias": "documents_current"}},
        {"add": {"index": "documents_v2", "alias": "documents_current"}},
    ]
})
print(response.succeeded)
```

## Performance

### Paging Through Everything

There is no scroll wrapper. For a bounded corpus, page with `from`/`size`;
beyond Elasticsearch's `max_result_window` (10,000 by default) use
`search_after` against the cluster directly.

```python
def process_all_documents(index, batch_size=100):
    """Process all documents, a page at a time."""
    all_documents = []
    offset = 0

    while True:
        response = index.search({
            "from": offset,
            "size": batch_size,
            "query": {"match_all": {}},
            "sort": [{"_doc": "asc"}],
        })
        hits = response.result["hits"]["hits"]
        if not hits:
            break

        all_documents.extend(hit["_source"] for hit in hits)
        offset += len(hits)

    return all_documents

# Example usage
all_docs = process_all_documents(index)
print(f"Processed {len(all_docs)} documents")
```

### Caching Strategies

```python
import hashlib
import json

class CachedElasticsearchClient:
    """Elasticsearch client with a query cache."""

    def __init__(self, index):
        self.index = index
        self._cache = {}

    def search_with_cache(self, query):
        """Search, reusing the result of an identical earlier query."""
        query_hash = hashlib.sha256(
            json.dumps(query, sort_keys=True).encode()
        ).hexdigest()

        if query_hash not in self._cache:
            self._cache[query_hash] = self.index.search(query)

        return self._cache[query_hash]

# Example usage
cached_client = CachedElasticsearchClient(index)

result1 = cached_client.search_with_cache({"query": {"match_all": {}}})  # server
result2 = cached_client.search_with_cache({"query": {"match_all": {}}})  # cache
```

## Error Handling

### Retry Is Already There

`index()` retries transient 5xx errors itself, with exponential backoff --
`max_retries` and `initial_delay` are its parameters, so a retry loop around
it duplicates what it does:

```python
response = index.index(
    doc_data,
    max_retries=5,
    initial_delay=1.0,
)
```

What it deliberately does not retry is a 409 conflict from
`op_type="create"`, which is definitive rather than transient:

```python
from dataknobs_utils.elasticsearch_utils import ElasticsearchConflictError

doc_data = {
    "content": "Important document",
    "metadata": {"priority": "high"},
}

try:
    result = index.index(doc_data, doc_id="doc_001", op_type="create")
    print("Document indexed successfully")
except ElasticsearchConflictError:
    print("doc_001 already exists -- update it instead")
    index.update("doc_001", {"doc": doc_data})
```

Everything else is reported rather than raised: `search` answers a
`ServerResponse` whose `.succeeded` is False, and `create` / `refresh` /
`update` / `delete` answer bools.

## Best Practices

1. **Use appropriate mappings**: Define mappings before indexing -- pass them
   to the index and call `create()` before the first document
2. **Batch operations**: Build an NDJSON batch file with `add_batch_data`
   rather than indexing one document at a time
3. **Refresh deliberately**: `index(..., refresh=True)` and `refresh()` cost
   throughput; use them in tests and after a batch, not per document
4. **Check `.succeeded`**: a failed search answers an empty hit list rather
   than raising
5. **Let `index()` do the retrying**: it already backs off on 5xx
6. **Use `op_type="create"`** when an insert must not overwrite
7. **Implement pagination**: page with `from`/`size`, and move to
   `search_after` past `max_result_window`
8. **Optimize queries**: use filters instead of queries when scoring does not
   matter

## Related Examples

- [Text Normalization](text-normalization.md)
- [Basic Tree Operations](basic-tree.md)
