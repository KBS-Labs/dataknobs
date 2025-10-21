# Markdown Chunking Examples

This page demonstrates practical examples of using the markdown chunking utilities for RAG applications.

## Basic Chunking

The simplest use case - parse and chunk a markdown document:

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree

markdown = """
# User Guide

## Installation

Install the package using pip or uv.

## Quick Start

Here's a simple example to get started.
"""

# Parse into tree structure
tree = parse_markdown(markdown)

# Generate chunks
chunks = chunk_markdown_tree(tree, max_chunk_size=500)

# Access chunks
for chunk in chunks:
    print(f"Chunk {chunk.metadata.chunk_index}:")
    print(f"  Path: {chunk.metadata.get_heading_path()}")
    print(f"  Text: {chunk.text[:50]}...")
```

## Handling Code Blocks

Code blocks are preserved as atomic units:

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree

markdown = """
# API Reference

## Authentication

Use the following code to authenticate:

```python
import requests

def authenticate(api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post("https://api.example.com/auth", headers=headers)
    return response.json()
```

The authentication returns a token.
"""

tree = parse_markdown(markdown)
chunks = chunk_markdown_tree(tree, max_chunk_size=200)  # Small size

# Code block won't be split even though it exceeds 200 chars
for chunk in chunks:
    construct_type = chunk.metadata.custom.get("node_type")
    if construct_type == "code":
        print(f"Code block found (language: {chunk.metadata.custom.get('language')})")
        print(f"Size: {len(chunk.text)} chars")  # May exceed max_chunk_size
        print(f"Preserved intact!")
```

## Vector Store Integration

Loading chunks into a vector database:

```python
from dataknobs_xization import stream_markdown_file, HeadingInclusion
import chromadb

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection(
    name="documentation",
    metadata={"description": "Product documentation chunks"}
)

# Stream and load chunks
for chunk in stream_markdown_file(
    "documentation.md",
    max_chunk_size=500,
    heading_inclusion=HeadingInclusion.IN_METADATA  # Clean text, context in metadata
):
    collection.add(
        documents=[chunk.text],
        metadatas=[{
            **chunk.metadata.to_dict(),
            "heading_path": chunk.metadata.get_heading_path()
        }],
        ids=[f"doc_chunk_{chunk.metadata.chunk_index}"]
    )

print(f"Loaded {collection.count()} chunks into vector store")
```

## RAG Query with Context

Using heading metadata for better retrieval:

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree, HeadingInclusion

# Parse documentation
with open("api_docs.md") as f:
    tree = parse_markdown(f.read())

# Chunk with metadata
chunks = chunk_markdown_tree(
    tree,
    heading_inclusion=HeadingInclusion.BOTH,
    max_chunk_size=500
)

# Simulate retrieval (in practice, use vector similarity)
def find_relevant_chunks(query, chunks, top_k=3):
    """Find chunks relevant to query (simplified)."""
    scored_chunks = []
    for chunk in chunks:
        # Score based on keyword match (use embeddings in production)
        score = sum(1 for word in query.lower().split() if word in chunk.text.lower())
        scored_chunks.append((score, chunk))

    # Return top-k by score
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k]]

# Query
query = "How do I authenticate with the API?"
relevant = find_relevant_chunks(query, chunks)

# Build context for LLM
context_parts = []
for chunk in relevant:
    # Include heading context
    heading_path = chunk.metadata.get_heading_path(" → ")
    context_parts.append(f"## {heading_path}\n\n{chunk.text}")

context = "\n\n".join(context_parts)

# Send to LLM
prompt = f"""Based on the following documentation:

{context}

Question: {query}

Answer:"""

# Use with your LLM...
```

## Processing Different Construct Types

Handle code, tables, and lists specially:

```python
from dataknobs_xization import parse_markdown, chunk_markdown_tree

markdown = """
# Database Guide

## Schema

The database uses the following schema:

| Table | Columns | Purpose |
|-------|---------|---------|
| users | id, name, email | User accounts |
| posts | id, user_id, content | User posts |

## Query Examples

Use these SQL queries:

```sql
SELECT * FROM users WHERE email = ?;
```

## Best Practices

- Always use parameterized queries
- Index foreign keys
- Regular backups
"""

tree = parse_markdown(markdown)
chunks = chunk_markdown_tree(tree)

# Process chunks by type
for chunk in chunks:
    node_type = chunk.metadata.custom.get("node_type")

    if node_type == "table":
        print(f"Table chunk - {chunk.metadata.custom.get('rows')} rows")
        # Store table with special indexing

    elif node_type == "code":
        lang = chunk.metadata.custom.get("language", "text")
        print(f"Code chunk - language: {lang}")
        # Apply syntax highlighting, store separately

    elif node_type == "list":
        list_type = chunk.metadata.custom.get("list_type")
        print(f"List chunk - {list_type}")
        # Extract list items for structured storage

    else:
        print(f"Text chunk - {chunk.metadata.chunk_size} chars")
        # Standard text processing
```

## Streaming Large Documents

Process large files without loading entirely into memory:

```python
from dataknobs_xization import AdaptiveStreamingProcessor

# For very large documents
processor = AdaptiveStreamingProcessor(
    max_chunk_size=1000,
    memory_limit_nodes=10000,  # Limit nodes in memory
    adaptive_threshold=0.8      # Start chunking at 80% capacity
)

# Process incrementally
for chunk in processor.process_file("large_documentation.md"):
    # Chunk yielded as soon as ready
    # Memory usage stays bounded
    process_and_store(chunk)
```

## Custom Chunk Sizes for Different Sections

Adjust chunking based on heading level:

```python
from dataknobs_xization import parse_markdown, MarkdownChunker, HeadingInclusion

tree = parse_markdown(document)

# Get all terminal nodes
terminal_nodes = tree.collect_terminal_nodes(
    accept_node_fn=lambda n: not n.data.is_heading() and n.data.node_type != "root"
)

# Custom chunking logic
all_chunks = []

for node in terminal_nodes:
    # Get heading path to determine context
    headings = []
    current = node.parent
    while current and current.data.node_type == "heading":
        headings.insert(0, current.data.text)
        current = current.parent

    # Adjust chunk size based on depth
    depth = len(headings)
    if depth == 1:  # Top-level sections
        chunk_size = 1500  # Larger chunks for overview
    elif depth == 2:  # Sub-sections
        chunk_size = 800   # Medium chunks
    else:  # Deep sections
        chunk_size = 400   # Smaller, focused chunks

    # Create chunker with appropriate size
    chunker = MarkdownChunker(
        max_chunk_size=chunk_size,
        heading_inclusion=HeadingInclusion.BOTH
    )

    # Process this node's subtree
    # (simplified - would need proper tree handling)
    # ...
```

## JSON Export for Analysis

Export chunks as JSON for analysis or storage:

```python
import json
from dataknobs_xization import parse_markdown, chunk_markdown_tree

tree = parse_markdown(document)
chunks = chunk_markdown_tree(tree)

# Convert to JSON
chunks_json = [chunk.to_dict() for chunk in chunks]

# Save to file
with open("chunks.json", "w") as f:
    json.dump(chunks_json, f, indent=2, ensure_ascii=False)

# Or analyze chunk distribution
import pandas as pd

df = pd.DataFrame(chunks_json)
df_meta = pd.json_normalize(df['metadata'])

print("Chunk statistics:")
print(f"Total chunks: {len(df)}")
print(f"Mean size: {df_meta['chunk_size'].mean():.1f} chars")
print(f"Max size: {df_meta['chunk_size'].max()} chars")
print("\nBy node type:")
print(df_meta['node_type'].value_counts())
```

## Command-Line Workflow

Use the CLI for quick analysis and testing:

```bash
# Analyze document structure
uv run python -m dataknobs_xization.markdown.md_cli info my_doc.md

# Generate chunks and inspect
uv run python -m dataknobs_xization.markdown.md_cli chunk my_doc.md \
  --max-size 500 \
  --show-metadata \
  | head -100

# Export for loading into vector store
uv run python -m dataknobs_xization.markdown.md_cli chunk my_doc.md \
  --output-format json \
  --headings metadata \
  --output chunks.json

# Then load in Python
import json
with open("chunks.json") as f:
    chunks = json.load(f)
    # Load into your vector store...
```

## Complete RAG Pipeline Example

End-to-end example combining chunking with retrieval:

```python
from dataknobs_xization import stream_markdown_file, HeadingInclusion
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None

    def load_documents(self, file_path):
        """Load and chunk markdown documents."""
        print(f"Loading {file_path}...")

        for chunk in stream_markdown_file(
            file_path,
            max_chunk_size=500,
            heading_inclusion=HeadingInclusion.IN_METADATA
        ):
            self.chunks.append(chunk)

        # Generate embeddings
        texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(texts)

        print(f"Loaded {len(self.chunks)} chunks")

    def query(self, question, top_k=3):
        """Query the knowledge base."""
        # Embed question
        q_embedding = self.model.encode([question])

        # Find similar chunks
        similarities = cosine_similarity(q_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return results with context
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "context": chunk.metadata.get_heading_path(),
                "similarity": similarities[idx],
                "metadata": chunk.metadata.to_dict()
            })

        return results

# Use it
rag = SimpleRAG()
rag.load_documents("documentation.md")

results = rag.query("How do I authenticate?")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Context: {result['context']}")
    print(f"   Similarity: {result['similarity']:.3f}")
    print(f"   Text: {result['text'][:100]}...")
```

## See Also

- [API Reference](../packages/xization/markdown-chunking.md)
- [Example Files](https://github.com/kbs-labs/dataknobs/tree/main/packages/xization/examples/markdown)
- [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/xization/src/dataknobs_xization/markdown)
