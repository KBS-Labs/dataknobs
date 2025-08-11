# Document Processing Examples

This guide demonstrates various document processing workflows using Dataknobs.

## Basic Document Creation

### Creating a Simple Document

```python
from dataknobs_structures import Document

# Create a document with text content
doc = Document(
    text="This is a sample document containing important information.",
    metadata={
        "title": "Sample Document",
        "author": "John Doe",
        "created": "2024-01-01"
    }
)

print(f"Document ID: {doc.doc_id}")
print(f"Text: {doc.text}")
print(f"Metadata: {doc.metadata}")
```

### Document with Sections

```python
from dataknobs_structures import Document

# Create a structured document
sections = [
    {"title": "Introduction", "content": "This is the introduction."},
    {"title": "Main Body", "content": "This is the main content."},
    {"title": "Conclusion", "content": "This is the conclusion."}
]

doc = Document(
    text="\n\n".join([s["content"] for s in sections]),
    metadata={"sections": sections}
)
```

## File-based Document Processing

### Processing Text Files

```python
from dataknobs_utils import file_utils
from dataknobs_structures import Document

def process_text_file(filepath):
    """Process a text file into a Document."""
    # Read file content
    content = file_utils.read_file(filepath)
    
    # Create document
    doc = Document(
        text=content,
        metadata={
            "source": filepath,
            "size": len(content),
            "lines": content.count('\n') + 1
        }
    )
    
    return doc

# Process a file
doc = process_text_file("data/sample.txt")
```

### Batch Document Processing

```python
from dataknobs_utils import file_utils
from dataknobs_structures import Document
import os

def process_directory(directory_path):
    """Process all text files in a directory."""
    documents = []
    
    for filepath in file_utils.filepath_generator(directory_path):
        if filepath.endswith('.txt'):
            content = file_utils.read_file(filepath)
            doc = Document(
                text=content,
                metadata={
                    "source": filepath,
                    "filename": os.path.basename(filepath)
                }
            )
            documents.append(doc)
    
    return documents

# Process all documents
docs = process_directory("data/documents/")
print(f"Processed {len(docs)} documents")
```

## Document Transformation

### Text Normalization

```python
from dataknobs_structures import Document
from dataknobs_xization import normalize

def normalize_document(doc):
    """Normalize document text."""
    # Apply various normalizations
    normalized_text = doc.text
    
    # Expand camelCase
    normalized_text = normalize.expand_camelcase_fn(normalized_text)
    
    # Expand ampersands
    normalized_text = normalize.expand_ampersand_fn(normalized_text)
    
    # Remove extra whitespace
    normalized_text = normalize.normalize_whitespace_fn(normalized_text)
    
    # Create new document with normalized text
    return Document(
        text=normalized_text,
        metadata={**doc.metadata, "normalized": True}
    )

# Example usage
original = Document("getUserName&validateInput", metadata={"type": "code"})
normalized = normalize_document(original)
print(f"Original: {original.text}")
print(f"Normalized: {normalized.text}")
```

### Document Chunking

```python
from dataknobs_structures import Document

def chunk_document(doc, chunk_size=1000, overlap=100):
    """Split document into overlapping chunks."""
    text = doc.text
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        chunk_doc = Document(
            text=chunk_text,
            metadata={
                **doc.metadata,
                "chunk_index": len(chunks),
                "chunk_start": i,
                "chunk_end": min(i + chunk_size, len(text))
            }
        )
        chunks.append(chunk_doc)
    
    return chunks

# Split a large document
large_doc = Document("A" * 5000, metadata={"title": "Large Document"})
chunks = chunk_document(large_doc)
print(f"Created {len(chunks)} chunks")
```

## Document Analysis

### Extracting Statistics

```python
from dataknobs_structures import Document
import re

def analyze_document(doc):
    """Extract statistics from a document."""
    text = doc.text
    
    stats = {
        "character_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count('\n') + 1,
        "sentence_count": len(re.split(r'[.!?]+', text)),
        "paragraph_count": len(re.split(r'\n\n+', text)),
        "unique_words": len(set(text.lower().split())),
        "average_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1)
    }
    
    return stats

# Analyze a document
doc = Document("""
This is a sample document. It has multiple sentences!
And even multiple paragraphs?

This is the second paragraph.
""")

stats = analyze_document(doc)
for key, value in stats.items():
    print(f"{key}: {value}")
```

### Keyword Extraction

```python
from dataknobs_structures import Document
from collections import Counter
import re

def extract_keywords(doc, num_keywords=10):
    """Extract top keywords from a document."""
    # Simple keyword extraction based on frequency
    text = doc.text.lower()
    
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-z]+\b', text)
    
    # Filter out common stop words (simplified)
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Count frequencies
    word_counts = Counter(words)
    
    # Return top keywords
    return word_counts.most_common(num_keywords)

# Extract keywords
doc = Document("""
Machine learning is a subset of artificial intelligence. 
Machine learning algorithms build a model based on training data.
The model can make predictions or decisions without being explicitly programmed.
""")

keywords = extract_keywords(doc, 5)
print("Top keywords:")
for word, count in keywords:
    print(f"  {word}: {count}")
```

## Document Storage and Retrieval

### Using RecordStore

```python
from dataknobs_structures import Document, RecordStore

# Create a record store for documents
doc_store = RecordStore()

# Store documents
doc1 = Document("First document", metadata={"id": "doc1"})
doc2 = Document("Second document", metadata={"id": "doc2"})

doc_store.add("doc1", doc1)
doc_store.add("doc2", doc2)

# Retrieve documents
retrieved = doc_store.get("doc1")
print(f"Retrieved: {retrieved.text}")

# List all documents
all_docs = doc_store.get_all()
print(f"Total documents: {len(all_docs)}")
```

### Document Serialization

```python
from dataknobs_structures import Document
from dataknobs_utils import json_utils
import json

def serialize_document(doc):
    """Serialize a document to JSON."""
    doc_dict = {
        "text": doc.text,
        "metadata": doc.metadata,
        "doc_id": doc.doc_id
    }
    return json.dumps(doc_dict, indent=2)

def deserialize_document(json_str):
    """Deserialize a document from JSON."""
    doc_dict = json.loads(json_str)
    return Document(
        text=doc_dict["text"],
        metadata=doc_dict.get("metadata", {})
    )

# Example usage
original = Document("Sample text", metadata={"author": "Jane"})
serialized = serialize_document(original)
print(f"Serialized:\n{serialized}")

deserialized = deserialize_document(serialized)
print(f"\nDeserialized text: {deserialized.text}")
```

## Advanced Document Processing

### Document Pipeline

```python
from dataknobs_structures import Document
from dataknobs_xization import normalize
from dataknobs_utils import file_utils

class DocumentPipeline:
    """A configurable document processing pipeline."""
    
    def __init__(self):
        self.processors = []
    
    def add_processor(self, processor_func):
        """Add a processing step."""
        self.processors.append(processor_func)
        return self
    
    def process(self, doc):
        """Process document through all steps."""
        result = doc
        for processor in self.processors:
            result = processor(result)
        return result

# Define processing functions
def normalize_text(doc):
    """Normalize document text."""
    return Document(
        normalize.basic_normalization_fn(doc.text),
        metadata={**doc.metadata, "normalized": True}
    )

def add_statistics(doc):
    """Add text statistics to metadata."""
    doc.metadata["stats"] = {
        "words": len(doc.text.split()),
        "chars": len(doc.text)
    }
    return doc

def add_hash(doc):
    """Add content hash to metadata."""
    import hashlib
    doc.metadata["hash"] = hashlib.md5(doc.text.encode()).hexdigest()
    return doc

# Create and use pipeline
pipeline = DocumentPipeline()
pipeline.add_processor(normalize_text)
pipeline.add_processor(add_statistics)
pipeline.add_processor(add_hash)

# Process a document
input_doc = Document("ProcessThisText&CalculateStats")
output_doc = pipeline.process(input_doc)

print(f"Original: {input_doc.text}")
print(f"Processed: {output_doc.text}")
print(f"Metadata: {output_doc.metadata}")
```

### Document Comparison

```python
from dataknobs_structures import Document
import difflib

def compare_documents(doc1, doc2):
    """Compare two documents and find differences."""
    # Simple text similarity
    similarity = difflib.SequenceMatcher(
        None, doc1.text, doc2.text
    ).ratio()
    
    # Find differences
    differ = difflib.unified_diff(
        doc1.text.splitlines(),
        doc2.text.splitlines(),
        lineterm='',
        fromfile='doc1',
        tofile='doc2'
    )
    
    return {
        "similarity": similarity,
        "differences": list(differ),
        "doc1_unique_words": set(doc1.text.split()) - set(doc2.text.split()),
        "doc2_unique_words": set(doc2.text.split()) - set(doc1.text.split())
    }

# Compare documents
doc1 = Document("The quick brown fox jumps over the lazy dog")
doc2 = Document("The quick brown fox leaps over a lazy cat")

comparison = compare_documents(doc1, doc2)
print(f"Similarity: {comparison['similarity']:.2%}")
print(f"Doc1 unique words: {comparison['doc1_unique_words']}")
print(f"Doc2 unique words: {comparison['doc2_unique_words']}")
```

## Integration Examples

### Document to Tree Structure

```python
from dataknobs_structures import Document, Tree

def document_to_tree(doc):
    """Convert a document to a tree structure."""
    # Create root node from document
    root = Tree(doc)
    
    # If document has sections, create child nodes
    if "sections" in doc.metadata:
        for section in doc.metadata["sections"]:
            section_doc = Document(
                section["content"],
                metadata={"title": section["title"]}
            )
            root.add_child(section_doc)
    
    return root

# Create structured document
doc = Document(
    "Main document content",
    metadata={
        "title": "Main",
        "sections": [
            {"title": "Intro", "content": "Introduction text"},
            {"title": "Body", "content": "Body text"},
            {"title": "Conclusion", "content": "Conclusion text"}
        ]
    }
)

# Convert to tree
tree = document_to_tree(doc)
print(f"Root: {tree.data.metadata['title']}")
for child in tree.children:
    print(f"  Child: {child.data.metadata['title']}")
```

## Best Practices

1. **Always include metadata**: Track source, creation time, and processing steps
2. **Handle encoding properly**: Use UTF-8 for text files
3. **Validate input**: Check document content before processing
4. **Use appropriate chunk sizes**: Balance between processing efficiency and accuracy
5. **Implement error handling**: Gracefully handle malformed documents

## Related Examples

- [Text Normalization](text-normalization.md)
- [Tree Operations](basic-tree.md)
- [Elasticsearch Integration](elasticsearch-integration.md)