# Document API Documentation

The document module provides classes for working with text documents and their metadata.

## Overview

The document classes provide:

- **Text**: A wrapper for text strings with metadata
- **TextMetaData**: Container for document metadata
- **MetaData**: Base metadata container

## Classes

### MetaData

Base class for managing metadata.

```python
from dataknobs_structures.document import MetaData
```

#### Constructor

```python
MetaData(key_data: dict, **kwargs)
```

**Parameters:**
- `key_data` (dict): Mandatory metadata dictionary
- `**kwargs`: Additional optional metadata

**Example:**
```python
metadata = MetaData(
    {"author": "John Doe", "type": "article"},
    created="2024-01-01",
    tags=["python", "documentation"]
)
```

#### Properties

##### data
```python
@property
def data(self) -> dict
```
Returns the complete metadata dictionary.

**Example:**
```python
meta = MetaData({"title": "Document"}, version=1)
print(meta.data)  # {"title": "Document", "version": 1}
```

#### Methods

##### get_value()
```python
def get_value(self, attribute: str, missing=None) -> Any
```
Retrieves a metadata value with a fallback.

**Parameters:**
- `attribute` (str): The metadata key to retrieve
- `missing` (Any): Value to return if key not found

**Example:**
```python
meta = MetaData({"title": "My Document"})
print(meta.get_value("title"))        # "My Document"
print(meta.get_value("author", "Unknown"))  # "Unknown"
```

### TextMetaData

Specialized metadata container for text documents.

```python
from dataknobs_structures import TextMetaData
```

#### Constructor

```python
TextMetaData(text_id: Any, text_label: str = "text", **kwargs)
```

**Parameters:**
- `text_id` (Any): Unique identifier for the text
- `text_label` (str): Label for the text type (default: "text")
- `**kwargs`: Additional metadata

**Example:**
```python
metadata = TextMetaData(
    text_id=123,
    text_label="article",
    title="Introduction to Python",
    author="Jane Smith",
    created="2024-01-15"
)
```

#### Properties

##### text_id
```python
@property
def text_id(self) -> Any
```
Returns the text identifier.

##### text_label
```python
@property
def text_label(self) -> str | Any
```
Returns the text label.

**Example:**
```python
meta = TextMetaData(text_id="doc_001", text_label="tutorial")
print(meta.text_id)    # "doc_001"
print(meta.text_label) # "tutorial"
```

### Text

Main class for text documents with metadata.

```python
from dataknobs_structures import Text
```

#### Constructor

```python
Text(text: str, metadata: TextMetaData | None)
```

**Parameters:**
- `text` (str): The text content
- `metadata` (TextMetaData | None): Document metadata

**Example:**
```python
# With metadata
metadata = TextMetaData(text_id=1, text_label="article")
doc = Text("This is the document content.", metadata)

# Without metadata (creates default)
doc = Text("Simple document.")
```

#### Properties

##### text
```python
@property
def text(self) -> str
```
Returns the document text content.

##### text_id
```python
@property
def text_id(self) -> Any
```
Returns the text ID from metadata.

##### text_label
```python
@property
def text_label(self) -> str
```
Returns the text label from metadata.

##### metadata
```python
@property
def metadata(self) -> TextMetaData
```
Returns the document metadata.

**Example:**
```python
metadata = TextMetaData(
    text_id="article_001",
    text_label="news",
    headline="Breaking News",
    author="Reporter"
)

doc = Text("This is breaking news content...", metadata)

print(doc.text)      # "This is breaking news content..."
print(doc.text_id)   # "article_001"
print(doc.text_label) # "news"
print(doc.metadata.get_value("headline")) # "Breaking News"
```

## Constants

### Text Metadata Attributes

```python
from dataknobs_structures.document import (
    TEXT_ID_ATTR,    # "text_id"
    TEXT_LABEL_ATTR, # "text_label"  
    TEXT_LABEL       # "text"
)
```

These constants define standard metadata attribute names.

## Usage Examples

### Basic Document Creation

```python
from dataknobs_structures import Text, TextMetaData

# Create metadata
metadata = TextMetaData(
    text_id="doc_123",
    text_label="tutorial",
    title="Python Basics",
    difficulty="beginner"
)

# Create document
document = Text(
    "Python is a powerful programming language...",
    metadata
)

# Access properties
print(f"Document ID: {document.text_id}")
print(f"Type: {document.text_label}")
print(f"Title: {document.metadata.get_value('title')}")
print(f"Content length: {len(document.text)}")
```

### Working with Collections

```python
from dataknobs_structures import Text, TextMetaData

# Create a collection of documents
documents = []

for i in range(3):
    metadata = TextMetaData(
        text_id=f"doc_{i}",
        text_label="example",
        index=i
    )
    
    doc = Text(f"This is document number {i}", metadata)
    documents.append(doc)

# Process documents
for doc in documents:
    print(f"Document {doc.text_id}: {doc.text}")
    print(f"Index: {doc.metadata.get_value('index')}")
```

### Custom Metadata

```python
from dataknobs_structures import Text, TextMetaData
from datetime import datetime

# Rich metadata
metadata = TextMetaData(
    text_id="research_paper_001",
    text_label="research",
    title="Machine Learning Applications",
    authors=["Dr. Smith", "Dr. Johnson"],
    publication_date=datetime.now().isoformat(),
    keywords=["AI", "ML", "Python"],
    abstract="This paper explores...",
    page_count=15
)

# Create document
paper = Text(
    "Abstract: Machine learning has revolutionized...",
    metadata
)

# Access custom metadata
print(f"Title: {paper.metadata.get_value('title')}")
print(f"Authors: {paper.metadata.get_value('authors')}")
print(f"Keywords: {paper.metadata.get_value('keywords')}")
```

### Document Processing Pipeline

```python
from dataknobs_structures import Text, TextMetaData

def process_document(doc: Text) -> Text:
    """Process a document and update its metadata"""
    
    # Count words
    word_count = len(doc.text.split())
    
    # Count sentences (simple approximation)
    sentence_count = doc.text.count('.') + doc.text.count('!') + doc.text.count('?')
    
    # Update metadata
    doc.metadata.data['word_count'] = word_count
    doc.metadata.data['sentence_count'] = sentence_count
    doc.metadata.data['processed'] = True
    
    return doc

# Create and process document
metadata = TextMetaData(text_id="doc_001", text_label="article")
doc = Text("This is a sample document. It has multiple sentences! Isn't that great?", metadata)

processed_doc = process_document(doc)

print(f"Words: {processed_doc.metadata.get_value('word_count')}")
print(f"Sentences: {processed_doc.metadata.get_value('sentence_count')}")
print(f"Processed: {processed_doc.metadata.get_value('processed')}")
```

### Integration with Other Packages

```python
from dataknobs_structures import Text, TextMetaData
# from dataknobs_utils import json_utils  # Example integration

# Create document
metadata = TextMetaData(
    text_id="integration_example",
    text_label="demo"
)

doc = Text("Hello World! This is a test document.", metadata)

# Convert to dictionary for serialization
doc_dict = {
    'text': doc.text,
    'metadata': doc.metadata.data
}

# Example: could serialize with json_utils
# json_str = json_utils.to_json(doc_dict)
```

### Error Handling

```python
from dataknobs_structures import Text, TextMetaData

def safe_document_creation(text_content, text_id):
    """Safely create a document with error handling"""
    try:
        # Validate inputs
        if not isinstance(text_content, str):
            raise ValueError("Text content must be a string")
        
        if not text_content.strip():
            raise ValueError("Text content cannot be empty")
        
        # Create metadata
        metadata = TextMetaData(text_id=text_id)
        
        # Create document
        doc = Text(text_content, metadata)
        
        return doc
        
    except Exception as e:
        print(f"Error creating document: {e}")
        return None

# Use safe creation
doc = safe_document_creation("Valid content", "doc_001")
if doc:
    print(f"Created document: {doc.text_id}")

# This will fail safely
invalid_doc = safe_document_creation("", "doc_002")
```

### Default Metadata Handling

```python
from dataknobs_structures import Text

# Document without explicit metadata gets defaults
doc = Text("Content without metadata")

print(f"Default ID: {doc.text_id}")     # 0
print(f"Default label: {doc.text_label}") # "text"

# Access the auto-created metadata
print(f"All metadata: {doc.metadata.data}")
```

## Best Practices

1. **Always Provide Meaningful IDs**: Use descriptive, unique identifiers
2. **Include Rich Metadata**: Add as much contextual information as possible
3. **Use Consistent Labels**: Establish conventions for text_label values
4. **Handle Missing Metadata Gracefully**: Use `get_value()` with defaults
5. **Validate Input**: Check text content and metadata before document creation

## Integration Points

The document classes are designed to work with:

- **dataknobs-utils**: For serialization and file operations
- **dataknobs-xization**: For text processing and analysis
- **dataknobs-structures.tree**: Documents can be stored in tree structures

## Performance Considerations

- Metadata is stored as dictionaries for flexible access
- Text content is stored as strings (consider memory for large documents)
- Metadata updates modify the underlying dictionary in-place
- Use appropriate data types for metadata values to optimize memory usage

## See Also

- [Text Processing with Xization](../xization/index.md)
- [Tree Structures](tree.md)
- [Record Storage](record-store.md)