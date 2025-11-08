# Dataknobs Structures

Core data structures for AI knowledge bases and data processing pipelines.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/structures.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/structures/src/dataknobs_structures) - Browse on GitHub
> - [Package Guide](../packages/structures/index.md) - Detailed documentation

## Overview

The `dataknobs-structures` package provides fundamental data structures designed for building AI applications, knowledge bases, and data processing workflows. Each structure is designed to be simple, flexible, and practical for real-world use cases.

**Core Components:**

- **Tree** - Hierarchical data structures with parent-child relationships
- **Text & TextMetaData** - Document containers with metadata
- **RecordStore** - Flexible tabular data with multiple representations
- **cdict** - Validated dictionaries using the strategy pattern

## Installation

```bash
pip install dataknobs-structures
```

## Quick Start

### Tree - Build and traverse hierarchies

```python
from dataknobs_structures import Tree

# Build a tree structure
root = Tree("root")
child1 = root.add_child("child1")
child2 = root.add_child("child2")
grandchild = child1.add_child("grandchild")

# Search nodes
found = root.find_nodes(lambda n: "child" in str(n.data))

# Get all edges
edges = root.get_edges()  # [(root, child1), (root, child2), ...]
```

### Text - Manage documents with metadata

```python
from dataknobs_structures import Text, TextMetaData

# Create document with metadata
metadata = TextMetaData(
    text_id="doc_001",
    text_label="article",
    author="Alice",
    category="technology"
)
doc = Text("This is the document content...", metadata)

print(doc.text_id)     # "doc_001"
print(doc.text_label)  # "article"
```

### RecordStore - Manage tabular data

```python
from dataknobs_structures import RecordStore

# Create store with disk backing
store = RecordStore("/data/users.tsv")

# Add records
store.add_rec({"id": 1, "name": "Alice", "age": 30})
store.add_rec({"id": 2, "name": "Bob", "age": 25})

# Access as DataFrame or list
df = store.df
records = store.records

# Persist to disk
store.save()
```

### cdict - Validated dictionaries

```python
from dataknobs_structures import cdict

# Only accept positive integers
positive = cdict(lambda d, k, v: isinstance(v, int) and v > 0)
positive['a'] = 5    # Accepted
positive['b'] = -1   # Rejected

print(positive)          # {'a': 5}
print(positive.rejected)  # {'b': -1}
```

## Use Cases by Problem Domain

### Hierarchical Data Processing

Tree structures are ideal for representing hierarchical relationships in data:

#### Parsing and representing syntax trees

```python
from dataknobs_structures import Tree

# Parse expression into tree
def parse_expression(expr):
    """Parse '(a + b) * c' into tree structure."""
    root = Tree("*")
    left = root.add_child("+")
    left.add_child("a")
    left.add_child("b")
    root.add_child("c")
    return root

tree = parse_expression("(a + b) * c")

# Find all operators
operators = tree.find_nodes(lambda n: n.data in ['+', '-', '*', '/'])

# Get tree depth
max_depth = max(node.depth for node in tree.find_nodes(lambda n: True))
```

#### Building taxonomies and ontologies

```python
from dataknobs_structures import Tree

# Build taxonomy
taxonomy = Tree("living_things")
animals = taxonomy.add_child("animals")
mammals = animals.add_child("mammals")
mammals.add_child("dogs")
mammals.add_child("cats")
birds = animals.add_child("birds")
birds.add_child("eagles")
birds.add_child("sparrows")

# Find all terminal categories (leaf nodes)
leaf_categories = taxonomy.collect_terminal_nodes()
# Returns: [dogs, cats, eagles, sparrows]

# Get path to a category
dogs = taxonomy.find_nodes(lambda n: n.data == "dogs", only_first=True)[0]
path = taxonomy.get_path(dogs)
# Returns: [living_things, animals, mammals, dogs]
```

#### Representing file system hierarchies

```python
from dataknobs_structures import Tree
import os

def build_directory_tree(root_path):
    """Build tree representation of directory structure."""
    root = Tree(os.path.basename(root_path))

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            # Recursively build subtree
            subtree = build_directory_tree(item_path)
            root.add_child(subtree)
        else:
            # Add file as leaf
            root.add_child(item)

    return root

# Build and analyze
dir_tree = build_directory_tree("/path/to/project")

# Find all Python files
py_files = dir_tree.find_nodes(lambda n: str(n.data).endswith('.py'))

# Get deepest nesting level
max_depth = max(node.depth for node in dir_tree.find_nodes(lambda n: True))
print(f"Maximum directory depth: {max_depth}")
```

#### Building conversation threads

```python
from dataknobs_structures import Tree

# Build conversation thread
thread = Tree("Original Post")
reply1 = thread.add_child("Reply 1")
reply2 = thread.add_child("Reply 2")
nested1 = reply1.add_child("Reply to Reply 1")
nested2 = reply1.add_child("Another reply")
nested3 = reply2.add_child("Reply to Reply 2")

# Find deepest conversation branch
deepest = thread.get_deepest_left()
depth = deepest.depth
print(f"Maximum thread depth: {depth}")

# Get all replies to a specific comment
all_descendants = reply1.find_nodes(lambda n: True)  # All nodes in subtree
```

### Document Management

Text and TextMetaData provide containers for managing documents with rich metadata:

#### Processing document collections

```python
from dataknobs_structures import Text, TextMetaData

# Create document collection
documents = []

for doc_file in document_files:
    metadata = TextMetaData(
        text_id=doc_file.stem,
        text_label=doc_file.parent.name,  # Category from directory
        filename=doc_file.name,
        size=doc_file.stat().st_size,
        created=doc_file.stat().st_mtime
    )

    content = doc_file.read_text()
    doc = Text(content, metadata)
    documents.append(doc)

# Filter by category
tech_docs = [d for d in documents if d.text_label == "technology"]

# Find documents by metadata
large_docs = [
    d for d in documents
    if d.metadata.get_value("size") > 10000
]
```

#### Labeled datasets for ML

```python
from dataknobs_structures import Text, TextMetaData

# Build training dataset
training_data = []

for label, texts in labeled_texts.items():
    for i, text in enumerate(texts):
        metadata = TextMetaData(
            text_id=f"{label}_{i}",
            text_label=label,
            split="train",
            length=len(text)
        )
        training_data.append(Text(text, metadata))

# Split by metadata
train_set = [d for d in training_data if d.metadata.get_value("split") == "train"]
test_set = [d for d in training_data if d.metadata.get_value("split") == "test"]

# Group by label
from collections import defaultdict
by_label = defaultdict(list)
for doc in training_data:
    by_label[doc.text_label].append(doc)
```

#### Search result containers

```python
from dataknobs_structures import Text, TextMetaData

# Store search results with metadata
def create_search_result(hit):
    """Convert search engine hit to Text object."""
    metadata = TextMetaData(
        text_id=hit['id'],
        text_label="search_result",
        title=hit['title'],
        url=hit['url'],
        score=hit['relevance_score'],
        rank=hit['rank'],
        timestamp=hit['retrieved_at']
    )

    return Text(hit['snippet'], metadata)

# Process results
results = [create_search_result(hit) for hit in search_hits]

# Sort by score
results.sort(key=lambda r: r.metadata.get_value("score"), reverse=True)

# Filter by threshold
high_quality = [r for r in results if r.metadata.get_value("score") > 0.8]
```

### Data Collection & Analysis

RecordStore bridges the gap between Python dictionaries and pandas DataFrames:

#### Experiment result tracking

```python
from dataknobs_structures import RecordStore
import time

# Initialize experiment log
results = RecordStore("/experiments/results.tsv")

# Run experiments and log results
for model_name in models:
    for hyperparams in param_grid:
        start = time.time()

        # Train and evaluate
        model = train_model(model_name, hyperparams)
        metrics = evaluate_model(model, test_data)

        # Log results
        results.add_rec({
            "model": model_name,
            "params": str(hyperparams),
            "accuracy": metrics['accuracy'],
            "f1_score": metrics['f1'],
            "training_time": time.time() - start,
            "timestamp": time.time()
        })

        # Save after each experiment
        results.save()

# Analyze with pandas
df = results.df
best_model = df.loc[df['accuracy'].idxmax()]
print(f"Best model: {best_model['model']} with accuracy {best_model['accuracy']}")
```

#### Data pipeline checkpoints

```python
from dataknobs_structures import RecordStore

def process_with_checkpoints(input_data, checkpoint_file):
    """Process data with automatic checkpointing."""
    # Initialize or restore checkpoint
    processed = RecordStore(checkpoint_file)

    # Track what's been processed
    processed_ids = {r['id'] for r in processed.records}

    # Process remaining items
    for item in input_data:
        if item['id'] not in processed_ids:
            try:
                result = process_item(item)
                processed.add_rec({
                    "id": item['id'],
                    "status": "success",
                    "result": result,
                    "timestamp": time.time()
                })
                processed.save()  # Save after each item

            except Exception as e:
                processed.add_rec({
                    "id": item['id'],
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                })
                processed.save()

    return processed

# Can be restarted after failure
results = process_with_checkpoints(large_dataset, "/tmp/checkpoint.tsv")
```

#### Batch processing logs

```python
from dataknobs_structures import RecordStore
from datetime import datetime

# Initialize processing log
log = RecordStore("/logs/batch_processing.tsv")

# Process batches
for batch_num, batch in enumerate(data_batches):
    batch_start = datetime.now()

    try:
        results = process_batch(batch)

        log.add_rec({
            "batch_num": batch_num,
            "batch_size": len(batch),
            "status": "success",
            "processed": len(results),
            "duration": (datetime.now() - batch_start).total_seconds(),
            "timestamp": batch_start.isoformat()
        })

    except Exception as e:
        log.add_rec({
            "batch_num": batch_num,
            "batch_size": len(batch),
            "status": "failed",
            "error": str(e),
            "duration": (datetime.now() - batch_start).total_seconds(),
            "timestamp": batch_start.isoformat()
        })

    log.save()

# Analyze logs
df = log.df
success_rate = (df['status'] == 'success').mean()
avg_duration = df[df['status'] == 'success']['duration'].mean()
print(f"Success rate: {success_rate:.1%}, Avg duration: {avg_duration:.2f}s")
```

### Validated Data Structures

cdict enables flexible data validation using the strategy pattern:

#### Type-safe configurations

```python
from dataknobs_structures import cdict

def validate_config(d, key, value):
    """Validate configuration key-value pairs."""
    valid_keys = {'host', 'port', 'timeout', 'retry', 'verbose'}

    if key not in valid_keys:
        return False

    # Type validation
    if key == 'host' and not isinstance(value, str):
        return False
    if key in ('port', 'timeout', 'retry') and not isinstance(value, int):
        return False
    if key == 'verbose' and not isinstance(value, bool):
        return False

    # Range validation
    if key == 'port' and not (1 <= value <= 65535):
        return False
    if key in ('timeout', 'retry') and value < 0:
        return False

    return True

# Create validated config
config = cdict(validate_config)
config['host'] = 'localhost'      # ✓ Accepted
config['port'] = 8080             # ✓ Accepted
config['timeout'] = 30            # ✓ Accepted
config['invalid_key'] = 'value'   # ✗ Rejected
config['port'] = 99999            # ✗ Rejected (out of range)

print(config)           # Valid configuration
print(config.rejected)  # Invalid attempts
```

#### Filtering data by business rules

```python
from dataknobs_structures import cdict

# Filter valid user records
def is_valid_user(d, key, user_data):
    """Validate user data meets business rules."""
    required_fields = {'username', 'email', 'age'}

    # Check required fields
    if not all(field in user_data for field in required_fields):
        return False

    # Age validation
    if not (18 <= user_data['age'] <= 120):
        return False

    # Email validation (simple check)
    if '@' not in user_data['email']:
        return False

    # Username uniqueness (using dict state)
    if any(u.get('username') == user_data['username'] for u in d.values()):
        return False

    return True

users = cdict(is_valid_user)

# Add users
users['user1'] = {'username': 'alice', 'email': 'alice@example.com', 'age': 30}  # ✓
users['user2'] = {'username': 'bob', 'email': 'bob@example.com', 'age': 25}      # ✓
users['user3'] = {'username': 'alice', 'email': 'other@example.com', 'age': 28}  # ✗ Duplicate username
users['user4'] = {'username': 'charlie', 'email': 'invalid', 'age': 15}          # ✗ Invalid email and age

print(f"Valid users: {len(users)}")
print(f"Rejected: {len(users.rejected)}")
```

#### Preventing duplicate entries

```python
from dataknobs_structures import cdict

# Prevent duplicate keys
no_overwrites = cdict(lambda d, k, v: k not in d)

no_overwrites['x'] = 1   # ✓ Accepted (new key)
no_overwrites['y'] = 2   # ✓ Accepted (new key)
no_overwrites['x'] = 10  # ✗ Rejected (key exists)

# Prevent duplicate values
def no_duplicate_values(d, key, value):
    return value not in d.values()

unique_values = cdict(no_duplicate_values)
unique_values['a'] = 1  # ✓ Accepted
unique_values['b'] = 2  # ✓ Accepted
unique_values['c'] = 1  # ✗ Rejected (value 1 already exists)
```

## Integration Examples

These components work well together in larger systems:

### Knowledge base with hierarchical documents

```python
from dataknobs_structures import Tree, Text, TextMetaData

class KnowledgeBase:
    """Knowledge base with hierarchical organization."""

    def __init__(self):
        self.taxonomy = Tree("root")
        self.documents = {}

    def add_category(self, parent_name, category_name):
        """Add category to taxonomy."""
        parent = self.taxonomy.find_nodes(lambda n: n.data == parent_name)[0]
        return parent.add_child(category_name)

    def add_document(self, category_name, text, doc_id):
        """Add document to knowledge base."""
        # Find category in taxonomy
        category_node = self.taxonomy.find_nodes(
            lambda n: n.data == category_name
        )[0]

        # Create document with metadata
        metadata = TextMetaData(
            text_id=doc_id,
            text_label=category_name,
            category_path='/'.join(str(n.data) for n in self.taxonomy.get_path(category_node))
        )
        doc = Text(text, metadata)

        self.documents[doc_id] = doc
        return doc

    def get_documents_in_category(self, category_name, include_subcategories=True):
        """Get all documents in a category."""
        category = self.taxonomy.find_nodes(lambda n: n.data == category_name)[0]

        if include_subcategories:
            # Get all descendant categories
            subcategories = [category] + category.find_nodes(lambda n: True)
            category_names = {str(n.data) for n in subcategories}
        else:
            category_names = {category_name}

        # Find documents in these categories
        return [
            doc for doc in self.documents.values()
            if doc.text_label in category_names
        ]

# Build knowledge base
kb = KnowledgeBase()
kb.add_category("root", "science")
kb.add_category("science", "physics")
kb.add_category("science", "biology")

# Add documents
kb.add_document("physics", "Newton's laws...", "doc1")
kb.add_document("biology", "Cell structure...", "doc2")

# Query
physics_docs = kb.get_documents_in_category("physics")
all_science = kb.get_documents_in_category("science", include_subcategories=True)
```

### Data processing pipeline with validated records

```python
from dataknobs_structures import RecordStore, cdict

class ValidatedPipeline:
    """Data pipeline with validation and logging."""

    def __init__(self, output_file, validation_fn):
        self.output = RecordStore(output_file)
        self.validation_fn = validation_fn

    def process(self, input_records):
        """Process records with validation."""
        # Validate inputs
        validated = cdict(self.validation_fn)
        validated.update({r['id']: r for r in input_records})

        # Process valid records
        for record_id, record in validated.items():
            try:
                result = self._process_record(record)
                self.output.add_rec({
                    "id": record_id,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                self.output.add_rec({
                    "id": record_id,
                    "status": "error",
                    "error": str(e)
                })

        # Log rejected records
        for record_id, record in validated.rejected.items():
            self.output.add_rec({
                "id": record_id,
                "status": "rejected",
                "reason": "validation_failed"
            })

        self.output.save()
        return self.output

    def _process_record(self, record):
        """Process a single record."""
        # Processing logic here
        return {"processed": True}

# Use pipeline
def validate_record(d, record_id, record):
    return 'required_field' in record and record['required_field'] is not None

pipeline = ValidatedPipeline("/output/results.tsv", validate_record)
results = pipeline.process(input_data)

# Analyze results
df = results.df
print(f"Success: {(df['status'] == 'success').sum()}")
print(f"Rejected: {(df['status'] == 'rejected').sum()}")
print(f"Errors: {(df['status'] == 'error').sum()}")
```

### Hierarchical experiment tracking

```python
from dataknobs_structures import Tree, RecordStore

class ExperimentTracker:
    """Track experiments with hierarchical organization."""

    def __init__(self, log_file):
        self.experiments = Tree("all_experiments")
        self.results = RecordStore(log_file)

    def create_experiment_group(self, parent_name, group_name):
        """Create experiment group in hierarchy."""
        if parent_name is None:
            return self.experiments.add_child(group_name)
        else:
            parent = self.experiments.find_nodes(lambda n: n.data == parent_name)[0]
            return parent.add_child(group_name)

    def log_experiment(self, group_name, experiment_id, params, metrics):
        """Log experiment results."""
        # Find group in tree
        group = self.experiments.find_nodes(lambda n: n.data == group_name)[0]

        # Add to results with path
        path = '/'.join(str(n.data) for n in self.experiments.get_path(group))

        self.results.add_rec({
            "experiment_id": experiment_id,
            "group": group_name,
            "path": path,
            **params,
            **metrics
        })
        self.results.save()

    def get_results_for_group(self, group_name, include_subgroups=True):
        """Get results for experiment group."""
        if include_subgroups:
            group = self.experiments.find_nodes(lambda n: n.data == group_name)[0]
            subgroups = {str(n.data) for n in [group] + group.find_nodes(lambda n: True)}
        else:
            subgroups = {group_name}

        # Filter results
        return [r for r in self.results.records if r['group'] in subgroups]

# Use tracker
tracker = ExperimentTracker("/experiments/log.tsv")

# Create hierarchy
tracker.create_experiment_group(None, "model_comparison")
tracker.create_experiment_group("model_comparison", "bert_variants")
tracker.create_experiment_group("model_comparison", "gpt_variants")

# Log experiments
tracker.log_experiment(
    "bert_variants",
    "exp_001",
    {"model": "bert-base", "lr": 0.001},
    {"accuracy": 0.92, "f1": 0.89}
)

# Query results
bert_results = tracker.get_results_for_group("bert_variants")
all_results = tracker.get_results_for_group("model_comparison", include_subgroups=True)
```

## Design Patterns

### Builder Pattern with Tree

```python
from dataknobs_structures import Tree

class TreeBuilder:
    """Builder for constructing trees fluently."""

    def __init__(self, root_data):
        self.root = Tree(root_data)
        self.current = self.root

    def add_child(self, data):
        """Add child to current node."""
        self.current = self.current.add_child(data)
        return self

    def add_sibling(self, data):
        """Add sibling to current node."""
        if self.current.parent:
            self.current = self.current.parent.add_child(data)
        return self

    def up(self):
        """Move up to parent."""
        if self.current.parent:
            self.current = self.current.parent
        return self

    def build(self):
        """Return the constructed tree."""
        return self.root

# Use builder
tree = (TreeBuilder("root")
    .add_child("child1")
        .add_child("grandchild1")
        .up()
        .add_child("grandchild2")
        .up()
    .up()
    .add_child("child2")
    .build())
```

### Factory Pattern for Documents

```python
from dataknobs_structures import Text, TextMetaData
from datetime import datetime

class DocumentFactory:
    """Factory for creating documents with consistent metadata."""

    def __init__(self, default_label, **default_metadata):
        self.default_label = default_label
        self.default_metadata = default_metadata
        self.counter = 0

    def create(self, text, **metadata):
        """Create document with auto-generated ID and defaults."""
        self.counter += 1

        # Merge metadata
        full_metadata = {
            **self.default_metadata,
            **metadata,
            'created_at': datetime.now().isoformat()
        }

        meta = TextMetaData(
            text_id=f"{self.default_label}_{self.counter:04d}",
            text_label=self.default_label,
            **full_metadata
        )

        return Text(text, meta)

# Use factory
article_factory = DocumentFactory(
    "article",
    source="web_scraper",
    format="html"
)

doc1 = article_factory.create("Article text...", url="http://example.com/1")
doc2 = article_factory.create("Another article...", url="http://example.com/2")
```

### Repository Pattern with RecordStore

```python
from dataknobs_structures import RecordStore

class Repository:
    """Repository pattern for data access."""

    def __init__(self, storage_file):
        self.store = RecordStore(storage_file)
        self._build_index()

    def _build_index(self):
        """Build index for fast lookups."""
        self.index = {r['id']: r for r in self.store.records}

    def add(self, record):
        """Add record to repository."""
        self.store.add_rec(record)
        self.index[record['id']] = record
        self.store.save()

    def get(self, record_id):
        """Get record by ID."""
        return self.index.get(record_id)

    def find(self, predicate):
        """Find records matching predicate."""
        return [r for r in self.store.records if predicate(r)]

    def update(self, record_id, updates):
        """Update existing record."""
        if record_id in self.index:
            self.index[record_id].update(updates)
            # Rebuild store from index
            self.store.clear()
            for record in self.index.values():
                self.store.add_rec(record)
            self.store.save()

    def delete(self, record_id):
        """Delete record."""
        if record_id in self.index:
            del self.index[record_id]
            self.store.clear()
            for record in self.index.values():
                self.store.add_rec(record)
            self.store.save()

# Use repository
repo = Repository("/data/users.tsv")
repo.add({"id": "u1", "name": "Alice", "role": "admin"})
repo.add({"id": "u2", "name": "Bob", "role": "user"})

# Query
user = repo.get("u1")
admins = repo.find(lambda r: r['role'] == 'admin')

# Update
repo.update("u1", {"last_login": "2024-01-15"})
```

### Strategy Pattern with cdict

```python
from dataknobs_structures import cdict

class ValidationStrategy:
    """Strategy pattern for validation rules."""

    @staticmethod
    def positive_numbers(d, k, v):
        """Only accept positive numbers."""
        return isinstance(v, (int, float)) and v > 0

    @staticmethod
    def non_empty_strings(d, k, v):
        """Only accept non-empty strings."""
        return isinstance(v, str) and len(v.strip()) > 0

    @staticmethod
    def unique_values(d, k, v):
        """Only accept unique values."""
        return v not in d.values()

    @staticmethod
    def email_addresses(d, k, v):
        """Only accept email addresses."""
        return isinstance(v, str) and '@' in v and '.' in v.split('@')[1]

# Use strategies
positive_dict = cdict(ValidationStrategy.positive_numbers)
string_dict = cdict(ValidationStrategy.non_empty_strings)
unique_dict = cdict(ValidationStrategy.unique_values)
email_dict = cdict(ValidationStrategy.email_addresses)

# Combine strategies
def combined_validation(d, k, v):
    """Combine multiple validation strategies."""
    return (
        ValidationStrategy.non_empty_strings(d, k, v) and
        ValidationStrategy.unique_values(d, k, v)
    )

validated = cdict(combined_validation)
```

## Best Practices

### Tree Best Practices

1. **Use find_nodes for queries, not manual traversal**
   ```python
   # Good
   found = root.find_nodes(lambda n: n.data == "target")

   # Avoid manual traversal
   for child in root.children:
       for grandchild in child.children:
           # ...
   ```

2. **Leverage method chaining with add_child**
   ```python
   root = Tree("root").add_child("a").parent.add_child("b")
   ```

3. **Use get_path for ancestry information**
   ```python
   path = root.get_path(node)
   path_str = " > ".join(str(n.data) for n in path)
   ```

### Text & Metadata Best Practices

1. **Use TextMetaData for documents, MetaData for generic data**
   ```python
   # For documents
   doc = Text("content", TextMetaData(text_id="doc1", text_label="article"))

   # For arbitrary metadata
   metadata = MetaData(author="Alice", year=2024)
   ```

2. **Use get_value() with defaults for optional metadata**
   ```python
   author = doc.metadata.get_value("author", default="Unknown")
   ```

3. **Store metadata that enables filtering and organization**
   ```python
   metadata = TextMetaData(
       text_id="doc1",
       text_label="article",
       category="tech",      # For filtering
       language="en",        # For filtering
       priority=1,           # For sorting
       created_at="2024-01-15"  # For sorting
   )
   ```

### RecordStore Best Practices

1. **Call save() after important operations**
   ```python
   store.add_rec(important_record)
   store.save()  # Persist immediately
   ```

2. **Use restore() for undo functionality**
   ```python
   original_count = len(store.records)
   store.add_rec({"test": "data"})
   store.restore()  # Back to original state
   ```

3. **Access df property only when needed (it's lazily created)**
   ```python
   # Efficient: work with records
   for rec in store.records:
       process(rec)

   # Use DataFrame when you need pandas functionality
   df = store.df
   summary = df.describe()
   ```

### cdict Best Practices

1. **Check rejected dict after updates**
   ```python
   validated.update(user_input)
   if validated.rejected:
       log_validation_errors(validated.rejected)
   ```

2. **Use dict state in validation functions**
   ```python
   # Prevent duplicates
   unique = cdict(lambda d, k, v: k not in d)

   # Enforce constraints based on existing data
   def max_items(d, k, v):
       return len(d) < 100

   limited = cdict(max_items)
   ```

3. **Validation functions should be pure (no side effects)**
   ```python
   # Good
   def validate(d, k, v):
       return isinstance(v, int) and v > 0

   # Bad - has side effects
   def validate_with_logging(d, k, v):
       logger.info(f"Validating {k}={v}")  # Side effect!
       return isinstance(v, int)
   ```

## Complete API Reference

For comprehensive auto-generated API documentation with all classes, methods, and functions including full signatures and type annotations, see:

**[📖 dataknobs-structures Complete API Reference](reference/structures.md)**

This curated guide focuses on practical examples and usage patterns. The complete reference provides exhaustive technical documentation auto-generated from source code docstrings.
