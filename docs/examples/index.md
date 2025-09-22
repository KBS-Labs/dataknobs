# Examples

Practical examples of using Dataknobs packages.

## Quick Examples

### Finite State Machines
- [Simple FSM Pipeline](fsm-simple.md)
- [ETL with FSM](fsm-etl.md)
- [Streaming Data Processing](fsm-streaming.md)
- [Debugging FSMs](fsm-debugging.md)

### Data Structures
- [Basic Tree Operations](basic-tree.md)
- [Document Processing](document-processing.md)

### Text Processing
- [Text Normalization](text-normalization.md)
- [Regex Transformations](regex-transforms.md)

### Integrations
- [Elasticsearch Integration](elasticsearch-integration.md)
- [Database Operations](database-operations.md)

## Complete Applications

### Data Processing FSM

```python
from dataknobs_fsm import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

# Define a data validation and transformation pipeline
config = {
    "name": "data_processor",
    "states": [
        {"name": "start", "is_start": True},
        {"name": "validate"},
        {"name": "transform"},
        {"name": "store"},
        {"name": "end", "is_end": True},
        {"name": "error"}
    ],
    "arcs": [
        {
            "from": "start",
            "to": "validate",
            "transform": {
                "type": "builtin",
                "name": "validate_required_fields",
                "params": {"fields": ["id", "data"]}
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
        {
            "from": "transform",
            "to": "store"
        },
        {
            "from": "store",
            "to": "end"
        }
    ]
}

# Create and run the FSM
fsm = SimpleFSM(config, data_mode=DataHandlingMode.COPY)
result = fsm.process({"id": 1, "data": "example"})
```

### Document Analysis Pipeline

```python
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_xization import basic_normalization_fn
from dataknobs_utils import json_utils

def analyze_document(content, metadata):
    # Create document
    meta = TextMetaData(**metadata)
    doc = Text(content, meta)

    # Normalize text
    normalized = basic_normalization_fn(doc.content)

    # Extract structure
    tree = Tree()
    # ... build tree from document structure

    return {
        "original": doc.content,
        "normalized": normalized,
        "metadata": metadata
    }
```
