# API Reference

Complete API documentation for all Dataknobs packages.

## Package APIs

- [dataknobs-structures](dataknobs-structures.md) - Core data structures API
- [dataknobs-utils](dataknobs-utils.md) - Utility functions API  
- [dataknobs-xization](dataknobs-xization.md) - Text processing API
- [dataknobs-common](dataknobs-common.md) - Common components API

## Quick Reference

### Structures
```python
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict
```

### Utils
```python
from dataknobs_utils import json_utils, file_utils, elasticsearch_utils
```

### Xization
```python
from dataknobs_xization import normalize, masking_tokenizer, annotations
```

## Documentation Conventions

- **Required parameters** are shown without default values
- **Optional parameters** show their default values
- **Return types** are indicated with `->` notation
- **Exceptions** are documented in the Raises section

## Type Hints

All packages use Python type hints for better IDE support and documentation:

```python
def get_value(data: dict, path: str, default: Any = None) -> Any:
    """Get a value from nested dictionary."""
    pass
```