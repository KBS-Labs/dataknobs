# API Reference

Complete API documentation for all Dataknobs packages.

## Package APIs

- [dataknobs-fsm](dataknobs-fsm.md) - Finite State Machine framework API
- [dataknobs-data](dataknobs-data.md) - Data abstraction API
- [dataknobs-config](dataknobs-config.md) - Configuration management API
- [dataknobs-structures](dataknobs-structures.md) - Core data structures API
- [dataknobs-utils](dataknobs-utils.md) - Utility functions API
- [dataknobs-xization](dataknobs-xization.md) - Text processing API
- [dataknobs-common](dataknobs-common.md) - Common components API

## Quick Reference

### FSM
```python
from dataknobs_fsm import SimpleFSM, AdvancedFSM, AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
```

### Data
- See [dataknobs-data API reference](../packages/data/api-reference.md) documentation

### Config
```python
from dataknobs_config import Config
```

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
