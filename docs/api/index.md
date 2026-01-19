# API Reference

Complete API documentation for all Dataknobs packages.

## Documentation Types

### 📖 [Complete Reference](complete-reference.md)
**Auto-generated comprehensive documentation** showing all classes, methods, and functions with full signatures and source code. Best for:
- Looking up specific method signatures
- Exploring all available functionality
- Understanding type annotations
- Browsing source code

### 📚 Curated Guides (below)
**Hand-crafted guides with examples** focusing on common use cases and best practices. Best for:
- Learning how to use the API
- Understanding design patterns
- Seeing practical examples
- Quick reference with context

---

## Package APIs

All packages listed alphabetically:

- [dataknobs-bots](dataknobs-bots.md) - AI chatbots and agents API
- [dataknobs-common](dataknobs-common.md) - Common components, registries, and exceptions API
- [dataknobs-config](dataknobs-config.md) - Configuration management API
- [dataknobs-data](dataknobs-data.md) - Data abstraction API
- [dataknobs-fsm](dataknobs-fsm.md) - Finite State Machine framework API
- [dataknobs-llm](dataknobs-llm.md) - LLM integration API
- [dataknobs-structures](dataknobs-structures.md) - Core data structures API
- [dataknobs-utils](dataknobs-utils.md) - Utility functions API
- [dataknobs-xization](dataknobs-xization.md) - Text processing API

## Quick Reference

### Bots
```python
from dataknobs_bots import DynaBot, BotContext, BotRegistry
```

### Common
```python
from dataknobs_common import DataknobsError, ValidationError, Registry, serialize
```

### Config
```python
from dataknobs_config import Config, EnvironmentConfig, EnvironmentAwareConfig
```

### Data
```python
from dataknobs_data import database_factory, async_database_factory, Record, Query, Filter, Operator
```

### FSM
```python
from dataknobs_fsm import SimpleFSM, AsyncSimpleFSM, AdvancedFSM, DataHandlingMode
```

### LLM
```python
from dataknobs_llm import create_llm_provider, LLMConfig, LLMMessage, Tool, ToolRegistry
```

### Structures
```python
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict
```

### Utils
```python
from dataknobs_utils import json_utils, file_utils, requests_utils, pandas_utils
```

### Xization
```python
from dataknobs_xization import MarkdownChunker, parse_markdown, normalize, annotations
# Access: normalize.basic_normalization_fn(), annotations.Annotations
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
