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

### AI & LLM Packages
- [dataknobs-bots](dataknobs-bots.md) - AI chatbots and agents API
- [dataknobs-llm](dataknobs-llm.md) - LLM integration API

### Data & Configuration Packages
- [dataknobs-data](dataknobs-data.md) - Data abstraction API
- [dataknobs-config](dataknobs-config.md) - Configuration management API

### Workflow & Processing Packages
- [dataknobs-fsm](dataknobs-fsm.md) - Finite State Machine framework API

### Core Utilities Packages
- [dataknobs-structures](dataknobs-structures.md) - Core data structures API
- [dataknobs-utils](dataknobs-utils.md) - Utility functions API
- [dataknobs-xization](dataknobs-xization.md) - Text processing API
- [dataknobs-common](dataknobs-common.md) - Common components API

## Quick Reference

### Bots
```python
from dataknobs_bots import DynaBot, BotContext
```

### LLM
```python
from dataknobs_llm import (
    create_llm_provider,
    LLMMessage,
    LLMResponse,
    MessageTemplate,
    MessageBuilder,
    Tool,
    ToolRegistry
)
```

### FSM
```python
from dataknobs_fsm import SimpleFSM, AdvancedFSM, AsyncSimpleFSM, DataHandlingMode
```

### Data
```python
from dataknobs_data import (
    database_factory,
    async_database_factory,
    Record,
    Query,
    Filter,
    Operator
)
```
- See [dataknobs-data API reference](../packages/data/api-reference.md) for complete documentation

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
