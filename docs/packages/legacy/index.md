# Dataknobs Legacy

The `dataknobs` package is the legacy compatibility package that provides backward compatibility for existing users.

## ⚠️ Deprecation Notice

**This package is deprecated.** Please migrate to the modular packages:

- `dataknobs-common`
- `dataknobs-structures` 
- `dataknobs-utils`
- `dataknobs-xization`

## Installation

```bash
pip install dataknobs
```

**Note**: Installing this package will show a deprecation warning.

## Purpose

The legacy package exists to:

1. **Maintain Backward Compatibility**: Existing code continues to work
2. **Facilitate Migration**: Provides time to migrate to modular packages
3. **Gradual Transition**: Allows incremental adoption of new packages

## Package Structure

```
dataknobs/
├── src/
│   └── dataknobs/
│       ├── __init__.py          # Deprecation warning
│       ├── flask_api.py         # Legacy Flask API
│       ├── structures/
│       │   └── __init__.py
│       ├── utils/
│       │   └── __init__.py
│       └── xization/
│           └── __init__.py
└── tests/
```

## Migration Guide

### Current Usage (Deprecated)

```python
# OLD - Deprecated usage
import dataknobs
from dataknobs.structures import Tree
from dataknobs.utils import json_utils
from dataknobs.xization import normalize
```

### New Usage (Recommended)

```python
# NEW - Modular packages
from dataknobs_structures import Tree
from dataknobs_utils import json_utils  
from dataknobs_xization import normalize
```

## Migration Steps

### Step 1: Install New Packages

```bash
# Install the modular packages
pip install dataknobs-structures dataknobs-utils dataknobs-xization dataknobs-common

# Optional: Keep legacy package during transition
pip install dataknobs
```

### Step 2: Update Imports

Replace legacy imports with modular package imports:

#### Structures

```python
# OLD
from dataknobs.structures import Tree, Text, TextMetaData, RecordStore, cdict

# NEW  
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict
```

#### Utils

```python
# OLD
from dataknobs.utils import json_utils, file_utils, elasticsearch_utils

# NEW
from dataknobs_utils import json_utils, file_utils, elasticsearch_utils
```

#### Xization

```python
# OLD
from dataknobs.xization import normalize, masking_tokenizer

# NEW
from dataknobs_xization import normalize
from dataknobs_xization.masking_tokenizer import TextFeatures, CharacterFeatures
```

### Step 3: Update Dependencies

Update your `requirements.txt` or `pyproject.toml`:

```txt
# OLD
dataknobs>=0.0.15

# NEW
dataknobs-structures>=1.0.0
dataknobs-utils>=1.0.0
dataknobs-xization>=1.0.0
```

### Step 4: Remove Legacy Package

After migration is complete:

```bash
pip uninstall dataknobs
```

## Common Migration Patterns

### Tree Operations

```python
# Works in both old and new
from dataknobs_structures import Tree  # Preferred

tree = Tree("root")
child = tree.add_child("child")
```

### JSON Processing

```python
# Works in both old and new
from dataknobs_utils import json_utils  # Preferred

data = {"key": "value"}
result = json_utils.get_value(data, "key")
```

### Text Normalization

```python
# Works in both old and new
from dataknobs_xization import normalize  # Preferred

text = "CamelCaseText"
normalized = normalize.basic_normalization_fn(text, expand_camelcase=True)
```

## Breaking Changes

### None Expected

The modular packages maintain API compatibility with the legacy package. No breaking changes are expected during migration.

### Version Mapping

| Legacy Version | Modular Packages Version | Notes |
|---------------|--------------------------|-------|
| 0.0.15 | 1.0.0 | Initial modular release |

## Legacy Flask API

The legacy package includes a Flask API that is not migrated to the modular packages:

```python
from dataknobs.flask_api import create_app

# This functionality is deprecated and not available in modular packages
app = create_app()
```

If you need web API functionality, consider building a new API using modern frameworks with the modular packages.

## Support Timeline

| Period | Legacy Package | Support Level |
|--------|---------------|---------------|
| Now - 6 months | Available | Full support |
| 6-12 months | Available | Security fixes only |
| 12+ months | Deprecated | No support |

## Troubleshooting Migration

### Import Errors

If you see import errors after migration:

```python
# Error: ModuleNotFoundError: No module named 'dataknobs.structures'

# Solution: Update import
from dataknobs_structures import Tree  # Instead of from dataknobs.structures import Tree
```

### Version Conflicts

If you see version conflicts:

```bash
# Uninstall legacy package first
pip uninstall dataknobs

# Then install modular packages
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Testing Migration

Test your migration incrementally:

```python
# Keep both imports temporarily during testing
try:
    from dataknobs_structures import Tree  # New
except ImportError:
    from dataknobs.structures import Tree  # Fallback to old

# Remove fallback after confirming migration
```

## Benefits of Migration

### Modular Dependencies

Install only what you need:

```bash
# Only need structures? Just install that
pip install dataknobs-structures

# Need utils and xization? Install both
pip install dataknobs-utils dataknobs-xization
```

### Better Maintenance

- Smaller, focused packages
- Independent versioning
- Clearer dependency management
- Easier testing and CI/CD

### Performance

- Reduced import overhead
- Smaller memory footprint
- Faster startup times

## Migration Checklist

- [ ] Install modular packages
- [ ] Update import statements
- [ ] Update dependency files (requirements.txt, pyproject.toml)
- [ ] Run tests to verify functionality
- [ ] Update documentation
- [ ] Remove legacy package from dependencies
- [ ] Uninstall legacy package

## Example Migration

### Before (Legacy)

```python
# requirements.txt
dataknobs>=0.0.15

# main.py
from dataknobs.structures import Tree, Text, TextMetaData
from dataknobs.utils import json_utils
from dataknobs.xization import normalize

def process_data():
    # Create tree
    tree = Tree("root")
    
    # Process JSON
    data = json_utils.get_value({"key": "value"}, "key")
    
    # Normalize text
    text = normalize.basic_normalization_fn("CamelCase")
    
    return tree, data, text
```

### After (Modular)

```python
# requirements.txt  
dataknobs-structures>=1.0.0
dataknobs-utils>=1.0.0
dataknobs-xization>=1.0.0

# main.py
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

def process_data():
    # Create tree - same code!
    tree = Tree("root")
    
    # Process JSON - same code!
    data = json_utils.get_value({"key": "value"}, "key")
    
    # Normalize text - same code!
    text = normalize.basic_normalization_fn("CamelCase")
    
    return tree, data, text
```

## Getting Help

### Migration Support

1. Check the [Migration Guide](migration.md) for detailed instructions
2. Review [examples](../../examples/index.md) using modular packages
3. File issues on GitHub for migration problems

### Community

- GitHub Issues: Report migration problems
- Documentation: Comprehensive guides available
- Examples: Working code samples

## See Also

- [Migration Guide](migration.md) - Detailed migration instructions
- [Package Overview](../index.md) - Overview of all packages
- [Getting Started](../../getting-started.md) - Quick start guide