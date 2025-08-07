# Migration Guide

## Migrating from Legacy `dataknobs` to Modular Packages

This guide helps you migrate from the monolithic `dataknobs` package to the new modular architecture.

## Overview

The `dataknobs` package has been split into focused, modular packages:

| Old Package | New Packages |
|------------|--------------|
| `dataknobs` | `dataknobs-structures`<br>`dataknobs-utils`<br>`dataknobs-xization`<br>`dataknobs-common` |

## Why Migrate?

- **Better Performance**: Install only what you need
- **Clearer Dependencies**: Each package has focused dependencies
- **Improved Maintainability**: Easier to update individual components
- **Modern Tooling**: Built with `uv` for faster dependency resolution

## Migration Steps

### Step 1: Update Dependencies

Update your `requirements.txt` or `pyproject.toml`:

**Before:**
```txt
dataknobs>=0.0.14
```

**After:**
```txt
dataknobs-structures>=1.0.0
dataknobs-utils>=1.0.0
dataknobs-xization>=1.0.0
```

### Step 2: Update Imports

Update all import statements in your code:

**Before:**
```python
from dataknobs.structures.tree import Tree
from dataknobs.utils.json_utils import get_value
from dataknobs.xization.normalize import normalize_text
```

**After:**
```python
from dataknobs_structures import Tree
from dataknobs_utils.json_utils import get_value
from dataknobs_xization.normalize import basic_normalization_fn
```

### Step 3: Common Import Changes

Here are the most common import changes:

#### Structures
```python
# Old
from dataknobs.structures.tree import Tree
from dataknobs.structures.document import Text, TextMetaData
from dataknobs.structures.record_store import RecordStore
from dataknobs.structures.conditional_dict import cdict

# New
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict
```

#### Utils
```python
# Old
from dataknobs.utils.json_utils import get_value
from dataknobs.utils.file_utils import filepath_generator
from dataknobs.utils.elasticsearch_utils import ElasticsearchClient

# New
from dataknobs_utils.json_utils import get_value
from dataknobs_utils.file_utils import filepath_generator
from dataknobs_utils.elasticsearch_utils import ElasticsearchClient
```

#### Xization
```python
# Old
from dataknobs.xization.normalize import normalize_text
from dataknobs.xization.masking_tokenizer import MaskingTokenizer

# New
from dataknobs_xization.normalize import basic_normalization_fn
from dataknobs_xization.masking_tokenizer import TextFeatures
```

## API Changes

### Renamed Functions

Some functions have been renamed for clarity:

| Old Name | New Name | Package |
|----------|----------|---------|
| `normalize_text` | `basic_normalization_fn` | `dataknobs_xization.normalize` |
| `MaskingTokenizer` | `TextFeatures` | `dataknobs_xization.masking_tokenizer` |

### Removed Functions

These functions are no longer available:
- `set_value` from `json_utils` (use dict operations directly)
- `read_file`, `write_file` from `file_utils` (use `fileline_generator` and `write_lines`)

## Automated Migration Script

Use this script to automatically update imports in your codebase:

```python
#!/usr/bin/env python3
"""
Migration script to update dataknobs imports.
Usage: python migrate_imports.py <directory>
"""

import os
import re
import sys
from pathlib import Path

# Define import mappings
IMPORT_MAPPINGS = {
    r'from dataknobs\.structures\.tree import': 'from dataknobs_structures import',
    r'from dataknobs\.structures\.document import': 'from dataknobs_structures import',
    r'from dataknobs\.structures\.record_store import': 'from dataknobs_structures import',
    r'from dataknobs\.structures\.conditional_dict import': 'from dataknobs_structures import',
    r'from dataknobs\.structures import': 'from dataknobs_structures import',
    
    r'from dataknobs\.utils\.(\w+) import': r'from dataknobs_utils.\1 import',
    r'from dataknobs\.utils import': 'from dataknobs_utils import',
    
    r'from dataknobs\.xization\.(\w+) import': r'from dataknobs_xization.\1 import',
    r'from dataknobs\.xization import': 'from dataknobs_xization import',
    
    r'import dataknobs\.structures': 'import dataknobs_structures',
    r'import dataknobs\.utils': 'import dataknobs_utils',
    r'import dataknobs\.xization': 'import dataknobs_xization',
}

def migrate_file(filepath):
    """Update imports in a single Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for pattern, replacement in IMPORT_MAPPINGS.items():
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Updated: {filepath}")
        return True
    return False

def main(directory):
    """Migrate all Python files in directory."""
    path = Path(directory)
    updated = 0
    
    for filepath in path.rglob('*.py'):
        if migrate_file(filepath):
            updated += 1
    
    print(f"\n✅ Migration complete! Updated {updated} files.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python migrate_imports.py <directory>")
        sys.exit(1)
    
    main(sys.argv[1])
```

## Backward Compatibility

If you need to maintain backward compatibility temporarily:

```python
try:
    # Try new imports first
    from dataknobs_structures import Tree
except ImportError:
    # Fall back to old imports
    from dataknobs.structures.tree import Tree
```

## Using the Legacy Package

The `dataknobs` package (version 0.0.15) provides compatibility wrappers:

```python
# These still work but show deprecation warnings
from dataknobs.structures import tree
from dataknobs.utils import json_utils

# Access via module attributes
tree_obj = tree.Tree("root")
value = json_utils.get_value(data, "key")
```

⚠️ **Note**: The legacy package shows deprecation warnings. Plan to migrate fully.

## Testing Your Migration

After migration, run your test suite:

```bash
# Run your tests
pytest

# Check for any import errors
python -c "from dataknobs_structures import Tree; print('✓ Structures OK')"
python -c "from dataknobs_utils import json_utils; print('✓ Utils OK')"
python -c "from dataknobs_xization import normalize; print('✓ Xization OK')"
```

## Getting Help

- **GitHub Issues**: [Report migration issues](https://github.com/yourusername/dataknobs/issues)
- **Documentation**: See package-specific guides in [Packages](packages/index.md)
- **Examples**: Check [Examples](examples/index.md) for updated code samples

## Timeline

- **Current**: New modular packages available
- **6 months**: Legacy package deprecated but supported
- **12 months**: Legacy package support ends

Plan your migration accordingly!