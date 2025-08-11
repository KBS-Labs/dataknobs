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

### Step 1: Identify Your Dependencies

Check which parts of dataknobs you're using:

```python
# Run this to see your imports
import ast
import os

def find_dataknobs_imports(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom):
                                if node.module and node.module.startswith('dataknobs'):
                                    imports.add(node.module)
                    except:
                        pass
    return sorted(imports)

# Usage
imports = find_dataknobs_imports('your_project_directory')
for imp in imports:
    print(imp)
```

### Step 2: Install New Packages

Based on your imports, install the appropriate packages:

```bash
# If you use dataknobs.structures.*
pip install dataknobs-structures

# If you use dataknobs.utils.*
pip install dataknobs-utils

# If you use dataknobs.xization.*
pip install dataknobs-xization

# Or install all at once
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Step 3: Update Requirements Files

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

For Poetry users:
```toml
# Old pyproject.toml
[tool.poetry.dependencies]
dataknobs = "^0.0.14"

# New pyproject.toml
[tool.poetry.dependencies]
dataknobs-structures = "^1.0.0"
dataknobs-utils = "^1.0.0"
dataknobs-xization = "^1.0.0"
```

### Step 4: Update Imports

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

### Step 5: Test Your Application

After migration, run your test suite to ensure everything works correctly.

## Import Mapping Reference

### Complete Mapping Table

#### Structures
- `dataknobs.structures.tree` → `dataknobs_structures.tree`
- `dataknobs.structures.document` → `dataknobs_structures.document`
- `dataknobs.structures.record_store` → `dataknobs_structures.record_store`
- `dataknobs.structures.conditional_dict` → `dataknobs_structures.conditional_dict`

#### Utils
- `dataknobs.utils.json_utils` → `dataknobs_utils.json_utils`
- `dataknobs.utils.file_utils` → `dataknobs_utils.file_utils`
- `dataknobs.utils.pandas_utils` → `dataknobs_utils.pandas_utils`
- `dataknobs.utils.xml_utils` → `dataknobs_utils.xml_utils`
- `dataknobs.utils.requests_utils` → `dataknobs_utils.requests_utils`
- `dataknobs.utils.elasticsearch_utils` → `dataknobs_utils.elasticsearch_utils`
- `dataknobs.utils.sql_utils` → `dataknobs_utils.sql_utils`
- `dataknobs.utils.llm_utils` → `dataknobs_utils.llm_utils`
- `dataknobs.utils.emoji_utils` → `dataknobs_utils.emoji_utils`
- `dataknobs.utils.json_extractor` → `dataknobs_utils.json_extractor`
- `dataknobs.utils.resource_utils` → `dataknobs_utils.resource_utils`
- `dataknobs.utils.stats_utils` → `dataknobs_utils.stats_utils`
- `dataknobs.utils.subprocess_utils` → `dataknobs_utils.subprocess_utils`
- `dataknobs.utils.sys_utils` → `dataknobs_utils.sys_utils`

#### Xization
- `dataknobs.xization.normalize` → `dataknobs_xization.normalize`
- `dataknobs.xization.masking_tokenizer` → `dataknobs_xization.masking_tokenizer`
- `dataknobs.xization.annotations` → `dataknobs_xization.annotations`
- `dataknobs.xization.authorities` → `dataknobs_xization.authorities`
- `dataknobs.xization.lexicon` → `dataknobs_xization.lexicon`

### Common Import Examples

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

Here's a comprehensive script to help automate the migration:

```python
#!/usr/bin/env python3
import os
import re
import sys

def migrate_imports(file_path):
    """Update imports in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update imports
    replacements = [
        (r'from dataknobs\.structures', 'from dataknobs_structures'),
        (r'import dataknobs\.structures', 'import dataknobs_structures'),
        (r'from dataknobs\.utils', 'from dataknobs_utils'),
        (r'import dataknobs\.utils', 'import dataknobs_utils'),
        (r'from dataknobs\.xization', 'from dataknobs_xization'),
        (r'import dataknobs\.xization', 'import dataknobs_xization'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            modified = True
            content = new_content
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")
    
    return modified

def migrate_directory(directory):
    """Migrate all Python files in a directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and other common directories
        dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git'}]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if migrate_imports(file_path):
                    count += 1
    
    print(f"\nMigration complete! Updated {count} files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_dataknobs.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    migrate_directory(directory)
```

Save this as `migrate_dataknobs.py` and run:
```bash
python migrate_dataknobs.py /path/to/your/project
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

## Common Issues and Solutions

### Issue: ImportError after migration
**Solution**: Make sure you've installed all required packages. The modular packages have their own dependencies.

### Issue: Deprecation warnings still showing
**Solution**: You might still be importing from the legacy package. Double-check all imports are updated.

### Issue: Type hints not working
**Solution**: Your IDE might need to re-index. Restart your IDE after installing the new packages.

## Getting Help

- **GitHub Issues**: [Report migration issues](https://github.com/kbs-labs/dataknobs/issues)
- **Documentation**: See package-specific guides in [Packages](packages/index.md)
- **Examples**: Check [Examples](examples/index.md) for updated code samples
- **README Files**: Review the test files for usage examples

## Timeline

- **Current**: New modular packages available
- **6 months**: Legacy package deprecated but supported
- **12 months**: Legacy package support ends

Plan your migration accordingly!