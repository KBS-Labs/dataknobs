# Migration Guide

This guide provides detailed instructions for migrating from the legacy `dataknobs` package to the modular packages.

## Overview

The migration transforms a single monolithic package into focused, modular packages:

```
dataknobs (legacy) → dataknobs-structures + dataknobs-utils + dataknobs-xization + dataknobs-common
```

## Why Migrate?

### Benefits

- **Modular Dependencies**: Install only what you need
- **Better Maintenance**: Smaller, focused packages
- **Improved Performance**: Reduced import overhead
- **Future Support**: Legacy package will be deprecated

### Timeline

- **Now - 6 months**: Full support for both legacy and modular packages
- **6-12 months**: Security fixes only for legacy package
- **12+ months**: Legacy package discontinued

## Pre-Migration Assessment

### Check Current Usage

Audit your codebase to identify what you're using:

```bash
# Find all dataknobs imports
grep -r "from dataknobs" your_project/
grep -r "import dataknobs" your_project/

# Common patterns to look for:
# - from dataknobs.structures import ...
# - from dataknobs.utils import ...
# - from dataknobs.xization import ...
# - from dataknobs import ...
```

### Identify Required Packages

Based on your usage, determine which modular packages you need:

| If you use... | Install... |
|--------------|-----------|
| `dataknobs.structures` | `dataknobs-structures` |
| `dataknobs.utils` | `dataknobs-utils` |
| `dataknobs.xization` | `dataknobs-xization` |
| Flask API | Consider rebuilding with modern framework |

## Step-by-Step Migration

### Step 1: Install Modular Packages

```bash
# Install the packages you need
pip install dataknobs-structures  # For Tree, Text, RecordStore, cdict
pip install dataknobs-utils       # For JSON, file, Elasticsearch utils
pip install dataknobs-xization    # For tokenization, normalization

# dataknobs-common is automatically installed as a dependency
```

### Step 2: Update Import Statements

#### Structures Package

```python
# OLD (Legacy)
from dataknobs.structures import Tree
from dataknobs.structures import Text, TextMetaData
from dataknobs.structures import RecordStore
from dataknobs.structures import cdict

# NEW (Modular)
from dataknobs_structures import Tree
from dataknobs_structures import Text, TextMetaData
from dataknobs_structures import RecordStore
from dataknobs_structures import cdict

# Or import all at once
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict
```

#### Utils Package

```python
# OLD (Legacy)
from dataknobs.utils import json_utils
from dataknobs.utils import file_utils
from dataknobs.utils import elasticsearch_utils
from dataknobs.utils import llm_utils

# NEW (Modular)
from dataknobs_utils import json_utils
from dataknobs_utils import file_utils
from dataknobs_utils import elasticsearch_utils
from dataknobs_utils import llm_utils
```

#### Xization Package

```python
# OLD (Legacy)
from dataknobs.xization import normalize
from dataknobs.xization import masking_tokenizer

# NEW (Modular)  
from dataknobs_xization import normalize
from dataknobs_xization import masking_tokenizer
# Or more specific imports
from dataknobs_xization.masking_tokenizer import TextFeatures, CharacterFeatures
```

### Step 3: Update Dependencies

#### requirements.txt

```txt
# OLD
dataknobs>=0.0.15

# NEW
dataknobs-structures>=1.0.0
dataknobs-utils>=1.0.0
dataknobs-xization>=1.0.0
```

#### pyproject.toml (Poetry)

```toml
# OLD
[tool.poetry.dependencies]
dataknobs = "^0.0.15"

# NEW  
[tool.poetry.dependencies]
dataknobs-structures = "^1.0.0"
dataknobs-utils = "^1.0.0"
dataknobs-xization = "^1.0.0"
```

#### setup.py

```python
# OLD
install_requires=[
    'dataknobs>=0.0.15',
]

# NEW
install_requires=[
    'dataknobs-structures>=1.0.0',
    'dataknobs-utils>=1.0.0', 
    'dataknobs-xization>=1.0.0',
]
```

### Step 4: Test Migration

Run your tests to ensure everything works:

```bash
# Run your test suite
python -m pytest tests/

# Or other test runners
python -m unittest discover
tox
```

### Step 5: Update Documentation

Update any documentation that references the legacy package:

- README files
- API documentation  
- Code comments
- User guides

### Step 6: Remove Legacy Package

After confirming everything works:

```bash
pip uninstall dataknobs
```

## Migration Examples

### Example 1: Simple Script

#### Before (Legacy)

```python
# migrate_example.py
from dataknobs.structures import Tree, Text, TextMetaData
from dataknobs.utils import json_utils
from dataknobs.xization import normalize

def process_document(content, doc_id):
    # Normalize content
    normalized = normalize.basic_normalization_fn(
        content, 
        expand_camelcase=True,
        lowercase=True
    )
    
    # Create document
    metadata = TextMetaData(text_id=doc_id)
    doc = Text(normalized, metadata)
    
    # Create tree structure
    tree = Tree("documents")
    tree.add_child(doc)
    
    return tree

# Usage
result = process_document("CamelCaseText", "doc_001")
```

#### After (Modular)

```python
# migrate_example.py  
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

def process_document(content, doc_id):
    # Normalize content - same code!
    normalized = normalize.basic_normalization_fn(
        content,
        expand_camelcase=True,
        lowercase=True
    )
    
    # Create document - same code!
    metadata = TextMetaData(text_id=doc_id)
    doc = Text(normalized, metadata)
    
    # Create tree structure - same code!
    tree = Tree("documents")
    tree.add_child(doc)
    
    return tree

# Usage - same code!
result = process_document("CamelCaseText", "doc_001")
```

### Example 2: Complex Application

#### Before (Legacy)

```python
# app.py
from dataknobs.structures import RecordStore, Tree
from dataknobs.utils import json_utils, elasticsearch_utils
from dataknobs.xization import normalize, masking_tokenizer

class DocumentProcessor:
    def __init__(self, config_path):
        self.store = RecordStore("documents.tsv")
        self.config = json_utils.get_value(
            json.load(open(config_path)), 
            "processor.settings"
        )
    
    def process_text(self, text):
        # Tokenize
        features = masking_tokenizer.TextFeatures(text, split_camelcase=True)
        tokens = features.get_tokens()
        
        # Normalize
        normalized = normalize.basic_normalization_fn(text)
        
        return tokens, normalized
```

#### After (Modular)

```python
# app.py
import json
from dataknobs_structures import RecordStore, Tree
from dataknobs_utils import json_utils, elasticsearch_utils  
from dataknobs_xization import normalize, masking_tokenizer

class DocumentProcessor:
    def __init__(self, config_path):
        self.store = RecordStore("documents.tsv")
        self.config = json_utils.get_value(
            json.load(open(config_path)),
            "processor.settings"
        )
    
    def process_text(self, text):
        # Tokenize - same code!
        features = masking_tokenizer.TextFeatures(text, split_camelcase=True)
        tokens = features.get_tokens()
        
        # Normalize - same code!
        normalized = normalize.basic_normalization_fn(text)
        
        return tokens, normalized
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError

```python
# Error
ModuleNotFoundError: No module named 'dataknobs.structures'

# Solution: Update import
# OLD
from dataknobs.structures import Tree
# NEW  
from dataknobs_structures import Tree
```

### Issue 2: Package Not Installed

```bash
# Error
ImportError: No module named 'dataknobs_structures'

# Solution: Install the package
pip install dataknobs-structures
```

### Issue 3: Version Conflicts

```bash
# Error
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed

# Solution: Clean install
pip uninstall dataknobs dataknobs-structures dataknobs-utils dataknobs-xization
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Issue 4: Circular Dependencies

If you have both legacy and modular packages installed, you might see conflicts:

```bash
# Solution: Uninstall legacy first
pip uninstall dataknobs
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

## Testing Migration

### Automated Testing

Create a test to verify both old and new imports work during transition:

```python
# test_migration.py
def test_tree_import():
    # Test new import (preferred)
    try:
        from dataknobs_structures import Tree
        new_tree = Tree("test")
        assert new_tree.data == "test"
        print("✓ New import works")
    except ImportError as e:
        print(f"✗ New import failed: {e}")
    
    # Test old import (fallback)
    try:
        from dataknobs.structures import Tree as LegacyTree  
        old_tree = LegacyTree("test")
        assert old_tree.data == "test"
        print("✓ Legacy import works")
    except ImportError as e:
        print(f"✗ Legacy import failed: {e}")

if __name__ == "__main__":
    test_tree_import()
```

### Manual Testing

Test key functionality:

```python
# Test structures
from dataknobs_structures import Tree, Text, TextMetaData, RecordStore, cdict

tree = Tree("root")
child = tree.add_child("child")
assert len(tree.children) == 1

# Test utils  
from dataknobs_utils import json_utils

data = {"key": "value"}
result = json_utils.get_value(data, "key")
assert result == "value"

# Test xization
from dataknobs_xization import normalize

text = "CamelCase"
normalized = normalize.expand_camelcase_fn(text)
assert normalized == "Camel Case"

print("✓ All packages working correctly")
```

## Rollback Plan

If you need to rollback the migration:

### 1. Reinstall Legacy Package

```bash
pip install dataknobs
```

### 2. Revert Import Changes

Use git or your version control system to revert import changes:

```bash
git checkout -- your_files.py
```

### 3. Revert Dependencies

```bash
# Revert requirements.txt or pyproject.toml
git checkout -- requirements.txt pyproject.toml
pip install -r requirements.txt
```

## Post-Migration Cleanup

### Remove Unused Dependencies

```bash
# Check for unused packages
pip-autoremove dataknobs -y

# Or use pip-check
pip install pip-check
pip-check
```

### Update CI/CD

Update your continuous integration configuration:

```yaml
# .github/workflows/test.yml (GitHub Actions)
- name: Install dependencies
  run: |
    pip install dataknobs-structures dataknobs-utils dataknobs-xization
    pip install -r requirements-dev.txt
```

### Update Docker

```dockerfile
# Dockerfile
RUN pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

## Migration Verification

### Checklist

- [ ] All imports updated
- [ ] Dependencies updated (requirements.txt, pyproject.toml, etc.)
- [ ] Tests passing
- [ ] Documentation updated  
- [ ] CI/CD updated
- [ ] Docker/deployment configs updated
- [ ] Legacy package uninstalled
- [ ] Team notified of changes

### Performance Check

Compare performance before and after migration:

```python
import time
from dataknobs_structures import Tree

# Time tree operations
start = time.time()
tree = Tree("root")
for i in range(1000):
    tree.add_child(f"child_{i}")
end = time.time()

print(f"Tree operations took {end - start:.4f} seconds")
```

## Getting Help

### Resources

1. **Documentation**: Comprehensive guides for each package
2. **Examples**: Working code samples in the repository
3. **Issues**: GitHub issues for bug reports and questions
4. **Community**: Discussion forums and chat

### Reporting Problems

If you encounter issues during migration:

1. Check this guide first
2. Search existing GitHub issues
3. Create a new issue with:
   - Python version
   - Package versions (old and new)
   - Error messages
   - Minimal reproduction code

### Support Timeline

- **Immediate**: Migration assistance available
- **6 months**: Continued support for both versions
- **12 months**: Legacy package deprecated
- **Ongoing**: Modular packages fully supported

## Best Practices

### Migration Strategy

1. **Test Environment First**: Migrate test/dev environments before production
2. **Gradual Migration**: Migrate one module/package at a time
3. **Backup Code**: Use version control and create backups
4. **Document Changes**: Keep track of what you've migrated
5. **Team Communication**: Inform team members about the migration

### Code Organization

After migration, consider organizing imports:

```python
# Group modular package imports together
from dataknobs_structures import Tree, Text, TextMetaData
from dataknobs_utils import json_utils, file_utils
from dataknobs_xization import normalize

# Separate from other third-party imports
import pandas as pd
import numpy as np
```

### Future-Proofing

- Pin package versions in production
- Set up dependency monitoring
- Subscribe to package release notifications
- Review upgrade guides for new versions

## See Also

- [Legacy Package Documentation](index.md)
- [Package Overview](../index.md)
- [Getting Started Guide](../../getting-started.md)