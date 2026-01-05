---
globs:
  - "**/*.py"
---

# Dataknobs-First Development

## Before Implementing New Utilities

When you need common functionality, follow this order:

### 1. Check Dataknobs First

Search for existing implementations:

```bash
# Search for similar functionality
uv run grep -r "class.*Store" packages/*/src/
uv run grep -r "def.*config" packages/*/src/

# Check specific packages
ls packages/common/src/dataknobs_common/
ls packages/utils/src/dataknobs_utils/
ls packages/data/src/dataknobs_data/
```

### 2. Common Utilities Available

**Data Storage:**
- `dataknobs_data.stores.InMemoryStore` - In-memory key-value
- `dataknobs_data.stores.FileStore` - File-based persistence
- `dataknobs_data.stores.PostgresStore` - PostgreSQL backend

**Configuration:**
- `dataknobs_config.ConfigLoader` - Multi-format config loading
- `dataknobs_config.EnvConfig` - Environment variable config

**Logging:**
- `dataknobs_common.logging` - Standardized logging setup

**Validation:**
- `dataknobs_data.validation` - Data validation framework

### 3. If Utility Doesn't Exist

Ask these questions:
1. Is this generally reusable across projects?
2. Does it fit an existing dataknobs package?
3. Should it be a new dataknobs utility?

**If YES to reusability:**
- Add to appropriate dataknobs package
- Include tests
- Update both documentation locations

**If project-specific:**
- Implement in the current project
- Consider if patterns emerge for future extraction

### 4. Using Dataknobs in Tests

Prefer real dataknobs utilities over mocks:

```python
# PREFER: Real in-memory store
from dataknobs_data.stores import InMemoryStore
store = InMemoryStore()

# AVOID: Mock store
store = MagicMock(spec=DataStore)
```

This catches integration issues early and validates real behavior.
