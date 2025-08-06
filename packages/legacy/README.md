# dataknobs (Legacy Package)

⚠️ **DEPRECATED**: This package is maintained for backward compatibility only. Please use the modular packages instead.

## Migration Notice

The `dataknobs` package has been split into modular packages for better maintainability and flexibility:

- **dataknobs-structures**: Data structures for AI knowledge bases
- **dataknobs-utils**: Utility functions
- **dataknobs-xization**: Text normalization and tokenization
- **dataknobs-common**: Shared base functionality

## Installation

For backward compatibility:
```bash
pip install dataknobs
```

For new projects, install only what you need:
```bash
# Install specific packages
pip install dataknobs-structures
pip install dataknobs-utils
pip install dataknobs-xization

# Or install all
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

## Migration Guide

### Update Imports

The legacy package maintains the same import structure, but shows deprecation warnings. To migrate:

```python
# Old (deprecated)
from dataknobs.structures.tree import Tree
from dataknobs.utils.json_utils import get_value

# New (recommended)
from dataknobs_structures.tree import Tree
from dataknobs_utils.json_utils import get_value
```

### Package Mapping

- `dataknobs.structures.*` → `dataknobs_structures.*`
- `dataknobs.utils.*` → `dataknobs_utils.*`
- `dataknobs.xization.*` → `dataknobs_xization.*`

## Deprecation Timeline

- **Current**: Deprecation warnings are shown when using this package
- **Future v2.0.0**: This legacy package will be removed

Please migrate to the modular packages as soon as possible.

## License

See LICENSE file in the root repository.
