# Dataknobs Common

The `dataknobs-common` package provides shared utilities and base classes used across all Dataknobs packages.

## Installation

```bash
pip install dataknobs-common
```

Note: This package is automatically installed as a dependency of other Dataknobs packages.

## Overview

The Common package serves as the foundation for all other Dataknobs packages, providing:

- **Base Classes**: Common interfaces and abstract classes
- **Shared Utilities**: Helper functions used across packages
- **Common Constants**: Shared constants and configuration
- **Version Management**: Centralized version handling

## Package Structure

The Common package is designed to be lightweight and focused on shared functionality:

```
dataknobs-common/
├── src/
│   └── dataknobs_common/
│       └── __init__.py  # Version and shared imports
└── tests/
    └── test_version.py  # Version testing
```

## Version Information

```python
from dataknobs_common import __version__

print(f"Dataknobs Common version: {__version__}")  # 1.0.0
```

## Design Principles

The Common package follows these design principles:

1. **Minimal Dependencies**: Keep external dependencies to a minimum
2. **Backward Compatibility**: Maintain stability for dependent packages
3. **Shared Standards**: Provide consistent interfaces across packages
4. **Version Consistency**: Centralized version management

## Usage with Other Packages

The Common package is primarily used internally by other Dataknobs packages:

### In Structures Package

```python
# Internal usage in dataknobs-structures
from dataknobs_common import __version__
```

### In Utils Package

```python
# Internal usage in dataknobs-utils
from dataknobs_common import __version__
```

### In Xization Package

```python
# Internal usage in dataknobs-xization
from dataknobs_common import __version__
```

## Development

When developing with Dataknobs packages, you typically won't need to import from `dataknobs-common` directly. Instead, use the higher-level packages that depend on it.

### For Package Developers

If you're extending or contributing to Dataknobs packages:

```python
# Check version compatibility
from dataknobs_common import __version__ as common_version

print(f"Using dataknobs-common version: {common_version}")
```

## Integration Examples

### Version Checking

```python
from dataknobs_common import __version__ as common_version
from dataknobs_structures import __version__ as structures_version
from dataknobs_utils import __version__ as utils_version

print(f"Common: {common_version}")
print(f"Structures: {structures_version}")
print(f"Utils: {utils_version}")
```

## API Reference

For complete API documentation, see the [Common API Reference](api.md).

## Package Dependencies

The Common package has minimal dependencies and serves as the foundation for:

- **dataknobs-structures**: Core data structures
- **dataknobs-utils**: Utility functions and helpers
- **dataknobs-xization**: Text processing and normalization

## Best Practices

1. **Don't Import Directly**: Use higher-level packages instead of importing from Common directly
2. **Version Awareness**: Check version compatibility when using multiple packages
3. **Dependency Management**: Keep Common package updated for security and stability

## Contributing

When contributing to the Common package:

1. **Maintain Backward Compatibility**: Changes must not break existing functionality
2. **Minimal Changes**: Only add truly shared functionality
3. **Documentation**: Update this documentation for any new features
4. **Testing**: Ensure all dependent packages continue to work

## Support

For issues related to the Common package:

1. Check if the issue is specific to Common or affects other packages
2. Provide version information for all installed Dataknobs packages
3. Include minimal reproduction code if applicable

## Next Steps

- Explore [Structures Package](../structures/index.md)
- Learn about [Utils Package](../utils/index.md)
- See [Xization Package](../xization/index.md)