# Common API Reference

This document provides the complete API reference for the `dataknobs-common` package.

## Package Information

### Version

```python
from dataknobs_common import __version__
```

The version string for the dataknobs-common package.

**Type**: `str`

**Example**:
```python
from dataknobs_common import __version__

print(__version__)  # "1.0.0"
```

## Module Structure

### dataknobs_common

The main module provides version information and serves as the foundation for other packages.

```python
import dataknobs_common
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `__version__` | `str` | Package version string |

## Version Management

The Common package provides centralized version management for all Dataknobs packages.

### Version Format

The version follows [Semantic Versioning](https://semver.org/) format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Usage Example

```python
from dataknobs_common import __version__
import pkg_resources

# Get version information
common_version = __version__
print(f"dataknobs-common version: {common_version}")

# Compare versions programmatically
from packaging import version

current_version = version.parse(__version__)
minimum_required = version.parse("1.0.0")

if current_version >= minimum_required:
    print("Version requirement met")
else:
    print("Please update dataknobs-common")
```

## Integration Points

### For Package Developers

When developing packages that depend on dataknobs-common:

```python
# In your package's __init__.py
from dataknobs_common import __version__ as _common_version

# Version compatibility checking
def check_common_compatibility():
    from packaging import version
    min_version = version.parse("1.0.0")
    current = version.parse(_common_version)
    
    if current < min_version:
        raise ImportError(
            f"dataknobs-common {min_version} or higher required, "
            f"but {current} is installed"
        )
```

### For Application Developers

When using multiple Dataknobs packages:

```python
def print_package_versions():
    """Print versions of all installed Dataknobs packages"""
    packages = [
        'dataknobs-common',
        'dataknobs-structures', 
        'dataknobs-utils',
        'dataknobs-xization'
    ]
    
    for package in packages:
        try:
            import importlib
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: Not installed")
```

## Error Handling

### Import Errors

If the Common package is not properly installed:

```python
try:
    from dataknobs_common import __version__
    print(f"Common package version: {__version__}")
except ImportError as e:
    print(f"Error importing dataknobs-common: {e}")
    print("Please install with: pip install dataknobs-common")
```

### Version Compatibility

```python
def ensure_compatibility():
    try:
        from dataknobs_common import __version__
        from packaging import version
        
        current = version.parse(__version__)
        min_required = version.parse("1.0.0")
        
        if current < min_required:
            raise RuntimeError(
                f"dataknobs-common version {min_required} or higher required"
            )
            
        return True
    except ImportError:
        raise RuntimeError("dataknobs-common not installed")
```

## Testing

### Version Testing

```python
import unittest
from dataknobs_common import __version__

class TestVersion(unittest.TestCase):
    def test_version_format(self):
        """Test that version follows semantic versioning format"""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        self.assertRegex(__version__, pattern)
    
    def test_version_not_empty(self):
        """Test that version is not empty"""
        self.assertTrue(__version__)
        self.assertIsInstance(__version__, str)
```

## Dependencies

The Common package has minimal dependencies to ensure stability:

- **Python**: >= 3.8
- **No external dependencies**: Keeps the package lightweight

## Best Practices

### Version Checking

Always check version compatibility in production code:

```python
def check_dataknobs_versions():
    """Check all Dataknobs package versions for compatibility"""
    try:
        from dataknobs_common import __version__ as common_ver
        
        # Add other package checks as needed
        print(f"Common: {common_ver}")
        
        # Perform compatibility checks
        return True
        
    except ImportError as e:
        print(f"Package import error: {e}")
        return False
```

### Error Messages

Provide clear error messages when version issues occur:

```python
def validate_installation():
    try:
        from dataknobs_common import __version__
    except ImportError:
        raise ImportError(
            "dataknobs-common is required but not installed. "
            "Install with: pip install dataknobs-common"
        )
```

## Changelog

### Version 1.0.0
- Initial release
- Basic version management
- Foundation for other packages

## Support

For issues related to the Common package API:

1. Verify package installation: `pip list | grep dataknobs-common`
2. Check version compatibility with other packages
3. Report issues with full environment information