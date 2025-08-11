# dataknobs-common API Reference

Complete API documentation for the `dataknobs_common` package.

## Package Information

- **Package Name**: `dataknobs_common`
- **Version**: 1.0.0
- **Description**: Common components and utilities shared across dataknobs packages
- **Python Requirements**: >=3.8

## Installation

```bash
pip install dataknobs-common
```

## Import Statement

```python
from dataknobs_common import (
    # Common utilities and base classes will be imported here
)
```

## Module Documentation

### Core Components

::: dataknobs_common
    options:
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false

## Usage Examples

### Basic Usage

```python
from dataknobs_common import common_utilities

# Example usage of common components
# (Specific examples will depend on actual implementation)
```

## Package Structure

The `dataknobs_common` package provides shared functionality used across the dataknobs ecosystem:

### Base Classes
- Common abstract base classes
- Shared interface definitions
- Standard exception classes

### Utilities
- Configuration management
- Logging utilities
- Validation functions
- Type definitions

### Constants
- Package-wide constants
- Default configuration values
- Standard error messages

## API Overview

### Configuration Management

```python
from dataknobs_common import config

# Configuration utilities for managing package settings
config_manager = config.ConfigManager()
settings = config_manager.load_settings("config.json")
```

### Logging

```python
from dataknobs_common import logging

# Standardized logging across dataknobs packages
logger = logging.get_logger(__name__)
logger.info("Processing started")
```

### Validation

```python
from dataknobs_common import validation

# Common validation functions
validation.validate_text_input(text)
validation.validate_file_path(path)
```

### Type Definitions

```python
from dataknobs_common.types import (
    TextData,
    DocumentMetadata,
    ProcessingConfig
)

# Use common type definitions across packages
def process_document(doc: TextData) -> ProcessingConfig:
    pass
```

## Integration Examples

### With dataknobs-structures

```python
from dataknobs_common import base_classes
from dataknobs_structures import Tree

class CustomTree(base_classes.BaseDataStructure, Tree):
    """Custom tree with common functionality."""
    
    def validate(self) -> bool:
        """Use common validation methods."""
        return super().validate() and self._validate_tree_structure()
```

### With dataknobs-utils

```python
from dataknobs_common import config, logging
from dataknobs_utils import file_utils

logger = logging.get_logger(__name__)
config_manager = config.ConfigManager()

def process_files(input_dir: str):
    """Process files using common configuration and logging."""
    settings = config_manager.get_settings("file_processing")
    
    logger.info(f"Starting file processing in {input_dir}")
    
    for filepath in file_utils.filepath_generator(input_dir):
        logger.debug(f"Processing {filepath}")
        # Process file with common error handling
        try:
            process_file(filepath, settings)
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
```

### With dataknobs-xization

```python
from dataknobs_common import validation, exceptions
from dataknobs_xization import normalize

def safe_normalize_text(text: str) -> str:
    """Normalize text with common validation and error handling."""
    try:
        # Use common validation
        validation.validate_text_input(text)
        
        # Perform normalization
        normalized = normalize.basic_normalization_fn(text)
        
        # Validate result
        validation.validate_text_output(normalized)
        
        return normalized
        
    except validation.ValidationError as e:
        raise exceptions.DataProcessingError(f"Text validation failed: {e}")
    except Exception as e:
        raise exceptions.DataProcessingError(f"Normalization failed: {e}")
```

## Common Patterns

### Error Handling

```python
from dataknobs_common import exceptions, logging

logger = logging.get_logger(__name__)

try:
    # Process data
    result = process_data(data)
except exceptions.DataValidationError as e:
    logger.error(f"Validation error: {e}")
    raise
except exceptions.DataProcessingError as e:
    logger.error(f"Processing error: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise exceptions.DataProcessingError(f"Unexpected error: {e}")
```

### Configuration Management

```python
from dataknobs_common import config

# Global configuration manager
config_manager = config.ConfigManager()

# Load package-specific settings
utils_config = config_manager.get_package_config("dataknobs_utils")
structures_config = config_manager.get_package_config("dataknobs_structures")

# Override with local settings
local_config = config_manager.load_local_config("local_settings.json")
final_config = config_manager.merge_configs(utils_config, local_config)
```

### Logging Standards

```python
from dataknobs_common import logging

# Package-level logger
logger = logging.get_logger("dataknobs.mypackage")

def process_data(data):
    """Process data with standardized logging."""
    logger.info("Starting data processing")
    
    try:
        logger.debug(f"Processing {len(data)} items")
        
        for i, item in enumerate(data):
            logger.debug(f"Processing item {i}: {item}")
            result = process_item(item)
            
        logger.info(f"Successfully processed {len(data)} items")
        return results
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        raise
```

## Testing Utilities

```python
from dataknobs_common import testing

class TestMyDataProcessing(testing.BaseTestCase):
    """Test case using common testing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.test_data = testing.create_test_data()
        self.config = testing.get_test_config()
    
    def test_data_validation(self):
        """Test data validation using common patterns."""
        with testing.assert_raises_validation_error():
            validate_invalid_data(self.test_data)
    
    def test_processing_pipeline(self):
        """Test processing pipeline."""
        result = process_data(self.test_data, self.config)
        testing.assert_valid_result(result)
        
    def tearDown(self):
        """Clean up test environment."""
        testing.cleanup_test_data()
        super().tearDown()
```

## Performance Utilities

```python
from dataknobs_common import performance

@performance.monitor_performance
def expensive_operation(data):
    """Operation with performance monitoring."""
    # Expensive processing here
    return processed_data

# Context manager for performance monitoring
with performance.PerformanceMonitor("data_processing") as monitor:
    result = process_large_dataset(data)
    monitor.log_memory_usage()
    monitor.log_timing_stats()
```

## Security Utilities

```python
from dataknobs_common import security

# Input sanitization
sanitized_input = security.sanitize_text_input(user_input)

# Data masking for logging
masked_data = security.mask_sensitive_data(data, patterns=[
    security.EMAIL_PATTERN,
    security.PHONE_PATTERN,
    security.SSN_PATTERN
])

logger.info(f"Processing data: {masked_data}")
```

## Version Compatibility

```python
from dataknobs_common import version

# Check version compatibility
if version.is_compatible("dataknobs_utils", "1.2.0"):
    # Use new features
    pass
else:
    # Fallback to older API
    pass

# Get version information
version_info = version.get_package_versions()
print(f"Installed versions: {version_info}")
```

## Migration Utilities

```python
from dataknobs_common import migration

# Data format migration
old_data = load_old_format_data("legacy_data.json")
new_data = migration.migrate_data_format(
    old_data, 
    from_version="0.9", 
    to_version="1.0"
)

# Configuration migration
old_config = load_old_config("old_config.yaml")
new_config = migration.migrate_configuration(
    old_config,
    migration_rules=migration.get_migration_rules("0.9", "1.0")
)
```

## Error Handling Standards

```python
from dataknobs_common.exceptions import (
    DataknobsError,
    DataValidationError,
    DataProcessingError,
    ConfigurationError,
    CompatibilityError
)

def robust_data_processing(data, config):
    """Example of standardized error handling."""
    try:
        # Validate input
        if not validation.is_valid_data(data):
            raise DataValidationError("Invalid input data format")
        
        # Validate configuration
        if not validation.is_valid_config(config):
            raise ConfigurationError("Invalid configuration")
        
        # Process data
        result = process_data(data, config)
        
        # Validate output
        if not validation.is_valid_result(result):
            raise DataProcessingError("Processing produced invalid result")
        
        return result
        
    except DataValidationError:
        logger.error("Data validation failed")
        raise
    except ConfigurationError:
        logger.error("Configuration error")
        raise
    except DataProcessingError:
        logger.error("Data processing failed")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise DataknobsError(f"Unexpected error in data processing: {e}")
```

## Best Practices

### 1. Use Common Base Classes
```python
from dataknobs_common.base import BaseProcessor

class MyProcessor(BaseProcessor):
    """Custom processor inheriting common functionality."""
    
    def process(self, data):
        """Process data using inherited validation and logging."""
        self.validate_input(data)
        result = self._do_processing(data)
        self.validate_output(result)
        return result
```

### 2. Standard Configuration Patterns
```python
from dataknobs_common import config

class MyComponent:
    def __init__(self, config_path=None):
        self.config = config.load_config(
            config_path or config.get_default_config_path()
        )
        self.logger = logging.get_logger(self.__class__.__module__)
```

### 3. Consistent Error Handling
```python
from dataknobs_common.decorators import handle_common_errors

@handle_common_errors
def process_document(doc):
    """Process document with standard error handling."""
    # Implementation here - common errors will be caught and handled
    return processed_doc
```

## Testing

```python
import pytest
from dataknobs_common import testing, validation, exceptions

class TestCommonFunctionality:
    """Test common functionality."""
    
    def test_validation(self):
        """Test validation functions."""
        # Test valid data
        valid_data = testing.create_valid_test_data()
        assert validation.is_valid_data(valid_data)
        
        # Test invalid data
        invalid_data = testing.create_invalid_test_data()
        with pytest.raises(exceptions.DataValidationError):
            validation.validate_data(invalid_data)
    
    def test_configuration(self):
        """Test configuration management."""
        from dataknobs_common import config
        
        # Test loading configuration
        test_config = testing.create_test_config()
        config_manager = config.ConfigManager()
        loaded_config = config_manager.load_config_from_dict(test_config)
        
        assert loaded_config == test_config
    
    def test_error_handling(self):
        """Test error handling patterns."""
        from dataknobs_common.exceptions import DataProcessingError
        
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(DataProcessingError):
            try:
                failing_function()
            except ValueError as e:
                raise DataProcessingError(f"Processing failed: {e}")
```

## Performance Considerations

- Common utilities are optimized for performance across all packages
- Configuration loading is cached to avoid repeated file I/O
- Logging is configured for minimal performance impact
- Validation functions use efficient algorithms

## Dependencies

Core dependencies for dataknobs_common:

```txt
# Minimal dependencies to avoid conflicts
python>=3.8
typing-extensions>=4.0.0  # For older Python versions
```

## Contributing

For contributing to dataknobs_common:

1. Fork the repository
2. Create feature branch for common functionality
3. Ensure changes don't break other packages
4. Add comprehensive tests
5. Update documentation for all affected packages
6. Submit pull request

See [Contributing Guide](../development/contributing.md) for detailed information.

## Changelog

### Version 1.0.0
- Initial release
- Base classes and interfaces
- Common configuration management
- Standardized logging
- Error handling framework
- Validation utilities
- Testing support

## License

See [License](../license.md) for license information.