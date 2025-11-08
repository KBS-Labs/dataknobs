# Dataknobs Common

Common utilities and base classes shared across dataknobs packages.

> **💡 Quick Links:**
> - [Complete API Documentation](reference/common.md) - Full auto-generated reference
> - [Source Code](https://github.com/kbs-labs/dataknobs/tree/main/packages/common/src/dataknobs_common) - Browse on GitHub
> - [Package Guide](../packages/common/index.md) - Detailed documentation

## Overview

The `dataknobs-common` package provides shared functionality and common abstractions used across the dataknobs ecosystem. It establishes consistent patterns for configuration, logging, error handling, and validation across all dataknobs packages.

> **Note**: This documentation describes the conceptual architecture and intended usage patterns for common utilities. Some features described here represent planned functionality and design patterns that packages should follow.

**Package Information:**

- **Package Name**: `dataknobs_common`
- **Version**: 1.0.0
- **Python Requirements**: >=3.8
- **License**: See [project license](https://github.com/KBS-Labs/dataknobs/blob/main/LICENSE)

## Installation

```bash
pip install dataknobs-common
```

## Quick Start

The `dataknobs-common` package establishes patterns and conventions used across the dataknobs ecosystem. While the package itself is minimal, it defines the interfaces and standards that other packages follow.

```python
from dataknobs_common import __version__

print(f"dataknobs-common version: {__version__}")
```

> **Note**: The examples in this documentation demonstrate the conceptual patterns and conventions that dataknobs packages should follow. Some features represent planned functionality and architectural guidelines.

## Core Concepts

The `dataknobs_common` package establishes consistent patterns across the dataknobs ecosystem:

### Design Principles

1. **Consistency** - Standardized interfaces and behaviors across all packages
2. **Simplicity** - Minimal abstractions that don't obscure underlying functionality
3. **Flexibility** - Common patterns that adapt to different use cases
4. **Interoperability** - Seamless integration between dataknobs packages

### Package Architecture

**Base Classes:**

- Common abstract base classes for data structures
- Shared interface definitions
- Standard exception hierarchy

**Utilities:**

- Configuration management with environment variable support
- Structured logging with consistent formatting
- Input validation and sanitization
- Type definitions and protocols

**Constants & Defaults:**

- Package-wide constants for configuration
- Default values for common parameters
- Standard error messages and codes

## Common Patterns

These patterns demonstrate the conventions and standards used across dataknobs packages.

### Configuration Management Pattern

Standardized configuration loading and management:

```python
from dataknobs_common import config

def setup_component(config_path: str = None):
    """Set up component with configuration.

    Args:
        config_path: Optional path to configuration file.
            If not provided, uses default location.

    Returns:
        Configured component instance
    """
    # Load configuration with standard pattern
    config_manager = config.ConfigManager()
    settings = config_manager.load_settings(
        config_path or config.get_default_config_path()
    )

    return Component(settings)
```

### Logging Pattern

Consistent logging across all dataknobs packages:

```python
from dataknobs_common import logging

# Get logger for current module
logger = logging.get_logger(__name__)

def process_data(data):
    """Process data with structured logging.

    Args:
        data: Input data to process

    Returns:
        Processed result
    """
    logger.info("Processing started", extra={"data_size": len(data)})

    try:
        result = perform_processing(data)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        logger.error("Processing failed", exc_info=True)
        raise
```

### Validation Pattern

Common validation functions for input checking:

```python
from dataknobs_common import validation

def safe_text_operation(text: str) -> str:
    """Perform text operation with validation.

    Args:
        text: Input text to process

    Returns:
        Processed text

    Raises:
        ValidationError: If input validation fails
    """
    # Validate input
    validation.validate_text_input(text)

    # Process
    result = process(text)

    # Validate output
    validation.validate_text_output(result)

    return result
```

### Type Definitions Pattern

Common type definitions for cross-package consistency:

```python
from dataknobs_common.types import (
    TextData,
    DocumentMetadata,
    ProcessingConfig
)

def process_document(
    doc: TextData,
    metadata: DocumentMetadata,
    config: ProcessingConfig
) -> TextData:
    """Process document with type safety.

    Args:
        doc: Input document text
        metadata: Document metadata
        config: Processing configuration

    Returns:
        Processed document
    """
    # Type-safe processing with common types
    return perform_processing(doc, metadata, config)
```

## Integration Examples

These examples show how `dataknobs-common` provides shared functionality across different dataknobs packages.

> **Related Packages:**
> - [dataknobs-structures](https://github.com/KBS-Labs/dataknobs/tree/main/packages/structures) - Core data structures
> - [dataknobs-utils](https://github.com/KBS-Labs/dataknobs/tree/main/packages/utils) - Utility functions
> - [dataknobs-xization](https://github.com/KBS-Labs/dataknobs/tree/main/packages/xization) - Text normalization

### Integration with dataknobs-structures

Extend data structures with common validation and error handling:

```python
from dataknobs_common import base_classes
from dataknobs_structures import Tree

class CustomTree(base_classes.BaseDataStructure, Tree):
    """Custom tree with common functionality."""

    def validate(self) -> bool:
        """Use common validation methods."""
        return super().validate() and self._validate_tree_structure()
```

### Integration with dataknobs-utils

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

### Integration with dataknobs-xization

Use common validation and error handling with text normalization:

```python
from dataknobs_common import validation, exceptions
from dataknobs_xization import normalize

def safe_normalize_text(text: str) -> str:
    """Normalize text with common validation and error handling.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string

    Raises:
        DataProcessingError: If validation or normalization fails
    """
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

## Design Patterns

Common design patterns for building dataknobs components.

### Error Handling Pattern

Standardized error handling across packages:

```python
from dataknobs_common import exceptions, logging

logger = logging.get_logger(__name__)

def robust_operation(data):
    """Perform operation with standard error handling.

    Args:
        data: Input data to process

    Returns:
        Processed result

    Raises:
        DataValidationError: If input validation fails
        DataProcessingError: If processing fails
    """
    try:
        # Validate and process data
        validate_data(data)
        result = process_data(data)
        return result
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

### Multi-Package Configuration Pattern

Managing configuration across multiple packages:

```python
from dataknobs_common import config

class ApplicationConfig:
    """Application configuration manager."""

    def __init__(self):
        self.config_manager = config.ConfigManager()

    def load_all_configs(self):
        """Load configurations for all packages.

        Returns:
            Dictionary of package configurations
        """
        # Load package-specific settings
        utils_config = self.config_manager.get_package_config("dataknobs_utils")
        structures_config = self.config_manager.get_package_config("dataknobs_structures")

        # Override with local settings
        local_config = self.config_manager.load_local_config("local_settings.json")

        # Merge configurations
        return {
            "utils": self.config_manager.merge_configs(utils_config, local_config),
            "structures": self.config_manager.merge_configs(structures_config, local_config)
        }
```

### Hierarchical Logging Pattern

Package-level logging with consistent formatting:

```python
from dataknobs_common import logging

# Package-level logger
logger = logging.get_logger("dataknobs.mypackage")

def process_batch(items: list):
    """Process batch of items with detailed logging.

    Args:
        items: List of items to process

    Returns:
        List of processed results
    """
    logger.info("Starting batch processing", extra={"batch_size": len(items)})

    results = []
    try:
        for i, item in enumerate(items):
            logger.debug(f"Processing item {i+1}/{len(items)}")
            result = process_item(item)
            results.append(result)

        logger.info(f"Successfully processed {len(items)} items")
        return results

    except Exception as e:
        logger.error(
            f"Batch processing failed at item {i}",
            exc_info=True,
            extra={"failed_item_index": i}
        )
        raise
```

## Additional Utilities

These utilities demonstrate common patterns for specialized functionality.

### Testing Pattern

Standardized testing with common utilities:

```python
import pytest
from dataknobs_common import testing

class TestDataProcessing(testing.BaseTestCase):
    """Test case using common testing utilities."""

    def setUp(self):
        """Set up test environment with common fixtures."""
        super().setUp()
        self.test_data = testing.create_test_data()
        self.config = testing.get_test_config()

    def test_data_validation(self):
        """Test data validation using common patterns."""
        with pytest.raises(testing.ValidationError):
            validate_invalid_data(self.test_data)

    def test_processing_pipeline(self):
        """Test processing pipeline with validation."""
        result = process_data(self.test_data, self.config)
        testing.assert_valid_result(result)

    def tearDown(self):
        """Clean up test environment."""
        testing.cleanup_test_data()
        super().tearDown()
```

### Performance Monitoring Pattern

Monitor performance with decorators and context managers:

```python
from dataknobs_common import performance

@performance.monitor_performance
def expensive_operation(data):
    """Operation with automatic performance monitoring.

    Args:
        data: Input data to process

    Returns:
        Processed result
    """
    return process_large_dataset(data)

# Context manager for detailed monitoring
with performance.PerformanceMonitor("data_processing") as monitor:
    result = process_large_dataset(data)
    monitor.log_memory_usage()
    monitor.log_timing_stats()
```

### Security Pattern

Input sanitization and data masking:

```python
from dataknobs_common import security

def safe_user_input_processing(user_input: str):
    """Process user input safely.

    Args:
        user_input: Raw user input string

    Returns:
        Sanitized and processed result
    """
    # Input sanitization
    sanitized_input = security.sanitize_text_input(user_input)

    # Data masking for logging
    masked_data = security.mask_sensitive_data(
        sanitized_input,
        patterns=[
            security.EMAIL_PATTERN,
            security.PHONE_PATTERN,
            security.SSN_PATTERN
        ]
    )

    logger.info(f"Processing data: {masked_data}")

    return process(sanitized_input)
```

### Version Compatibility Pattern

Check version compatibility for feature detection:

```python
from dataknobs_common import version

def use_features_conditionally():
    """Use features based on package versions.

    Returns:
        Processing result using appropriate API version
    """
    # Check version compatibility
    if version.is_compatible("dataknobs_utils", "1.2.0"):
        # Use new features from 1.2.0+
        return use_new_api()
    else:
        # Fallback to older API
        return use_legacy_api()

    # Get version information for diagnostics
    version_info = version.get_package_versions()
    logger.debug(f"Installed versions: {version_info}")
```

### Migration Pattern

Data and configuration migration utilities:

```python
from dataknobs_common import migration

def migrate_legacy_data(legacy_data_path: str, output_path: str):
    """Migrate legacy data to current format.

    Args:
        legacy_data_path: Path to legacy data file
        output_path: Path for migrated data

    Returns:
        Migration success status
    """
    # Load and migrate data format
    old_data = load_old_format_data(legacy_data_path)
    new_data = migration.migrate_data_format(
        old_data,
        from_version="0.9",
        to_version="1.0"
    )

    # Save migrated data
    save_data(new_data, output_path)
    return True

def migrate_configuration(old_config_path: str):
    """Migrate configuration to new format.

    Args:
        old_config_path: Path to old configuration file

    Returns:
        Migrated configuration dictionary
    """
    old_config = load_old_config(old_config_path)
    migration_rules = migration.get_migration_rules("0.9", "1.0")

    return migration.migrate_configuration(old_config, migration_rules)
```

### Exception Hierarchy Pattern

Standardized exception handling with custom exceptions:

```python
from dataknobs_common.exceptions import (
    DataknobsError,
    DataValidationError,
    DataProcessingError,
    ConfigurationError,
    CompatibilityError
)

def robust_data_processing(data, config):
    """Process data with comprehensive error handling.

    Args:
        data: Input data to process
        config: Processing configuration

    Returns:
        Processed result

    Raises:
        DataValidationError: If input validation fails
        ConfigurationError: If configuration is invalid
        DataProcessingError: If processing fails
        DataknobsError: For unexpected errors
    """
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

Inherit from common base classes to ensure consistent behavior:

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

Use consistent configuration loading across components:

```python
from dataknobs_common import config, logging

class MyComponent:
    """Component with standard configuration."""

    def __init__(self, config_path=None):
        self.config = config.load_config(
            config_path or config.get_default_config_path()
        )
        self.logger = logging.get_logger(self.__class__.__module__)
```

### 3. Consistent Error Handling

Use common decorators and error handling patterns:

```python
from dataknobs_common.decorators import handle_common_errors

@handle_common_errors
def process_document(doc):
    """Process document with standard error handling.

    Common errors (ValidationError, ConfigurationError) will be
    caught and handled according to standard patterns.
    """
    # Implementation here
    return processed_doc
```

### 4. Structured Logging

Follow consistent logging patterns across all packages:

```python
from dataknobs_common import logging

logger = logging.get_logger(__name__)

def process_data(data):
    """Process data with structured logging."""
    logger.info("Processing started", extra={"data_size": len(data)})

    try:
        result = do_processing(data)
        logger.info("Processing completed", extra={"result_size": len(result)})
        return result
    except Exception as e:
        logger.error("Processing failed", exc_info=True, extra={"error": str(e)})
        raise
```

### 5. Type Hints and Validation

Use common type definitions and validation:

```python
from dataknobs_common.types import TextData, ProcessingConfig
from dataknobs_common import validation

def process_text(text: TextData, config: ProcessingConfig) -> TextData:
    """Process text with type safety and validation."""
    # Validate inputs
    validation.validate_text_input(text)
    validation.validate_config(config)

    # Process with type safety
    result = perform_processing(text, config)

    # Validate output
    validation.validate_text_output(result)
    return result
```

## Testing Common Functionality

Example tests demonstrating common functionality patterns:

```python
import pytest
from dataknobs_common import testing, validation, exceptions, config

class TestCommonFunctionality:
    """Test common functionality patterns."""

    def test_validation(self):
        """Test validation functions work correctly.

        Verifies that validation correctly accepts valid data
        and rejects invalid data with appropriate exceptions.
        """
        # Test valid data
        valid_data = testing.create_valid_test_data()
        assert validation.is_valid_data(valid_data)

        # Test invalid data
        invalid_data = testing.create_invalid_test_data()
        with pytest.raises(exceptions.DataValidationError):
            validation.validate_data(invalid_data)

    def test_configuration(self):
        """Test configuration management.

        Verifies that configuration can be loaded and
        managed correctly.
        """
        # Create and load test configuration
        test_config = testing.create_test_config()
        config_manager = config.ConfigManager()
        loaded_config = config_manager.load_config_from_dict(test_config)

        assert loaded_config == test_config

    def test_error_handling(self):
        """Test error handling patterns.

        Verifies that errors are properly wrapped and
        raised with appropriate exception types.
        """
        def failing_function():
            raise ValueError("Test error")

        # Test exception wrapping
        with pytest.raises(exceptions.DataProcessingError):
            try:
                failing_function()
            except ValueError as e:
                raise exceptions.DataProcessingError(f"Processing failed: {e}")
```

## Package Architecture

### Performance Considerations

The common package is designed for minimal overhead:

- **Optimized utilities** - Core functions use efficient algorithms
- **Configuration caching** - Config loading is cached to avoid repeated I/O
- **Lazy initialization** - Resources loaded only when needed
- **Minimal logging overhead** - Logging configured for production performance

### Dependencies

Minimal dependencies to avoid conflicts across packages:

```txt
# Core dependencies
python>=3.8
typing-extensions>=4.0.0  # For Python <3.10 compatibility
```

> **Philosophy**: Keep dependencies minimal to avoid version conflicts when using multiple dataknobs packages together.

## Complete API Reference

For comprehensive auto-generated API documentation with all classes, methods, and functions including full signatures and type annotations, see:

**[📖 dataknobs-common Complete API Reference](reference/common.md)**

This curated guide focuses on practical examples and usage patterns. The complete reference provides exhaustive technical documentation auto-generated from source code docstrings.

---

## Contributing

Contributions to `dataknobs-common` are welcome! Since this package provides shared functionality, changes here can affect all other dataknobs packages.

See the [Contributing Guide](https://github.com/KBS-Labs/dataknobs/blob/main/CONTRIBUTING.md) for information on how to contribute.

## Changelog

See the [project changelog](https://github.com/KBS-Labs/dataknobs/blob/main/CHANGELOG.md) for detailed version history.

### Version 1.0.0

- Initial release with core package structure
- Common patterns and conventions established

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/KBS-Labs/dataknobs/blob/main/LICENSE) file for details.