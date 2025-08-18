# Adding Configuration Support to DataKnobs Packages

This guide provides step-by-step instructions for adding configuration support to new or existing DataKnobs packages and classes.

## When to Add Configuration Support

Add configuration support when:
- Your class might be instantiated from external configuration files
- Users need to configure your class without modifying code
- Your class is part of a larger system that uses dependency injection
- You want to support environment-based configuration
- Your class has complex initialization parameters

## Step-by-Step Implementation

### 1. Add Dependencies

First, ensure your package depends on `dataknobs-config`:

```toml
# pyproject.toml
[project]
dependencies = [
    "dataknobs-config>=0.1.0",
    # ... other dependencies
]

[tool.uv.sources]
dataknobs-config = { workspace = true }
```

### 2. Import ConfigurableBase

```python
from dataknobs_config import ConfigurableBase
```

### 3. Update Class Definition

#### Option A: New Class

```python
from dataknobs_config import ConfigurableBase
from typing import Dict, Any, Optional

class MyConfigurableClass(ConfigurableBase):
    """A configurable class that follows DataKnobs patterns.
    
    Configuration Options:
        param1 (str): Description of param1 (required)
        param2 (int): Description of param2 (default: 100)
        param3 (bool): Description of param3 (default: False)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize from configuration dictionary.
        
        Args:
            config: Configuration dictionary with class parameters
        """
        self.config = config or {}
        self.param1 = self.config.get("param1")
        self.param2 = self.config.get("param2", 100)
        self.param3 = self.config.get("param3", False)
        
        if not self.param1:
            raise ValueError("param1 is required in configuration")
        
        self._initialize()
    
    def _initialize(self):
        """Perform any complex initialization."""
        # Your initialization logic here
        pass
    
    @classmethod
    def from_config(cls, config: dict) -> "MyConfigurableClass":
        """Create instance from configuration dictionary.
        
        This method is called by the Config.get_instance() method.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured instance of the class
        """
        return cls(config)
```

#### Option B: Existing Class (Backward Compatible)

```python
from dataknobs_config import ConfigurableBase
from typing import Dict, Any, Optional

class ExistingClass(SomeBaseClass, ConfigurableBase):
    """An existing class with added configuration support.
    
    Can be instantiated either directly or from configuration.
    
    Direct instantiation:
        obj = ExistingClass("value1", 42, flag=True)
    
    Configuration instantiation:
        obj = ExistingClass.from_config({
            "param1": "value1",
            "param2": 42,
            "flag": True
        })
    
    Configuration Options:
        param1 (str): Description of param1
        param2 (int): Description of param2
        flag (bool): Optional flag (default: False)
    """
    
    def __init__(self, 
                 param1: Optional[str] = None,
                 param2: Optional[int] = None,
                 flag: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize with parameters or configuration.
        
        Args:
            param1: First parameter (used if config not provided)
            param2: Second parameter (used if config not provided)
            flag: Optional flag
            config: Configuration dictionary (overrides other params)
        """
        if config:
            # Configuration-based initialization
            self.param1 = config.get("param1", param1)
            self.param2 = config.get("param2", param2)
            self.flag = config.get("flag", flag)
        else:
            # Direct initialization
            self.param1 = param1
            self.param2 = param2
            self.flag = flag
        
        # Call parent class initialization
        super().__init__()
    
    @classmethod
    def from_config(cls, config: dict) -> "ExistingClass":
        """Create instance from configuration dictionary."""
        return cls(config=config)
```

### 4. Handle Complex Configurations

For classes with nested configurations or references:

```python
class ComplexConfigurableClass(ConfigurableBase):
    """A class with complex configuration needs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Handle nested configuration
        self.database_config = self.config.get("database", {})
        self.cache_config = self.config.get("cache", {})
        
        # Initialize sub-components
        self._init_database()
        self._init_cache()
    
    def _init_database(self):
        """Initialize database from nested config."""
        if self.database_config:
            from dataknobs_data import Database
            self.db = Database.create(
                self.database_config.get("backend", "memory"),
                self.database_config
            )
    
    def _init_cache(self):
        """Initialize cache from nested config."""
        if self.cache_config:
            self.cache_size = self.cache_config.get("size", 1000)
            self.cache_ttl = self.cache_config.get("ttl", 3600)
```

### 5. Implement Factory Pattern (Optional)

For creating different implementations based on configuration:

```python
from dataknobs_config import FactoryBase

class MyClassFactory(FactoryBase):
    """Factory for creating MyClass instances based on type."""
    
    def create(self, **config) -> Any:
        """Create instance based on configuration.
        
        Args:
            **config: Configuration including 'type' field
            
        Returns:
            Instance of appropriate class
        """
        class_type = config.pop("type", "default")
        
        if class_type == "advanced":
            from .advanced import AdvancedImplementation
            return AdvancedImplementation(config)
        elif class_type == "simple":
            from .simple import SimpleImplementation
            return SimpleImplementation(config)
        else:
            from .default import DefaultImplementation
            return DefaultImplementation(config)
```

## Testing Configuration Support

### 1. Unit Tests

```python
import pytest
from mypackage import MyConfigurableClass

class TestConfigSupport:
    """Test configuration support for MyConfigurableClass."""
    
    def test_direct_instantiation(self):
        """Test direct instantiation still works."""
        obj = MyConfigurableClass({"param1": "value1"})
        assert obj.param1 == "value1"
    
    def test_from_config_method(self):
        """Test from_config classmethod."""
        config = {
            "param1": "value1",
            "param2": 200,
            "param3": True
        }
        obj = MyConfigurableClass.from_config(config)
        assert obj.param1 == "value1"
        assert obj.param2 == 200
        assert obj.param3 is True
    
    def test_missing_required_param(self):
        """Test that missing required parameters raise errors."""
        with pytest.raises(ValueError, match="param1 is required"):
            MyConfigurableClass({})
    
    def test_default_values(self):
        """Test that default values are applied."""
        obj = MyConfigurableClass({"param1": "value1"})
        assert obj.param2 == 100  # default value
        assert obj.param3 is False  # default value
```

### 2. Integration Tests

```python
from dataknobs_config import Config

def test_config_integration():
    """Test integration with Config class."""
    config = Config()
    config.load({
        "my_objects": [{
            "name": "test_object",
            "class": "mypackage.MyConfigurableClass",
            "param1": "test_value",
            "param2": 300
        }]
    })
    
    # Test that object can be built from config
    obj = config.get_instance("my_objects", "test_object")
    assert isinstance(obj, MyConfigurableClass)
    assert obj.param1 == "test_value"
    assert obj.param2 == 300

def test_environment_variables():
    """Test environment variable substitution."""
    import os
    os.environ["MY_PARAM"] = "env_value"
    
    config = Config()
    config.load({
        "my_objects": [{
            "name": "env_test",
            "class": "mypackage.MyConfigurableClass",
            "param1": "${MY_PARAM}"
        }]
    })
    
    obj = config.get_instance("my_objects", "env_test")
    assert obj.param1 == "env_value"
```

## Documentation Requirements

### 1. Class Docstring

Always document configuration options in the class docstring:

```python
class WellDocumentedClass(ConfigurableBase):
    """A well-documented configurable class.
    
    This class can be instantiated directly or from configuration files
    using the DataKnobs configuration system.
    
    Configuration Options:
        host (str): Server hostname (required)
        port (int): Server port (default: 8080)
        timeout (int): Connection timeout in seconds (default: 30)
        ssl (bool): Enable SSL/TLS (default: False)
        credentials (dict): Optional credentials dictionary with:
            - username (str): Username for authentication
            - password (str): Password for authentication
    
    Example Configuration:
        servers:
          - name: production
            class: mypackage.WellDocumentedClass
            host: prod.example.com
            port: 443
            ssl: true
            credentials:
              username: ${API_USER}
              password: ${API_PASSWORD}
    
    Example Usage:
        >>> from dataknobs_config import Config
        >>> config = Config("config.yaml")
        >>> server = config.get_instance("servers", "production")
    """
```

### 2. README Examples

Add configuration examples to your package README:

```markdown
## Configuration Support

This package supports the DataKnobs configuration system. All main classes
inherit from `ConfigurableBase` and can be instantiated from configuration files.

### Example Configuration

```yaml
# config.yaml
my_services:
  - name: processor
    class: mypackage.DataProcessor
    input_dir: /data/input
    output_dir: /data/output
    batch_size: 100
    
  - name: validator
    class: mypackage.DataValidator
    rules_file: /config/rules.yaml
    strict_mode: true
```

### Loading from Configuration

```python
from dataknobs_config import Config

config = Config("config.yaml")
processor = config.get_instance("my_services", "processor")
validator = config.get_instance("my_services", "validator")
```
```

## Common Patterns

### Pattern 1: Optional Dependencies

```python
class OptionalDependencyClass(ConfigurableBase):
    """Class with optional dependencies based on configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend = self.config.get("backend", "basic")
        
        if self.backend == "advanced":
            try:
                import advanced_library
                self._setup_advanced()
            except ImportError:
                raise ImportError(
                    "Advanced backend requires 'advanced_library'. "
                    "Install with: pip install mypackage[advanced]"
                )
        else:
            self._setup_basic()
```

### Pattern 2: Validation

```python
from dataknobs_config import ConfigurableBase, ValidationError

class ValidatedClass(ConfigurableBase):
    """Class with configuration validation."""
    
    @classmethod
    def from_config(cls, config: dict) -> "ValidatedClass":
        """Create instance with validation."""
        # Validate required fields
        required = ["field1", "field2"]
        missing = [f for f in required if f not in config]
        if missing:
            raise ValidationError(f"Missing required fields: {missing}")
        
        # Validate types
        if not isinstance(config.get("port"), int):
            raise ValidationError("'port' must be an integer")
        
        # Validate ranges
        port = config.get("port")
        if port and not (1 <= port <= 65535):
            raise ValidationError("'port' must be between 1 and 65535")
        
        return cls(config)
```

### Pattern 3: Lazy Initialization

```python
class LazyInitClass(ConfigurableBase):
    """Class with lazy initialization from config."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._connection = None
    
    @property
    def connection(self):
        """Lazily initialize connection when first accessed."""
        if self._connection is None:
            self._initialize_connection()
        return self._connection
    
    def _initialize_connection(self):
        """Initialize connection from config."""
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 8080)
        self._connection = create_connection(host, port)
```

## Checklist

Before considering configuration support complete:

- [ ] Class inherits from `ConfigurableBase`
- [ ] `from_config()` classmethod is implemented
- [ ] Configuration options are documented in class docstring
- [ ] Unit tests verify configuration-based instantiation
- [ ] Integration tests verify Config.get_instance() works
- [ ] README includes configuration examples
- [ ] Backward compatibility is maintained (if updating existing class)
- [ ] Environment variable substitution is tested
- [ ] Default values are documented and tested
- [ ] Required parameters are validated
- [ ] Error messages are helpful and specific

## Troubleshooting

### Issue: ImportError when using Config.get_instance()

**Solution**: Ensure the module path in the `class` attribute is correct and the module is importable:

```python
# Correct: full module path
"class": "mypackage.submodule.MyClass"

# Incorrect: missing package prefix
"class": "submodule.MyClass"
```

### Issue: TypeError on instantiation

**Solution**: Implement `from_config()` to handle the config dictionary:

```python
@classmethod
def from_config(cls, config: dict):
    return cls(config)  # or cls(**config) if appropriate
```

### Issue: Circular imports

**Solution**: Use lazy imports inside methods:

```python
def _init_component(self):
    # Import here to avoid circular dependency
    from .other_module import OtherClass
    self.component = OtherClass(self.config.get("component", {}))
```

## Next Steps

1. Review the [Configuration System Documentation](./configuration-system.md)
2. Look at examples in the `dataknobs-data` package
3. Test your implementation with the Config class
4. Add your class to the package documentation
5. Consider contributing your patterns back to this guide