# Configuration Inheritance

The `InheritableConfigLoader` provides simple YAML/JSON configuration loading with inheritance support via an `extends` field. This complements the main `Config` system for scenarios where you need lightweight, single-file configuration loading with inheritance chains.

## Overview

When building applications with multiple environments or domains, you often have:
- A base configuration with common settings
- Environment-specific overrides (dev, staging, prod)
- Domain-specific configurations that inherit from a common base

The `InheritableConfigLoader` handles this pattern elegantly.

## Quick Start

```python
from dataknobs_config import InheritableConfigLoader

# Create loader
loader = InheritableConfigLoader("./configs")

# Load configuration (resolves inheritance automatically)
config = loader.load("production")
```

Or use the convenience function:

```python
from dataknobs_config import load_config_with_inheritance

config = load_config_with_inheritance("configs/production.yaml")
```

## Configuration Files

### Base Configuration

```yaml
# configs/base.yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7

knowledge_base:
  chunk_size: 500
  overlap: 50

logging:
  level: INFO
```

### Child Configuration

```yaml
# configs/production.yaml
extends: base

llm:
  model: gpt-4-turbo
  api_key: ${OPENAI_API_KEY}

logging:
  level: WARNING
```

When you load `production`, the loader:
1. Loads `base.yaml`
2. Deep merges `production.yaml` on top
3. Substitutes environment variables
4. Returns the merged configuration

Result:
```python
{
    "llm": {
        "provider": "openai",        # From base
        "model": "gpt-4-turbo",      # Overridden
        "temperature": 0.7,          # From base
        "api_key": "sk-..."          # From env var
    },
    "knowledge_base": {              # From base
        "chunk_size": 500,
        "overlap": 50
    },
    "logging": {
        "level": "WARNING"           # Overridden
    }
}
```

## Deep Merge Behavior

Child values override parent values at the deepest level:

```python
from dataknobs_config import deep_merge

base = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "pool": {"min": 1, "max": 10}
    }
}

override = {
    "database": {
        "host": "prod.db.com",
        "pool": {"max": 50}
    }
}

result = deep_merge(base, override)
# {
#     "database": {
#         "host": "prod.db.com",  # Overridden
#         "port": 5432,           # Preserved
#         "pool": {
#             "min": 1,           # Preserved
#             "max": 50           # Overridden
#         }
#     }
# }
```

**Important**: Lists are replaced entirely, not merged:

```python
base = {"items": [1, 2, 3]}
override = {"items": [4, 5]}
result = deep_merge(base, override)
# {"items": [4, 5]}
```

## Multi-Level Inheritance

Configurations can chain inheritance:

```yaml
# configs/base.yaml
app:
  name: MyApp
  version: 1.0

# configs/development.yaml
extends: base

app:
  debug: true

database:
  host: localhost

# configs/local.yaml
extends: development

database:
  host: 127.0.0.1
  name: local_db
```

Loading `local`:
```python
config = loader.load("local")
# Resolves: base -> development -> local
```

## Environment Variable Substitution

### Required Variables

```yaml
database:
  password: ${DB_PASSWORD}  # Raises error if not set
```

### Default Values

```yaml
database:
  host: ${DB_HOST:localhost}  # Uses "localhost" if not set
  port: ${DB_PORT:5432}
```

### Path Expansion

Tilde paths are expanded after substitution:

```yaml
paths:
  data_dir: ${DATA_DIR:~/data}  # Expands ~ to home directory
```

### Disabling Substitution

```python
# Load without environment variable substitution
config = loader.load("config", substitute_vars=False)
```

## API Reference

### InheritableConfigLoader

```python
class InheritableConfigLoader:
    def __init__(self, config_dir: str | Path | None = None):
        """Initialize loader.

        Args:
            config_dir: Directory containing configs (default: ./configs)
        """

    def load(
        self,
        name: str,
        use_cache: bool = True,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load configuration with inheritance.

        Args:
            name: Config name without extension
            use_cache: Use cached config if available
            substitute_vars: Substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If config not found or cycle detected
        """

    def load_from_file(
        self,
        filepath: str | Path,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load from specific file path.

        Inheritance is resolved relative to the file's directory.
        """

    def list_available(self) -> list[str]:
        """List all available configuration names."""

    def validate(self, name: str) -> tuple[bool, str | None]:
        """Validate a configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """

    def clear_cache(self, name: str | None = None) -> None:
        """Clear configuration cache."""
```

### Utility Functions

```python
def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Override values take precedence. Nested dicts are merged recursively;
    all other types are replaced.
    """

def substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables.

    Supports ${VAR} and ${VAR:default} patterns.
    Expands ~ in paths after substitution.

    Raises:
        ValueError: If required variable not set
    """

def load_config_with_inheritance(
    filepath: str | Path,
    substitute_vars: bool = True,
) -> dict[str, Any]:
    """Convenience function to load a config file with inheritance."""
```

### InheritanceError

```python
class InheritanceError(Exception):
    """Error during configuration inheritance resolution.

    Raised for:
    - Config file not found
    - Circular inheritance detected
    - Invalid YAML/JSON
    - Config is not a dictionary
    """
```

## Caching

Configurations are cached by default for performance:

```python
# First load - reads from disk
config1 = loader.load("production")

# Second load - returns cached version
config2 = loader.load("production")  # Same object

# Force reload
config3 = loader.load("production", use_cache=False)

# Clear specific cache
loader.clear_cache("production")

# Clear all cache
loader.clear_cache()
```

## Error Handling

### Missing Configuration

```python
try:
    config = loader.load("nonexistent")
except InheritanceError as e:
    print(f"Config not found: {e}")
```

### Circular Inheritance

```yaml
# configs/a.yaml
extends: b

# configs/b.yaml
extends: a  # Circular!
```

```python
try:
    config = loader.load("a")
except InheritanceError as e:
    print(f"Circular inheritance: {e}")
```

### Missing Environment Variable

```python
try:
    config = loader.load("config")  # Has ${REQUIRED_VAR}
except ValueError as e:
    print(f"Missing env var: {e}")
```

## Best Practices

1. **Keep Base Minimal**: Put only truly common values in base configs
2. **Use Descriptive Names**: `production.yaml`, `development.yaml`, not `prod.yaml`
3. **Document Required Variables**: Comment which env vars must be set
4. **Validate in CI**: Use `loader.validate()` in tests
5. **Avoid Deep Inheritance**: 2-3 levels maximum for maintainability

## Comparison with Config Class

| Feature | InheritableConfigLoader | Config |
|---------|------------------------|--------|
| **Use Case** | Simple YAML/JSON loading | Complex, type-organized configs |
| **Inheritance** | Single `extends` field | File references with `@` |
| **Structure** | Free-form dictionary | Type-organized arrays |
| **Env Vars** | `${VAR:default}` | `DATAKNOBS_*` pattern |
| **Object Building** | No | Yes (factories, classes) |
| **References** | No | Yes (`xref:type[name]`) |

Choose `InheritableConfigLoader` for simpler configuration needs where you don't need object construction or cross-references.
