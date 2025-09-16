# Dataknobs Documentation

Welcome to **Dataknobs** - a comprehensive Python library for AI Knowledge Base Structures.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Get up and running with Dataknobs in minutes

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

-   :material-package-variant:{ .lg .middle } __Modular Packages__

    ---

    Explore our modular architecture with specialized packages

    [:octicons-arrow-right-24: Package Overview](packages/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation with examples

    [:octicons-arrow-right-24: API Docs](api/index.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Learn best practices and advanced usage patterns

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

</div>

## What is Dataknobs?

Dataknobs is a collection of Python packages designed to facilitate the development of AI-powered knowledge base systems. It provides:

- **Data Structures**: Tree structures, documents, and record stores for organizing information
- **Utilities**: JSON processing, file handling, and integration tools
- **Text Processing**: Advanced tokenization, normalization, and text analysis capabilities
- **Modularity**: Use only what you need with our modular package design

## Package Overview

| Package | Description | Version |
|---------|-------------|---------|
| `dataknobs-data` | Data abstraction layer with multiple backend support | 1.0.0 |
| `dataknobs-fsm` | Finite State Machine framework for building workflows | 1.0.0 |
| `dataknobs-structures` | Core data structures (Tree, Document, RecordStore) | 1.0.0 |
| `dataknobs-utils` | Utility functions for JSON, files, and integrations | 1.0.0 |
| `dataknobs-xization` | Text processing, tokenization, and normalization | 1.0.0 |
| `dataknobs-common` | Shared components and base classes | 1.0.0 |
| `dataknobs` | Legacy compatibility package (deprecated) | 0.0.15 |

## Quick Installation

=== "All Packages"

    ```bash
    pip install dataknobs-structures dataknobs-utils dataknobs-xization
    ```

=== "Individual Package"

    ```bash
    # Install only what you need
    pip install dataknobs-structures
    ```

=== "Legacy (Deprecated)"

    ```bash
    # For backward compatibility only
    pip install dataknobs
    ```

## Quick Example

```python
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Create a tree structure
tree = Tree("root")
child = tree.add_child("child1")

# Process JSON data
data = {"key": {"nested": "value"}}
value = json_utils.get_value(data, "key.nested")

# Normalize text
normalized = normalize.basic_normalization_fn("Hello WORLD!")
print(normalized)  # Output: hello world!
```

## Migration from Legacy

If you're using the old `dataknobs` package, see our [Migration Guide](migration-guide.md) for upgrading to the new modular structure.

## Contributing

We welcome contributions! See our [Contributing Guide](development/contributing.md) for details.

## License

Dataknobs is released under the MIT License. See [License](license.md) for details.