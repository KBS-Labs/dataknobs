# Dataknobs Packages

Dataknobs is organized as a collection of modular packages, each serving a specific purpose.

## Package Overview

| Package | Purpose | Key Features |
|---------|---------|--------------|
| [dataknobs-data](data/index.md) | Data abstraction layer | Records, Fields, Multiple backends, Async support |
| [dataknobs-config](config/index.md) | Configuration management | Modular configs, Environment overrides, Cross-references |
| [dataknobs-structures](structures/index.md) | Core data structures | Tree, Document, RecordStore, ConditionalDict |
| [dataknobs-utils](utils/index.md) | Utility functions | JSON, File, Elasticsearch, LLM utilities |
| [dataknobs-xization](xization/index.md) | Text processing | Tokenization, Normalization, Masking |
| [dataknobs-common](common/index.md) | Shared components | Base classes, Common utilities |
| [dataknobs](legacy/index.md) | Legacy compatibility | Backward compatibility (deprecated) |

## Installation

Install the packages you need:

```bash
# Install all main packages
pip install dataknobs-data dataknobs-config dataknobs-structures dataknobs-utils dataknobs-xization

# Or install individually
pip install dataknobs-data
pip install dataknobs-config
pip install dataknobs-structures
```

## Package Dependencies

```mermaid
graph TD
    A[dataknobs-common] --> B[dataknobs-data]
    A --> C[dataknobs-structures]
    A --> D[dataknobs-utils]
    B --> C
    C --> D
    A --> E[dataknobs-xization]
    C --> E
    D --> E
    C --> F[dataknobs-legacy]
    D --> F
    E --> F
```

## Choosing Packages

- **dataknobs-data**: For data abstraction with multiple backend support (memory, file, PostgreSQL, Elasticsearch, S3)
- **dataknobs-config**: For configuration management with environment overrides
- **dataknobs-structures**: If you need tree structures, documents, or record storage
- **dataknobs-utils**: For JSON processing, file operations, or integrations
- **dataknobs-xization**: For text processing, tokenization, or normalization
- **dataknobs-common**: Automatically installed with other packages

## Migration from Legacy

See the [Migration Guide](../migration-guide.md) for upgrading from the legacy `dataknobs` package.