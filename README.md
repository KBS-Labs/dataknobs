DataKnobs
=============================

## Description

Useful implementations of data structures and design patterns for knowledge bases and AI, or the knobs and levers for fine-tuning and leveraging your data.

This monorepo contains modular packages for development, experimentation, and testing of general data structures, algorithms, and utilities for DS, AI, ML, and NLP.

## 📦 Packages

The project is organized as a monorepo with the following packages:

- **[dataknobs-config](packages/config/)**: Modular configuration system with environment variable substitution, factory registration, and cross-references
- **[dataknobs-data](packages/data/)**: Unified data abstraction layer supporting Memory, File, PostgreSQL, Elasticsearch, and S3 backends
- **[dataknobs-structures](packages/structures/)**: Data structures for AI knowledge bases (trees, documents, record stores)
- **[dataknobs-utils](packages/utils/)**: Utility functions (file I/O, JSON processing, pandas helpers, web requests)
- **[dataknobs-xization](packages/xization/)**: Text normalization and tokenization tools
- **[dataknobs-common](packages/common/)**: Shared base functionality
- **[dataknobs](packages/legacy/)**: Legacy compatibility package (deprecated)

## 🚀 Installation

### For New Projects (Recommended)

Install only the packages you need:

```bash
# Install specific packages
pip install dataknobs-config
pip install dataknobs-structures
pip install dataknobs-utils
pip install dataknobs-xization

# Or install multiple packages
pip install dataknobs-config dataknobs-structures dataknobs-utils
```

### For Existing Projects

For backward compatibility, you can still install the legacy package:

```bash
pip install dataknobs
```

⚠️ **Note**: The legacy package shows deprecation warnings. Please migrate to the modular packages.

## 📖 Usage

### Using Modular Packages (Recommended)

```python
# Import from specific packages
from dataknobs_config import Config
from dataknobs_data import Record, Query, database_factory
from dataknobs_structures import Tree, Document
from dataknobs_utils import json_utils, file_utils
from dataknobs_xization import MaskingTokenizer

# Configuration with environment variables and factories
config = Config("config.yaml")
config.register_factory("database", database_factory)

# Create database from configuration
# Supports: ${ENV_VAR:default} substitution
database = config.get_instance("databases", "primary")

# Work with unified data abstraction
record = Record({"name": "example", "value": 42})
record_id = database.create(record)
results = database.search(Query().filter("name", "=", "example"))

# Use factory directly for dynamic backend selection
from dataknobs_data import DatabaseFactory
factory = DatabaseFactory()
s3_db = factory.create(backend="s3", bucket="my-bucket")
memory_db = factory.create(backend="memory")

# Create a tree structure
tree = Tree("root")
tree.add_child("child1")

# Work with JSON
data = json_utils.load_json_file("data.json")
value = json_utils.get_value(data, "path.to.value")
```

### Using Legacy Package (Deprecated)

```python
# Old style imports (shows deprecation warning)
from dataknobs.structures.tree import Tree
from dataknobs.utils.json_utils import get_value
```


## 🛠️ Development

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management and a monorepo structure for better modularity.

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized development)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/dataknobs.git
cd dataknobs

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --all-packages

# Run tests for all packages
uv run pytest packages/*/tests/ -v

# Run tests for a specific package
uv run pytest packages/structures/tests/ -v
```

### Development with Docker

```bash
# Build and run development environment
docker-compose up dataknobs-dev

# Run Jupyter notebook server
docker-compose up jupyter

# Run production server
docker-compose up dataknobs-prod
```

### Testing

```bash
# Run all tests with coverage
uv run pytest packages/*/tests/ --cov=packages --cov-report=term-missing

# Run linting
uv run pylint packages/*/src --rcfile=.pylintrc

# Using tox (legacy)
tox -e tests  # Run tests
tox -e lint   # Run linting
```

### Building Packages

```bash
# Build all packages
for pkg in packages/*; do
  cd "$pkg" && uv build && cd ../..
done

# Build specific package
cd packages/structures && uv build
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
