# Installation Guide

## System Requirements

- Python 3.10 or higher
- pip or uv package manager
- Operating System: Linux, macOS, or Windows

## Installation Methods

### Using pip (Recommended)

Choose the packages you need based on your use case:

#### For AI/ML Applications

```bash
# Chatbots and AI agents
pip install dataknobs-bots dataknobs-llm

# Add configuration and data persistence
pip install dataknobs-config dataknobs-data
```

#### For Data Engineering

```bash
# Workflow orchestration and data processing
pip install dataknobs-fsm dataknobs-data dataknobs-config
```

#### For General Development

```bash
# Core data structures and utilities
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

#### All Packages

```bash
# Install everything
pip install dataknobs-config dataknobs-data dataknobs-fsm \
            dataknobs-llm dataknobs-bots \
            dataknobs-structures dataknobs-utils dataknobs-xization
```

### Individual Packages

Install specific packages as needed:

```bash
pip install dataknobs-config       # Configuration management
pip install dataknobs-data         # Data abstraction layer
pip install dataknobs-fsm          # Finite state machines
pip install dataknobs-llm          # LLM integration
pip install dataknobs-bots         # AI agents and chatbots
pip install dataknobs-structures   # Core data structures
pip install dataknobs-utils        # Utility functions
pip install dataknobs-xization     # Text processing
pip install dataknobs-common       # Shared base classes
```

### Using uv

If you prefer the uv package manager:

```bash
# Install specific packages
uv pip install dataknobs-bots dataknobs-llm

# Or install all packages
uv pip install dataknobs-config dataknobs-data dataknobs-fsm \
               dataknobs-llm dataknobs-bots \
               dataknobs-structures dataknobs-utils dataknobs-xization
```

### Development Installation

For contributing or development work:

```bash
# Clone the repository
git clone https://github.com/kbs-labs/dataknobs.git
cd dataknobs

# Install with uv (installs all packages in editable mode)
uv sync --all-packages

# Or install individual packages in editable mode with pip
pip install -e packages/config
pip install -e packages/data
pip install -e packages/fsm
pip install -e packages/llm
pip install -e packages/bots
pip install -e packages/structures
pip install -e packages/utils
pip install -e packages/xization
```

## Verifying Installation

Check that packages are installed correctly:

```python
# Test imports
from dataknobs_config import Config
from dataknobs_data import Record, Query
from dataknobs_fsm import SimpleFSM
from dataknobs_llm import LLM
from dataknobs_bots import BotRegistry
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Check versions
import dataknobs_config
import dataknobs_data
import dataknobs_fsm
import dataknobs_llm
import dataknobs_bots
import dataknobs_structures
import dataknobs_utils
import dataknobs_xization

print(f"Config: {dataknobs_config.__version__}")
print(f"Data: {dataknobs_data.__version__}")
print(f"FSM: {dataknobs_fsm.__version__}")
print(f"LLM: {dataknobs_llm.__version__}")
print(f"Bots: {dataknobs_bots.__version__}")
print(f"Structures: {dataknobs_structures.__version__}")
print(f"Utils: {dataknobs_utils.__version__}")
print(f"Xization: {dataknobs_xization.__version__}")
```

## Optional Dependencies

### Data Package Backends

The Data package supports multiple backends with additional dependencies:

```bash
# PostgreSQL support
pip install psycopg2-binary>=2.9.0
# or for async
pip install asyncpg>=0.27.0

# Elasticsearch support
pip install elasticsearch>=8.0.0

# S3 support
pip install boto3>=1.26.0
# or for async
pip install aioboto3>=11.0.0

# Pandas integration
pip install pandas>=1.5.0
```

### LLM Providers

Different LLM providers require their own packages:

```bash
# OpenAI
pip install openai>=1.0.0

# Anthropic
pip install anthropic>=0.18.0

# Ollama (local models)
# No package needed - just run Ollama server locally
```

### FSM Advanced Features

For advanced FSM features:

```bash
# Async FSM support
pip install aiofiles>=23.0.0

# Streaming support
pip install aiostream>=0.5.0
```

### Text Processing

For advanced text processing:

```bash
# NLP features
pip install nltk>=3.8.0
pip install spacy>=3.5.0
```

## Package Dependencies

Each package has its own dependencies automatically installed:

- **dataknobs-common**: No external dependencies
- **dataknobs-config**: PyYAML
- **dataknobs-data**: Varies by backend (see above)
- **dataknobs-fsm**: pydantic
- **dataknobs-llm**: Varies by provider (see above)
- **dataknobs-bots**: dataknobs-llm, dataknobs-data (optional)
- **dataknobs-structures**: No external dependencies
- **dataknobs-utils**: requests (optional)
- **dataknobs-xization**: No external dependencies

## Troubleshooting

### Import Errors

If you encounter import errors after installation:

1. **Check Python version**: `python --version` (must be 3.10+)
2. **Verify installation**: `pip list | grep dataknobs`
3. **Check for naming**: Use `dataknobs_package` not `dataknobs.package`
4. **Clear pip cache**: `pip cache purge`
5. **Reinstall**: `pip install --force-reinstall dataknobs-structures`

### Virtual Environment Issues

If you have dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv dataknobs-env
source dataknobs-env/bin/activate  # On Windows: dataknobs-env\Scripts\activate

# Install in clean environment
pip install dataknobs-config dataknobs-data dataknobs-fsm \
            dataknobs-llm dataknobs-bots \
            dataknobs-structures dataknobs-utils dataknobs-xization
```

### Using uv for Faster Installs

For faster dependency resolution:

```bash
# Install uv
pip install uv

# Use uv for package installation
uv pip install dataknobs-bots dataknobs-llm
```

### Platform-Specific Issues

**macOS with Apple Silicon**:
```bash
# Some dependencies may need compilation
brew install postgresql  # For psycopg2
pip install psycopg2-binary  # Pre-compiled version
```

**Windows**:
```bash
# May need Visual C++ Build Tools for some dependencies
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**Linux**:
```bash
# May need development headers
sudo apt-get install python3-dev libpq-dev  # Ubuntu/Debian
sudo yum install python3-devel postgresql-devel  # RHEL/CentOS
```

## Production Deployment

For production environments:

### Docker

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Dataknobs packages
RUN pip install --no-cache-dir \
    dataknobs-config \
    dataknobs-data \
    dataknobs-fsm \
    dataknobs-llm \
    dataknobs-bots \
    psycopg2-binary \
    boto3

# Copy application
COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

### Requirements File

```text
# requirements.txt
dataknobs-config>=0.3.3
dataknobs-data>=0.4.5
dataknobs-fsm>=0.1.6
dataknobs-llm>=0.3.1
dataknobs-bots>=0.3.1
dataknobs-structures>=1.0.5
dataknobs-utils>=1.2.2
dataknobs-xization>=1.2.4

# Optional dependencies
psycopg2-binary>=2.9.0
elasticsearch>=8.0.0
boto3>=1.26.0
openai>=1.0.0
```

Install from requirements:
```bash
pip install -r requirements.txt
```

## Next Steps

- [Getting Started](getting-started.md) - Quick start guide
- [User Guide](user-guide/index.md) - Detailed usage instructions
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples/index.md) - Real-world usage examples
