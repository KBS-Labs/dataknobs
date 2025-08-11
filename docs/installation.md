# Installation Guide

## System Requirements

- Python 3.10 or higher
- pip or uv package manager
- Operating System: Linux, macOS, or Windows

## Installation Methods

### Using pip (Recommended)

Install the packages you need:

```bash
# Install all main packages
pip install dataknobs-structures dataknobs-utils dataknobs-xization

# Or install individual packages
pip install dataknobs-structures  # Core data structures
pip install dataknobs-utils       # Utility functions
pip install dataknobs-xization    # Text processing
```

### Using uv

If you prefer the uv package manager:

```bash
# Install with uv
uv pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

### Development Installation

For contributing or development work:

```bash
# Clone the repository
git clone https://github.com/yourusername/dataknobs.git
cd dataknobs

# Install with uv in development mode
uv sync

# Or install individual packages in editable mode
pip install -e packages/structures
pip install -e packages/utils
pip install -e packages/xization
```

## Verifying Installation

Check that packages are installed correctly:

```python
# Test imports
from dataknobs_structures import Tree
from dataknobs_utils import json_utils
from dataknobs_xization import normalize

# Check versions
import dataknobs_structures
import dataknobs_utils
import dataknobs_xization

print(f"Structures: {dataknobs_structures.__version__}")
print(f"Utils: {dataknobs_utils.__version__}")
print(f"Xization: {dataknobs_xization.__version__}")
```

## Optional Dependencies

Some features require additional packages:

### For Pandas Support
```bash
pip install pandas>=1.3.0
```

### For Elasticsearch Integration
```bash
pip install elasticsearch>=7.10.0
```

### For LLM Utilities
```bash
pip install openai>=1.0.0
```

## Troubleshooting

### Import Errors

If you encounter import errors after installation:

1. Check Python version: `python --version` (must be 3.10+)
2. Verify installation: `pip list | grep dataknobs`
3. Clear pip cache: `pip cache purge`
4. Reinstall: `pip install --force-reinstall dataknobs-structures`

### Dependency Conflicts

If you have dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv dataknobs-env
source dataknobs-env/bin/activate  # On Windows: dataknobs-env\Scripts\activate

# Install in clean environment
pip install dataknobs-structures dataknobs-utils dataknobs-xization
```

## Next Steps

- [Getting Started](getting-started.md) - Quick start guide
- [User Guide](user-guide/index.md) - Detailed usage instructions
- [API Reference](api/index.md) - Complete API documentation