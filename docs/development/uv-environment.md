# UV Virtual Environment Guide

## Overview

UV is a fast Python package manager that replaces pip, pip-tools, pipx, poetry, pyenv, and virtualenv. Unlike Poetry, UV handles virtual environments slightly differently.

## Activating the Virtual Environment

### Option 1: Use `uv run` (Recommended for scripts)
```bash
# Run any command in the virtual environment
uv run python script.py
uv run pytest
uv run jupyter lab
```

### Option 2: Activate the virtual environment directly
```bash
# UV creates the virtual environment in .venv by default
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Now you can run commands directly
python script.py
pytest
jupyter lab

# Deactivate when done
deactivate
```

### Option 3: Create an alias for convenience
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
alias uva='source .venv/bin/activate'
alias uvd='deactivate'
```

Then use:
```bash
uva  # Activate
python script.py  # Run without uv run prefix
uvd  # Deactivate
```

## Setting up the Environment

### Initial Setup
```bash
# Install all dependencies and workspace packages
./bin/sync-packages.sh

# Or manually:
uv sync --all-packages
# Then install all workspace packages in editable mode
uv pip install -e packages/*
```

The `sync-packages.sh` script automatically discovers and installs all packages in the workspace, even newly added ones.

### Inside Docker Container
If you're working inside the dataknobs-dev container:
```bash
# The environment should already be set up
# But if needed:
uv sync --all-packages

# Then either:
uv run pytest  # Using uv run
# OR
source .venv/bin/activate  # Activate directly
pytest
```

## Running Tests

### With uv run (no activation needed)
```bash
# Run all tests
uv run pytest

# Run specific package tests
uv run pytest packages/utils/tests/

# Run with coverage
uv run pytest --cov=packages --cov-report=term
```

### With activated environment
```bash
# First activate
source .venv/bin/activate

# Then run directly
pytest
pytest packages/utils/tests/
pytest --cov=packages --cov-report=term

# Don't forget to deactivate when done
deactivate
```

## Why Tests Might Fail with ModuleNotFoundError

The tests need to know where to find your package modules. This is handled by:

1. **conftest.py** - Automatically adds all package src directories to Python path
2. **pytest.ini** - Configures test discovery and settings
3. **package-discovery.sh** - Dynamically discovers all packages

If tests fail with ModuleNotFoundError:

1. Ensure you're in the project root directory
2. Check that `.venv` exists: `ls -la .venv`
3. Ensure dependencies are installed: `uv sync --all-packages`
4. Verify conftest.py is being loaded: `uv run pytest --collect-only`

## Comparison with Poetry

| Task | Poetry | UV |
|------|--------|-----|
| Activate environment | `poetry shell` | `source .venv/bin/activate` |
| Run command | `poetry run pytest` | `uv run pytest` |
| Install deps | `poetry install` | `uv sync` |
| Add dependency | `poetry add package` | `uv add package` |
| Virtual env location | `poetry env info --path` | `.venv` (in project root) |

## Tips

1. **For development**: Consider activating the environment once at the start of your session
2. **For scripts/CI**: Use `uv run` to ensure the correct environment
3. **For Jupyter**: `uv run jupyter lab` or activate then `jupyter lab`
4. **Inside Docker**: The environment persists, so activation works well

## Troubleshooting

### Can't find .venv
```bash
# UV creates it automatically when you sync
uv sync --all-packages
```

### Command not found after activation
```bash
# Ensure you're in the right directory
pwd  # Should be /path/to/dataknobs

# Reinstall
uv sync --all-packages
```

### Tests work in package directory but not from root
This is fixed by the conftest.py file which uses package-discovery.sh to automatically add all package src directories to the Python path.