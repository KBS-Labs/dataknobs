# New Package Creation Checklist

This guide walks you through creating a new package in the DataKnobs monorepo using the automated `create-package.py` script and completing the manual integration steps.

## Quick Start

```bash
# Create a new package
./bin/create-package.py mypackage \
  --description "Brief description of package" \
  --version 0.1.0 \
  --category core

# Validate integration
./bin/validate-package-references.py

# Sync dependencies
uv sync --all-packages
```

## Automated Steps (via create-package.py)

The `create-package.py` script automatically handles:

- ✅ Package directory structure creation
- ✅ Basic package files (`pyproject.toml`, `__init__.py`, `README.md`)
- ✅ Test file template
- ✅ `.dataknobs/packages.json` registration
- ✅ GitHub workflows update (docs.yml, quality-validation.yml, release.yml)
- ✅ Root `README.md` integration
- ✅ Root `pyproject.toml` integration

## Manual Steps Checklist

After running the automated script, complete these manual steps:

### 1. Implement Package Code

**Location**: `packages/<name>/src/dataknobs_<name>/`

**Tasks**:
- [ ] Design and implement core functionality
- [ ] Add proper type hints to all functions/classes
- [ ] Follow DataKnobs coding standards (see [Contributing Guide](contributing.md))
- [ ] Add docstrings using Google style
- [ ] Import and export public API in `__init__.py`

**Example `__init__.py`**:
```python
"""DataKnobs MyPackage.

Core functionality for my package.
"""

__version__ = "0.1.0"

from .core import MyClass, my_function

__all__ = [
    "__version__",
    "MyClass",
    "my_function",
]
```

### 2. Write Comprehensive Tests

**Location**: `packages/<name>/tests/`

**Tasks**:
- [ ] Write unit tests for all public functions/classes
- [ ] Add integration tests if needed
- [ ] Test error cases and edge conditions
- [ ] Achieve >80% code coverage
- [ ] Add fixtures for common test data
- [ ] Test async functions if applicable

**Example Test Structure**:
```
tests/
├── __init__.py
├── test_core.py           # Core functionality tests
├── test_integration.py    # Integration tests
├── test_edge_cases.py     # Edge cases and errors
└── fixtures/
    └── sample_data.py     # Test fixtures
```

**Run Tests**:
```bash
# Run package tests
uv run pytest packages/<name>/tests/ -v

# With coverage
uv run pytest packages/<name>/tests/ --cov=packages/<name>/src --cov-report=term-missing
```

### 3. Create Package Documentation

**Location**: `packages/<name>/docs/`

**Tasks**:
- [ ] Create `README.md` with overview and quick examples
- [ ] Write `USER_GUIDE.md` with tutorials and how-to guides
- [ ] Create `API.md` with complete API reference
- [ ] Add `CONFIGURATION.md` if package uses configuration
- [ ] Include code examples in `examples/` directory

**Recommended Structure**:
```
docs/
├── README.md              # Package overview
├── USER_GUIDE.md          # Tutorials and guides
├── API.md                 # API reference
├── CONFIGURATION.md       # Configuration options (if needed)
├── ARCHITECTURE.md        # Design decisions (if complex)
└── examples/
    ├── basic_usage.py
    ├── advanced_usage.py
    └── integration_example.py
```

### 4. Integrate with MkDocs (if requires_docs_build=True)

**Location**: `docs/packages/<name>/`

**Tasks**:
- [ ] Create `docs/packages/<name>/` directory
- [ ] Add `index.md` with package overview
- [ ] Create `quickstart.md` with quick start guide
- [ ] Add symbolic links to package docs (or copy if needed)
- [ ] Update `mkdocs.yml` with package navigation

**Example `docs/packages/<name>/index.md`**:
```markdown
# dataknobs-mypackage

Brief description of the package and its purpose.

## Key Features

- Feature 1
- Feature 2
- Feature 3

## Quick Start

\`\`\`python
from dataknobs_mypackage import MyClass

# Basic usage example
obj = MyClass()
result = obj.do_something()
\`\`\`

## Documentation

- [User Guide](guides/user-guide.md)
- [API Reference](api/reference.md)
- [Examples](examples/index.md)
```

**Update `mkdocs.yml`**:
```yaml
nav:
  # ... existing nav items ...
  - Packages:
      # ... existing packages ...
      - MyPackage:
          - packages/mypackage/index.md
          - Guides:
              - packages/mypackage/guides/user-guide.md
          - API Reference:
              - packages/mypackage/api/reference.md
          - Examples:
              - packages/mypackage/examples/index.md
```

### 5. Update Package Index Documentation

**Location**: `docs/packages/index.md`

**Tasks**:
- [ ] Add package to package overview table
- [ ] Add installation command
- [ ] Add to package selection guide

**Example Addition**:
```markdown
| Package | Description | Installation |
|---------|-------------|--------------|
| ... existing packages ... |
| **[dataknobs-mypackage](mypackage/)** | Brief description | `pip install dataknobs-mypackage` |
```

### 6. Add Package Dependencies

**Location**: `packages/<name>/pyproject.toml`

**Tasks**:
- [ ] Add required dependencies to `dependencies` list
- [ ] Add optional dependencies using extras (e.g., `[dev]`, `[docs]`)
- [ ] Specify version constraints appropriately
- [ ] Document why each dependency is needed

**Example**:
```toml
dependencies = [
    "dataknobs-common>=1.0.0",
    "dataknobs-config>=0.2.0",
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.21.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
]
all = [
    "dataknobs-mypackage[dev,docs]",
]
```

### 7. Configure Quality Checks

**Tasks**:
- [ ] Ensure code passes `ruff` linting
- [ ] Ensure code passes `mypy` type checking
- [ ] Run quality checks locally

**Commands**:
```bash
# Lint code
uv run ruff check packages/<name>/src/

# Type check
uv run mypy packages/<name>/src/

# Auto-fix issues
uv run ruff check --fix packages/<name>/src/

# Format code
uv run ruff format packages/<name>/src/
```

### 8. Add Examples and Use Cases

**Location**: `packages/<name>/examples/` or `docs/packages/<name>/examples/`

**Tasks**:
- [ ] Create at least 3 complete working examples
- [ ] Cover basic usage, advanced usage, and integration
- [ ] Add README explaining each example
- [ ] Ensure examples are tested and working

### 9. Update CHANGELOG

**Location**: `packages/<name>/CHANGELOG.md`

**Tasks**:
- [ ] Create CHANGELOG.md following Keep a Changelog format
- [ ] Document initial release (v0.1.0)
- [ ] Include features, changes, and any breaking changes

**Example**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2025-01-15

### Added
- Initial release
- Core functionality for X, Y, and Z
- Basic configuration support
- Documentation and examples
```

### 10. Validate Everything

**Tasks**:
- [ ] Run package reference validation
- [ ] Sync all dependencies
- [ ] Build documentation locally
- [ ] Run all tests
- [ ] Check quality artifacts

**Commands**:
```bash
# Validate package references
./bin/validate-package-references.py

# Sync dependencies
uv sync --all-packages

# Build docs
mkdocs build --strict

# Run all tests
uv run pytest packages/<name>/tests/ -v

# Run quality checks
./bin/run-quality-checks.sh
```

### 11. Prepare for First Release

**Tasks**:
- [ ] Review all code and documentation
- [ ] Create initial quality artifacts
- [ ] Test installation from local build
- [ ] Write release notes
- [ ] Create GitHub release draft (optional)

**Test Local Installation**:
```bash
# Build package
cd packages/<name>
uv build

# Install in clean environment
uv pip install dist/dataknobs_<name>-0.1.0-py3-none-any.whl

# Test import
uv run python -c "import dataknobs_<name>; print(dataknobs_<name>.__version__)"
```

## Package Categories

### Core Packages (`category: core`)
- Essential packages for typical DataKnobs usage
- Appear in README and main documentation
- Included in release workflow
- Require comprehensive documentation
- Examples: config, data, llm, bots

### Experimental Packages (`category: experimental`)
- New or unstable features
- Not included in main README
- Not in release workflow (manual release only)
- Documentation optional
- Examples: kv (key-value stores)

### Legacy Packages (`category: legacy`)
- Deprecated or compatibility packages
- Marked with deprecation warnings
- Minimal maintenance
- Examples: legacy (dataknobs v1)

## Common Issues

### Import Errors After Creation

**Problem**: Cannot import the new package

**Solution**:
```bash
# Re-sync dependencies
uv sync --all-packages

# Install in editable mode
uv pip install -e packages/<name>
```

### Documentation Not Building

**Problem**: MkDocs build fails

**Solution**:
1. Check that package is installed: `uv pip list | grep dataknobs-<name>`
2. Verify mkdocs.yml navigation structure
3. Check symbolic links are created correctly
4. Ensure all referenced files exist

### Validation Failures

**Problem**: `validate-package-references.py` reports issues

**Solution**:
1. Check error messages carefully
2. Ensure package is in `.dataknobs/packages.json`
3. Verify all workflows are updated
4. Check README.md includes package
5. Verify pyproject.toml has all required entries

## Script Options

### create-package.py Options

```bash
./bin/create-package.py <name> [options]

Required:
  -d, --description DESC    Package description

Optional:
  -v, --version VERSION     Initial version (default: 0.1.0)
  -c, --category CATEGORY   Package category: core, experimental, legacy
  --no-docs                 Skip documentation build integration
  --dry-run                 Show what would be done without making changes

Examples:
  # Standard core package
  ./bin/create-package.py ml --description "Machine learning utilities"

  # Experimental package without docs
  ./bin/create-package.py alpha-feature \
    --description "Experimental feature" \
    --category experimental \
    --no-docs

  # Preview changes
  ./bin/create-package.py mypackage \
    --description "My package" \
    --dry-run
```

## Best Practices

1. **Start with Tests**: Write tests as you develop functionality
2. **Document Early**: Write documentation alongside code
3. **Small PRs**: Create package incrementally with focused PRs
4. **Follow Patterns**: Look at existing packages (bots, llm, fsm) as examples
5. **Type Everything**: Use type hints for all public APIs
6. **Validate Often**: Run `validate-package-references.py` frequently
7. **Quality First**: Don't skip quality checks - fix issues immediately

## Resources

- [Contributing Guide](contributing.md) - Coding standards and practices
- [Testing Guide](testing-guide.md) - Testing patterns and best practices
- [Documentation Guide](documentation-guide.md) - Documentation standards
- [Quality Checks](quality-checks.md) - Running quality validation
- [Release Process](release-process.md) - Publishing packages

## Getting Help

If you encounter issues:

1. Check this checklist thoroughly
2. Review similar packages in the monorepo
3. Run validation and read error messages carefully
4. Ask in GitHub Discussions or create an issue

## Checklist Summary

Quick reference of all steps:

- [ ] Run `./bin/create-package.py <name> --description "..."`
- [ ] Implement core functionality
- [ ] Write comprehensive tests
- [ ] Create package documentation
- [ ] Integrate with MkDocs (if needed)
- [ ] Update package index docs
- [ ] Add dependencies to pyproject.toml
- [ ] Configure quality checks
- [ ] Add examples and use cases
- [ ] Update CHANGELOG
- [ ] Run validation: `./bin/validate-package-references.py`
- [ ] Sync dependencies: `uv sync --all-packages`
- [ ] Build docs: `mkdocs build --strict`
- [ ] Run tests: `uv run pytest packages/<name>/tests/ -v`
- [ ] Run quality checks: `./bin/run-quality-checks.sh`
- [ ] Test local installation
- [ ] Create pull request
