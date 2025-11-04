# Package Registry System

This directory contains the centralized package registry and tools for managing package references across the monorepo.

## Overview

The package registry system solves the problem of keeping package references consistent across:
- GitHub workflows
- Documentation
- Configuration files
- README files

## Files

- **`packages.json`**: Single source of truth for all packages
- **`packages-schema.json`**: JSON schema for validation (optional)
- **`example-dynamic-workflow.yml`**: Example of using the registry in workflows

## Usage

### Adding a New Package

1. **Add to registry** (`.dataknobs/packages.json`):
   ```json
   {
     "name": "newpackage",
     "pypi_name": "dataknobs-newpackage",
     "description": "New package description",
     "version": "0.1.0",
     "category": "core",
     "requires_docs_build": true,
     "deprecated": false
   }
   ```

2. **Validate references**:
   ```bash
   python bin/validate-package-references.py
   ```

3. **Update workflows** (if needed):
   ```bash
   # Get list for workflow update
   python bin/list-packages.py --format uv-install --requires-docs
   ```

### Utility Scripts

#### List Packages

```bash
# YAML format for GitHub Actions matrix
python bin/list-packages.py --format yaml

# Workflow input choices format
python bin/list-packages.py --format choices

# pip install command
python bin/list-packages.py --format pip

# uv install commands for docs
python bin/list-packages.py --format uv-install --requires-docs

# Markdown table rows
python bin/list-packages.py --format table --category core
```

#### Validate References

```bash
# Check all package references
python bin/validate-package-references.py

# Add to CI to prevent inconsistencies
```

## Package Fields

Each package in `packages.json` has:

- **`name`**: Package directory name (e.g., "llm")
- **`pypi_name`**: PyPI package name (e.g., "dataknobs-llm")
- **`description`**: One-line description
- **`version`**: Current version
- **`category`**: One of: `core`, `experimental`, `legacy`
- **`requires_docs_build`**: Whether package needs to be installed for mkdocs build
- **`deprecated`**: Whether package is deprecated

## Integration with CI/CD

### Option 1: Dynamic Discovery (Recommended)

Use the registry to dynamically generate package lists in workflows:

```yaml
jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      packages: ${{ steps.get-packages.outputs.packages }}
    steps:
      - uses: actions/checkout@v4
      - name: Get package list
        id: get-packages
        run: |
          PACKAGES=$(python bin/list-packages.py --format yaml --requires-docs)
          echo "packages=$PACKAGES" >> $GITHUB_OUTPUT

  build:
    needs: setup
    strategy:
      matrix:
        package: ${{ fromJson(needs.setup.outputs.packages) }}
    # ... use ${{ matrix.package }}
```

### Option 2: Validation Only

Keep manual lists but validate them in CI:

```yaml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate package references
        run: python bin/validate-package-references.py
```

## Benefits

1. **Single Source of Truth**: All package metadata in one place
2. **Automatic Validation**: Catch missing references in CI
3. **Easy Updates**: Add package once, update everywhere
4. **Documentation**: Clear structure for package metadata
5. **Extensible**: Easy to add new fields or checks

## Migration Path

1. ✅ Create registry and utility scripts
2. ⏳ Add validation to CI (non-blocking)
3. ⏳ Gradually migrate workflows to use dynamic discovery
4. ⏳ Make validation blocking once stabilized

## Future Enhancements

- Auto-generate README package sections
- Auto-generate docs/index.md tables
- Sync versions from pyproject.toml files
- Detect new packages automatically
- GitHub Action for package registration
