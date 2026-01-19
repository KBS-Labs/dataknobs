# Dataknobs Release Process

## Overview

Dataknobs uses a structured release process with version tracking and documentation synchronization.

## Version Management

### Source of Truth

Package versions are managed in two locations that must stay synchronized:

1. **Package pyproject.toml files**: `packages/*/pyproject.toml` - Each package's version
2. **Central registry**: `.dataknobs/packages.json` - All package versions in one place

The `release-helper.sh` script keeps these synchronized during version bumps.

### Documentation Sync

Documentation version references are automatically updated from `.dataknobs/packages.json`:

- `docs/index.md` - Version table in Package Overview section
- `docs/installation.md` - Requirements.txt example versions

```bash
# Manual update
bin/docs-update-versions.sh

# Check if in sync (used by CI)
bin/docs-update-versions.sh --check
```

## Release Workflow

### 1. Check Changes

```bash
bin/release-helper.sh check
```

Shows what changed since the last release for each package.

### 2. Bump Versions

```bash
bin/release-helper.sh bump
```

Interactive workflow that:
- Shows changes for each package
- Prompts for version bump type (major/minor/patch)
- Updates `pyproject.toml` and `.dataknobs/packages.json`
- Runs `uv lock` to update lock file
- Updates `docs/index.md` version table automatically

### 3. Sync Init Versions (if needed)

```bash
bin/release-helper.sh sync-versions
```

Ensures `__version__` in `__init__.py` files matches `pyproject.toml`.

### 4. Generate Release Notes

```bash
bin/release-helper.sh notes
```

Generates release notes from commits since last release.

### 5. Create Tags

```bash
bin/release-helper.sh tag
```

Creates git tags for packages with version changes.

### 6. Publish to PyPI

```bash
bin/release-helper.sh publish
```

Publishes packages to PyPI.

### All-in-One

```bash
bin/release-helper.sh all
```

Runs the complete release process interactively.

## CI Validation

The `docs-version-check.yml` workflow runs on PRs that modify:
- `packages/*/pyproject.toml`
- `.dataknobs/packages.json`
- `docs/index.md`

It verifies that documentation versions match the central registry.

## Common Issues

### Documentation Versions Out of Sync

If CI reports version mismatch:

```bash
# Update documentation
bin/docs-update-versions.sh

# Verify
bin/docs-update-versions.sh --check
```

### Lock File Out of Date

After manual version edits:

```bash
uv lock
```

### Init Versions Mismatched

```bash
bin/release-helper.sh sync-versions
```
