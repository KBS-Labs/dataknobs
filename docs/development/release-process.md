# DataKnobs Release Process

This document provides a streamlined checklist and guide for releasing new versions of DataKnobs packages.

## Overview

The DataKnobs release process has been simplified from many manual steps to an automated workflow that:
- **Automatically detects** what changed since the last release
- **Suggests version bumps** based on the type of changes
- **Generates release notes** from commit messages
- **Provides a single command** for the entire process
- **Validates installations** to ensure packages work correctly

## Quick Commands

```bash
# Development phase
dk fix           # Auto-fix style issues
dk check         # Quick quality check
dk pr            # Full PR preparation

# Release phase  
bin/release-helper.sh check    # Check what changed
bin/release-helper.sh bump     # Bump versions
bin/release-helper.sh notes    # Generate release notes
bin/release-helper.sh tag      # Create tags
bin/release-helper.sh publish  # Publish to PyPI
```

## Pre-Release Checklist

### 1. Code Quality ✓
```bash
# Run full quality checks
dk pr

# If issues found, fix them:
dk fix                    # Auto-fix style issues
dk check <package>        # Re-check specific package
```

### 2. Merge to Main ✓
```bash
git add .
git commit -m "Your commit message"
git push origin <branch>
# Create PR on GitHub
# Wait for CI to pass
# Merge PR
```

## Release Checklist

### 1. Prepare Release Branch
```bash
git checkout main
git pull origin main
git checkout -b release/v<version>
```

### 2. Determine Version Bumps
```bash
# Check what changed since last release
bin/release-helper.sh check

# This will show:
# - Changed packages
# - Type of changes (features, fixes, breaking)
# - Suggested version bumps
```

### 3. Update Versions
```bash
# Interactive version bumping
bin/release-helper.sh bump

# Or manually edit pyproject.toml files
```

### 4. Generate Release Notes
```bash
# Generate notes for changed packages
bin/release-helper.sh notes

# This updates docs/changelog.md
# Review and edit as needed
```

### 5. Create Release PR
```bash
git add .
git commit -m "Release: <summary of packages and versions>"
git push origin release/v<version>

# Create PR on GitHub
# Ensure all CI checks pass
# Merge to main
```

### 6. Tag and Publish
```bash
# Pull the merged changes
git checkout main
git pull origin main

# Create release tags
bin/tag-releases.sh

# Build packages
bin/build-packages.sh

# Publish to PyPI
bin/publish-pypi.sh

# Or test first with TestPyPI
bin/publish-pypi.sh --test
```

### 7. Verify Installation
```bash
# Create a test environment
python -m venv test-release
source test-release/bin/activate

# Install from PyPI
pip install dataknobs-<package>

# Test import
python -c "import dataknobs_<package>; print(dataknobs_<package>.__version__)"

# Clean up
deactivate
rm -rf test-release
```

## Version Bump Guidelines

### Patch Version (0.0.X)
- Bug fixes
- Documentation updates
- Minor internal improvements

### Minor Version (0.X.0)
- New features (backwards compatible)
- New optional parameters
- Performance improvements
- Deprecations (with warnings)

### Major Version (X.0.0)
- Breaking changes
- Removed deprecated features
- Major architectural changes
- Incompatible API changes

## Automated Release Notes

The `release-helper.sh notes` command will:
1. Analyze git commits since last tag
2. Categorize changes by type
3. Group by package
4. Generate markdown for changelog

Format:
```markdown
## [Package Name] [Version] - YYYY-MM-DD

### Added
- New features

### Changed
- Modified functionality

### Fixed
- Bug fixes

### Breaking Changes
- Incompatible changes
```

## CI/CD Integration

GitHub Actions automatically:
- Run quality checks on PR
- Deploy docs when merged to main
- Validate release tags match versions

## Troubleshooting

### "Package already exists on PyPI"
- Check version wasn't already published
- Bump version if needed

### "Working directory has uncommitted changes"  
- Commit or stash changes before tagging

### "Tests failing in CI"
- Run `dk pr` locally to reproduce
- Fix issues before merging

### "Import fails after installation"
- Check dependencies in pyproject.toml
- Ensure all required packages published

## Quick Reference

| Task | Command | Description |
|------|---------|-------------|
| **Development** | | |
| Fix style issues | `dk fix` | Auto-fix linting and formatting |
| Quick check | `dk check [package]` | Fast quality check |
| Full PR check | `dk pr` | Complete quality validation |
| **Release Prep** | | |
| Check changes | `dk release-check` | See what changed since last release |
| Bump versions | `dk release-bump` | Interactive version updates |
| Generate notes | `dk release-notes` | Create changelog entries |
| Full process | `dk release` | Guided complete release |
| **Publishing** | | |
| Create tags | `bin/tag-releases.sh` | Tag package versions |
| Build packages | `bin/build-packages.sh` | Build distribution files |
| Publish to PyPI | `bin/publish-pypi.sh` | Upload to PyPI |
| Test publish | `bin/publish-pypi.sh --test` | Upload to TestPyPI |
| Verify install | `bin/release-helper.sh verify` | Test installations |

## Frequently Asked Questions (FAQ)

### Q: What's the simplest way to do a release?
**A:** Run `dk release` for an interactive, guided process that handles everything.

### Q: Do I need to run all the validation scripts (validate.sh, fix.sh, dev.sh lint)?
**A:** No, just use:
- `dk fix` to auto-fix issues
- `dk pr` for full quality checks before creating a PR

### Q: How do I know what version bump to use?
**A:** The `dk release-check` command analyzes your commits and suggests:
- **Patch** (0.0.X) for bug fixes
- **Minor** (0.X.0) for new features
- **Major** (X.0.0) for breaking changes

### Q: Should version bumps be part of the feature PR?
**A:** No, keep them separate:
- **Feature PR**: Contains the actual code changes
- **Release PR**: Contains version bumps and changelog updates

### Q: How do I test if packages will install correctly?
**A:** Run `bin/release-helper.sh verify` which:
1. Creates a clean virtual environment
2. Installs each package from PyPI
3. Verifies imports work correctly

### Q: What if I only want to release some packages?
**A:** The release tools are interactive and let you:
- Select specific packages to bump versions
- Choose which packages to tag
- Pick individual packages to publish

### Q: Can I test publishing before going to PyPI?
**A:** Yes! Use `bin/publish-pypi.sh --test` to publish to TestPyPI first.

### Q: How are release notes generated?
**A:** The `dk release-notes` command:
- Analyzes commit messages since the last tag
- Categorizes by type (Added, Changed, Fixed, Breaking)
- Groups by package
- Updates `docs/changelog.md`

### Q: What if the publish fails with "package already exists"?
**A:** This means the version was already published. You need to:
1. Bump the version number
2. Create a new tag
3. Try publishing again

### Q: Do docs deploy automatically?
**A:** Yes, documentation is automatically deployed to GitHub Pages when changes are merged to main.

### Q: What's the difference between all the scripts?
**A:** Here's what each does:
- **`dk`** - Main developer tool with shortcuts
- **`release-helper.sh`** - Comprehensive release automation
- **`validate.sh`** - Code validation checks
- **`fix.sh`** - Auto-fix code issues
- **`build-packages.sh`** - Build distribution files
- **`publish-pypi.sh`** - Upload to PyPI
- **`tag-releases.sh`** - Create git tags

### Q: How do I handle dependencies between packages?
**A:** The build and publish scripts automatically handle packages in dependency order using the `package-discovery.sh` utility.

### Q: What commit message format should I use?
**A:** For better release notes generation, use conventional commits:
- `feat:` or `add:` for new features
- `fix:` or `bug:` for bug fixes  
- `docs:` for documentation
- `chore:` for maintenance
- `BREAKING:` or `!:` for breaking changes

### Q: Can I customize the release process?
**A:** Yes! All scripts support individual steps:
```bash
# Run steps individually
dk release-check        # Just check changes
dk release-bump         # Just bump versions
dk release-notes        # Just generate notes
bin/tag-releases.sh     # Just create tags
bin/publish-pypi.sh     # Just publish
```

## Best Practices

1. **Always run `dk pr` before creating a PR** - Ensures quality checks pass
2. **Use semantic versioning** - Be consistent with version bumps
3. **Write clear commit messages** - They become your release notes
4. **Test with TestPyPI first** - For major releases or if unsure
5. **Keep feature and release PRs separate** - Cleaner history
6. **Tag after merging to main** - Ensures tags point to merged code
7. **Verify installations work** - Catch issues before users do

## Troubleshooting Guide

### Problem: "Working directory has uncommitted changes"
**Solution:** Commit or stash changes before tagging:
```bash
git add .
git commit -m "Your message"
# or
git stash
```

### Problem: "Tests failing in CI but not locally"
**Solution:** Reproduce CI environment locally:
```bash
dk pr  # Runs same checks as CI
```

### Problem: "Import fails after pip install"
**Solution:** Check dependencies:
1. Verify all dependencies in `pyproject.toml`
2. Ensure dependency packages were published first
3. Check for missing `__init__.py` files

### Problem: "Can't publish - authentication error"
**Solution:** Set up PyPI authentication:
```bash
# Create ~/.pypirc with your token
# OR set environment variable:
export UV_PUBLISH_TOKEN='pypi-...'
```

### Problem: "Version conflict between packages"
**Solution:** The scripts handle dependency order automatically, but ensure:
1. All interdependent packages are released together
2. Version constraints in `pyproject.toml` are compatible