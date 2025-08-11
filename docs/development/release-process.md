# Release Process

This guide describes the release process for the Dataknobs project.

## Overview

The Dataknobs project follows a structured release process to ensure quality and consistency across all packages. We use semantic versioning and maintain separate version numbers for each package.

## Release Workflow

### 1. Prepare for Release

Before creating a release:

1. **Update version numbers** in `pyproject.toml` files:
   ```bash
   # Update version in packages/*/pyproject.toml
   # Follow semantic versioning: MAJOR.MINOR.PATCH
   ```

2. **Update changelogs**:
   - Update `CHANGELOG.md` in the root
   - Update package-specific changelogs if applicable

3. **Run validation**:
   ```bash
   ./bin/validate.sh
   ```

### 2. Create Git Tags

Use the tagging script to create version tags:

```bash
./bin/tag-releases.sh
```

This script will:
- Show current package versions
- Allow you to tag individual packages or all at once
- Create annotated git tags in the format `package/vX.Y.Z`

### 3. Build Packages

Build all packages:

```bash
./bin/build-packages.sh
```

Or build individually:
```bash
cd packages/common && uv build
cd packages/structures && uv build
# etc...
```

### 4. Publish to PyPI

#### Test Publishing (Optional)

First, test with TestPyPI:

```bash
./bin/publish-pypi.sh --test
```

#### Production Publishing

Publish to PyPI:

```bash
./bin/publish-pypi.sh
```

The script will:
- Check for authentication (`.pypirc` or environment variables)
- Publish packages in the correct order (common first, legacy last)
- Skip already published versions

### 5. Deploy Documentation

Update and deploy documentation:

```bash
# Build and test locally
./bin/docs-build.sh
./bin/docs-serve.sh  # Preview at http://localhost:8000

# Deploy to GitHub Pages
./bin/docs-deploy.sh
```

### 6. Create GitHub Release

1. Go to [GitHub Releases](https://github.com/yourusername/dataknobs/releases)
2. Click "Create a new release"
3. Choose the appropriate tag
4. Add release notes from the changelog
5. Publish the release

## Version Management

### Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Package Versioning Strategy

- **Independent versions**: Each package has its own version number
- **Coordinated releases**: Major releases are coordinated across packages
- **Dependency updates**: Update inter-package dependencies as needed

### Version Bumping Guidelines

| Change Type | Version Bump | Example |
|------------|--------------|---------|
| Bug fix | PATCH | 1.0.0 → 1.0.1 |
| New feature | MINOR | 1.0.1 → 1.1.0 |
| Breaking change | MAJOR | 1.1.0 → 2.0.0 |

## Release Checklist

Before each release, ensure:

- [ ] All tests pass (`./bin/test-packages.sh`)
- [ ] Code is linted (`./bin/validate.sh`)
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] Version numbers are bumped
- [ ] Dependencies are updated
- [ ] Security vulnerabilities are addressed

## Release Schedule

- **Patch releases**: As needed for critical fixes
- **Minor releases**: Monthly or bi-monthly
- **Major releases**: Quarterly or semi-annually

## Rollback Procedure

If issues are discovered after release:

1. **Yank from PyPI** (if critical):
   ```bash
   # This doesn't delete but marks as "yanked"
   pip install --upgrade twine
   twine yank dataknobs-package==X.Y.Z
   ```

2. **Fix the issue**:
   - Create a hotfix branch
   - Apply the fix
   - Test thoroughly

3. **Release patch version**:
   - Bump patch version
   - Follow normal release process

## Automation

Consider setting up GitHub Actions for:

- Automated testing on PR
- Automated version bumping
- Automated PyPI publishing on tag push
- Documentation deployment

See `.github/workflows/` for CI/CD configuration.

## Troubleshooting

### Common Issues

1. **Authentication failures**:
   - Ensure `.pypirc` is configured correctly
   - Check PyPI tokens are valid

2. **Version conflicts**:
   - Ensure version hasn't been published already
   - Check all package versions are updated

3. **Build failures**:
   - Clear `dist/` directory
   - Check all dependencies are installed

## Related Documentation

- [Contributing Guide](contributing.md)
- [Testing Guide](testing.md)
- [CI/CD Pipeline](ci-cd.md)