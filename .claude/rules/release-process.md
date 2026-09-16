# Dataknobs Release Process

## Overview

Dataknobs uses a structured release process with version tracking and documentation synchronization.

## Version Management

### Source of Truth

Package versions are managed in two locations that must stay synchronized:

1. **Package pyproject.toml files**: `packages/*/pyproject.toml` - Each package's version
2. **Central registry**: `.dataknobs/packages.json` - All package versions in one place

The `release-helper.sh` script keeps these synchronized during version bumps.

### Every Package Bumps, In One Sitting

> **A release moves every package's version, not only the ones that changed.**

This is not one shared number. The packages sit on independent lines —
`dataknobs-common` in 3.x, `dataknobs-structures` in 1.0.x,
`dataknobs-utils` in 2.x — and collapsing them would throw away more than it
carries. What is shared is the **moment**: when a release is cut, every package
is bumped, tagged and published together, so there is such a thing as "the
workspace as of this release" rather than nine lines a consumer has to
reconcile from a lock file.

A package with nothing to release still takes a patch bump and still gets a
tag. That is not a courtesy to it; it is what makes the set coherent. The
precedent is written into a changelog already — `packages/structures/CHANGELOG.md`
records v1.0.17 as *"a maintenance release, cut so the workspace carries one
version set rather than because anything here changed"*.

**So a breaking change waits.** Describe it under `## Unreleased` in that
package's changelog and leave every version file alone; the major bump is taken
at the next release along with everything else. Do not cut one package early to
take its bump sooner — the dependency layers make that worse than waiting.
`packages/structures` is one layer above `common`, with six packages above it,
so a consumer who takes its new major alone is on a combination nothing in the
workspace was tested against.

**The helper cannot do this yet, and this is the step that gets skipped.**
`bin/release-helper.sh bump` builds its menu from the packages with commits
since their own last tag and offers only those. A package with nothing to
release is never listed, and naming it directly is refused —
`Error: Package 'X' not found or has no changes`. Until that is fixed, bump
those by hand in both locations named above, then `uv lock`, then
`bin/docs-update-versions.sh`. Before tagging, confirm that **every** package's
version moved; `release-helper.sh tag` only tags the ones that did, so a package
missed here is silently left behind.

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

It offers **only** packages with commits since their last tag. The rest still
have to be bumped — see [Every Package Bumps, In One
Sitting](#every-package-bumps-in-one-sitting) — by hand, for now.

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
