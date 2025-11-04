# Quality Validation Workflow Design

## Overview

The quality-validation workflow uses path filters to determine when to run checks. This document explains the design decisions and trade-offs.

## Current Behavior

### What Triggers Quality Validation

The workflow runs when PRs contain changes to:

**Code filter** (triggers `validate-quality-artifacts` and `quick-unit-tests`):
- `**/*.py` - Python source files
- `pyproject.toml` - Dependencies
- `uv.lock` - Locked dependencies
- `bin/**` - Developer scripts
- `tests/**` - Test files
- Other configuration files

**Docs filter** (triggers `build-docs`):
- `docs/**` - Documentation files
- `mkdocs.yml` - Documentation config
- `packages/*/src/**/*.py` - Source code (affects API docs)
- `.github/workflows/docs.yml` - Docs workflow itself

### What DOESN'T Trigger Quality Validation

- `.dataknobs/packages.json` - Package registry
- `.dataknobs/` - Registry utilities and docs
- `.github/workflows/` - Workflow files (except docs.yml)
- `.nojekyll` - Jekyll configuration
- Root-level markdown files (except in specific paths)

## Design Decision: Registry Changes Don't Trigger CI

### Why This is Acceptable

1. **Primary Validation is Local**
   - Developers run `dk pr` before submitting PRs
   - `bin/run-quality-checks.sh` includes package reference validation
   - CI validation is a safety net, not the primary check

2. **Most Registry Changes Include Other Changes**
   - Adding a package typically involves:
     - Modifying workflows (to add package install)
     - Modifying docs (to document the package)
     - Modifying README (to list the package)
   - These other changes WILL trigger the validation
   - Pure registry-only PRs are rare

3. **Simplicity**
   - Current workflow structure is simple and clear
   - All validation runs together when code changes
   - Avoids complex conditional logic

4. **Low Risk**
   - Package reference validation is fast and cheap
   - Running it locally is not a burden
   - False negatives (skipped checks) are caught in subsequent PRs
   - False positives (unnecessary runs) waste more CI time

### When This Could Be a Problem

If someone submits a PR that ONLY modifies:
- `.dataknobs/packages.json`
- `.github/workflows/release.yml` (to add package to dropdown)

Then the package reference validation won't run in CI. However:
- This is a rare scenario
- Developer should run `dk pr` locally anyway
- The validation will catch it in the next PR that touches code/docs

## Alternative Designs Considered

### Option A: Split Validation Jobs

Create separate jobs for different validation types:
```yaml
validate-package-references:
  if: code-changed OR registry-changed OR workflow-changed

validate-quality-artifacts:
  if: code-changed only
```

**Pros:**
- More precise triggering
- Package validation always runs when relevant

**Cons:**
- More complex workflow
- More jobs to manage
- Harder to understand
- Risk of misconfiguration

**Decision:** Not worth the complexity for the rare edge case

### Option B: Add Registry to Code Filter (REJECTED)

Add `.dataknobs/**` to the code filter.

**Pros:**
- Simple change
- Always validates package references

**Cons:**
- Would trigger quality artifacts validation unnecessarily
- Would trigger unit tests unnecessarily
- Wastes CI resources
- False signal (checks running when not needed)

**Decision:** Wrong granularity - triggers too much

### Option C: Current Behavior (SELECTED)

Keep the current path filters as-is.

**Pros:**
- Simple and clear
- Appropriate granularity
- Low maintenance
- Relies on local validation (appropriate)

**Cons:**
- Package validation can be skipped in rare cases

**Decision:** ✅ SELECTED - Best trade-off

## Documentation for Developers

When adding a new package, developers should:

1. **Add to `.dataknobs/packages.json`**
2. **Run local validation**: `dk validate-pkgs`
3. **Update files** as indicated by the validator
4. **Run full PR checks**: `dk pr` (includes validation)
5. **Submit PR** - CI will validate if code/docs changed

If your PR ONLY modifies registry files:
- CI validation may be skipped (this is OK)
- Ensure you ran `dk pr` locally
- The next PR with code changes will catch any issues

## Monitoring and Adjustment

If we find that registry-only PRs are:
- Common (they're not expected to be)
- Causing problems (package references getting out of sync)

Then we should revisit Option A (split validation jobs).

## Related Documentation

- Package registry system: `.dataknobs/README.md`
- Adding new packages: `docs/development/contributing.md`
- Quality checks: `docs/development/quality-checks.md`
