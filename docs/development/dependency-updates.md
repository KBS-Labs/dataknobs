# Dependency Updates

DataKnobs uses automated dependency updates with a manual review and validation process to keep packages current while ensuring nothing breaks.

## Automated Update Workflow

A GitHub Actions workflow (`.github/workflows/dependency-update.yml`) runs on a weekly schedule to create dependency update PRs.

### Schedule and Trigger

- **Runs weekly** on Monday at 9:00 AM UTC
- Can also be **triggered manually** via `workflow_dispatch` from the GitHub Actions UI

### What the Workflow Does

1. **Checks out** the repository
2. **Installs uv** via `astral-sh/setup-uv@v3`
3. **Upgrades all dependencies** by running `uv lock --upgrade`, which updates `uv.lock` to the latest compatible versions of all packages
4. **Syncs and runs quick tests** (`uv sync --all-packages` then `bin/dk testquick`) to catch obvious breakage
5. **Creates a pull request** using `peter-evans/create-pull-request@v6` with:
    - Branch: `update-dependencies`
    - Commit message and title: `chore: update dependencies`
    - Auto-delete branch after merge

!!! note
    The workflow runs `dk testquick`, which skips integration tests and coverage. The full validation is done manually during the review process (see below).

## Reviewing a Dependency Update PR

When the `chore: update dependencies` PR appears, follow these steps to review and merge it.

### 1. Check Out the PR Branch

Start from a clean `main` and fetch the PR branch:

```bash
git checkout main
git fetch
git pull
git branch -d update-dependencies
git checkout -b update-dependencies origin/update-dependencies
```

!!! tip
    The `git branch -d update-dependencies` step deletes any leftover local branch from a previous update cycle. If the branch doesn't exist locally yet, the delete will fail harmlessly.

### 2. Sync the Updated Dependencies

Install all updated packages into your local environment:

```bash
uv sync --all-packages --all-groups
```

!!! warning "Stale .venv"
    If the repository's working directory has moved (e.g., you renamed or relocated the repo folder), the `.venv` may contain stale absolute paths. In that case, delete `.venv` and let `uv sync` recreate it:

    ```bash
    rm -rf .venv
    uv sync --all-packages --all-groups
    ```

### 3. Review the Dependency Changes

Use `dk deps` to see a summary of what changed:

```bash
bin/dk deps
```

This compares the current `uv.lock` against `main` and shows:

- **Updated** packages with old and new versions (major version bumps are flagged)
- **Added** packages (new transitive dependencies)
- **Removed** packages (dropped transitive dependencies)

Example output:

```
Dependency changes (current vs main):

Updated (5):
  anthropic               0.79.0 -> 0.83.0
  isort                    7.0.0 -> 8.0.0  MAJOR
  pandas                   3.0.0 -> 3.0.1
  rich                    14.3.2 -> 14.3.3
  transformers             5.1.0 -> 5.2.0

Removed (1):
  old-package              1.2.3

Total: 5 updated, 1 removed, 1 major version bump
```

Pay attention to:

- **Major version bumps** -- these may introduce breaking changes and warrant a closer look at the changelog of the affected package
- **New or removed packages** -- understand why transitive dependencies shifted
- **Large jumps in minor versions** -- may include significant behavioral changes

### 4. Run Full Quality Checks

Start the development services (if not already running) and run the full PR validation suite:

```bash
bin/dk up
bin/dk pr
```

This runs linting, type checking, and the complete test suite (unit + integration) against the updated dependencies.

!!! note
    `bin/dk up` starts Docker services (PostgreSQL, Elasticsearch, LocalStack). Make sure Docker is running before this step. Ollama should also be running locally for LLM integration tests.

### 5. Commit the Quality Validation Artifacts

The `dk pr` command generates quality artifacts in `.quality-artifacts/`. Commit them so the CI pipeline can validate without re-running all checks:

```bash
git commit -a -m "ran quality checks"
git push
```

### 6. Approve and Merge

1. Review all changed files in the PR on GitHub (the `uv.lock` diff plus the quality artifacts)
2. Approve the PR
3. Merge the PR (the `update-dependencies` branch is auto-deleted after merge)

## The `dk deps` Command

The `dk deps` command is a convenience wrapper around `bin/dep-diff.py` for comparing dependency versions in `uv.lock`.

### Usage

```bash
dk deps              # Compare working tree vs main
dk deps <ref>        # Compare working tree vs any git ref (branch, tag, commit)
dk deps --staged     # Compare staged uv.lock vs HEAD
```

### How It Works

The script parses `[[package]]` blocks from the `uv.lock` file at two git snapshots (or the working tree), extracts `name` and `version` fields, and diffs them. It classifies changes into updated, added, and removed packages, and flags major version bumps.

## Troubleshooting

### Tests fail after dependency update

If `dk pr` fails:

1. Check which tests failed with `dk diagnose`
2. Look at the flagged packages from `dk deps` -- a major version bump is often the culprit
3. Check the upstream changelog for breaking changes
4. Fix any compatibility issues, commit, and re-run `dk pr`
5. If the breakage is significant, consider pinning the problematic package in `pyproject.toml` and opening a separate issue to address the upgrade

### uv sync fails

If `uv sync` fails with dependency resolution errors:

```bash
# Try a fresh lock
uv lock --upgrade
uv sync --all-packages --all-groups
```

If resolution still fails, a dependency conflict was introduced upstream. Check `uv.lock` for conflicting version constraints and resolve in `pyproject.toml`.

### Quality artifacts are stale

If CI rejects the quality artifacts:

```bash
# Re-run full checks
bin/dk pr

# Commit updated artifacts
git commit -a -m "ran quality checks"
git push
```
