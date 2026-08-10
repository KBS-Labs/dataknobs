# Quality Checks Process for DataKnobs

## Overview

DataKnobs uses a **developer-driven quality assurance process** where developers run comprehensive tests locally before creating pull requests. This approach ensures code quality while keeping CI/CD pipelines fast and cost-effective.

**Key Principle:** All tests, including integration tests with real services (PostgreSQL, Elasticsearch, LocalStack), must pass locally before a PR can be merged.

## Quick Start

Before creating a pull request:

```bash
# Run all quality checks and produce the artifacts CI verifies
./bin/dk pr
```

This single command will:
1. ✅ Start all required Docker services
2. ✅ Run unit tests
3. ✅ Run integration tests with real services
4. ✅ Check code style and linting
5. ✅ Generate coverage reports
6. ✅ Create artifacts in `.quality-artifacts/`

**Important:** The artifacts must be committed with your PR!

### Checking versus producing evidence

Two roles, and they are deliberately separate:

| Command | Runs the checks | Writes `.quality-artifacts/` |
|---|---|---|
| `./bin/run-quality-checks.sh` | yes, in every mode | **no** |
| `./bin/dk pr` (and `pr-all`, `pr-full`) | yes | yes |

`run-quality-checks.sh` is a checker: it writes its working output to a
temporary directory, removed on exit and kept on failure so the logs stay
readable. Use it to iterate. Only `--emit-artifacts` writes the artifacts
directory, and `bin/dk pr` is the only thing that passes it.

The separation matters because the artifacts are the evidence CI checks the tree
against. A checker that also rewrote them meant the documented remedy for a
failing gate — re-run the checks — was also the command that made the gate stop
disagreeing, whether or not anything had been fixed.

`bin/run-quality-checks.sh --print-output-dir` reports where a given invocation
would write, running nothing.

## Detailed Process

### 1. Development Workflow

During development, you can run tests incrementally:

```bash
# Run only unit tests (fast, no services needed)
uv run pytest packages/*/tests/ -v -m "not integration"

# Run specific package tests
uv run pytest packages/data/tests/ -v

# Run with coverage
uv run pytest packages/*/tests/ -v --cov=packages --cov-report=term
```

### 2. Pre-PR Quality Checks (Required)

Before creating a PR to `main` or `develop`:

```bash
# Ensure Docker is running
docker info

# Run the complete quality check suite and produce its artifacts
./bin/dk pr
```

The gate will:
- Start PostgreSQL, Elasticsearch, and LocalStack containers
- Wait for services to be healthy
- Validate package references across the codebase
- Lint the GitHub Actions workflow files
- Run code validation (syntax, ruff, imports, mypy, print statements)
- Build the documentation and check its version and mirror consistency
- Run all tests (unit and integration)
- Generate artifacts in `.quality-artifacts/`

**Expected output:**
```
═══════════════════════════════════════════════════════════════
                       Quality Check Summary
═══════════════════════════════════════════════════════════════
  Documentation:      ✓ PASSED
  Doc Versions:       ✓ PASSED
  Doc Mirrors:        ✓ PASSED
  Code Validation:    ✓ PASSED
  Workflow Lint:      ✓ PASSED
  Unit Tests:         ✓ PASSED
  Integration Tests:  ✓ PASSED

✓ All critical checks passed!
  Artifacts saved to: .quality-artifacts/
  You can now create your pull request.
```

The documentation lines appear in PR mode only, and a skipped check reads
`⊘ SKIPPED` rather than passing. Every line above corresponds to an entry in
`quality-summary.json`, which is the only thing CI reads — see
[CI Validation](#4-ci-validation).

### 3. Commit the Artifacts

After running quality checks successfully:

```bash
# Add the quality artifacts to your commit
git add .quality-artifacts/

# Commit with your code changes
git commit -m "feat: implement new feature with passing quality checks"

# Push and create PR
git push origin your-branch
```

### 4. CI Validation

When you create a PR, GitHub Actions will:
1. **Validate artifacts exist** - Checks for required files
2. **Verify freshness** - Ensures artifacts are < 24 hours old
3. **Confirm tests passed** - Validates all tests show "PASS" status
4. **Check coverage** - Ensures minimum coverage threshold (70%)

The CI validation is **fast** (< 30 seconds) since it only validates artifacts, not re-running tests.

## Required Tools

| Tool | Used for | If absent |
|---|---|---|
| `uv` | every Python invocation in the gate | fatal |
| Docker | the integration-test services below | fatal, unless tests are skipped or already running in a container |
| `shellcheck` | the workflow `run:` blocks, and the repository's own shell scripts | fatal |

`shellcheck` is a hard requirement rather than an optional extra, and that is
deliberate. CI validates the committed artifacts instead of re-running the gate,
so a run that quietly skipped the analysis would produce an artifact
indistinguishable from one where the linter ran and found nothing:
`environment.json` records no per-check tool availability, and nothing
downstream could tell the two apart. A check that skips silently reports green
having verified nothing, which is worse than not having the check at all,
because it also reports success.

Install it with `brew install shellcheck` (macOS) or `apt-get install shellcheck`
(Debian/Ubuntu).

Two checks use it — the workflow lint (`workflow_lint`) and the shell lint over
`bin/` and the two root scripts (`shell_lint`). Neither is gated by run mode or
package selection, so this applies
to every local gate run — a single-package `bin/dk check` as much as a full
`bin/dk pr`. It does **not** apply to CI: the workflow there validates the
committed artifacts and never invokes the gate, so no runner needs `shellcheck`
installed.

## Required Services

The quality checks require these Docker services:

| Service | Purpose | Port | Test Type |
|---------|---------|------|-----------|
| PostgreSQL | Database storage | 5432 | Integration |
| Elasticsearch | Search functionality | 9200 | Integration |
| LocalStack | S3-compatible storage | 4566 | Integration |

Services are automatically started by `run-quality-checks.sh`.

### Manual Service Management

If you need to manage services manually:

```bash
# Start all services
docker-compose up -d postgres elasticsearch localstack

# Check service health
docker-compose ps

# View service logs
docker-compose logs elasticsearch

# Stop services
docker-compose down
```

## Artifact Structure

Quality checks generate these artifacts in `.quality-artifacts/`. Only some are
committed — the rest are local output for your own inspection:

```
.quality-artifacts/
├── quality-summary.json       # committed — the attestation CI validates
├── environment.json           # committed — Python version, OS, git info
├── unit-test-results.xml      # committed — JUnit format test results
├── integration-test-results.xml   # committed
├── style-check.json           # committed — Ruff findings
├── signature.sha256           # committed — checksum of the files above
├── coverage.xml               # local only — see below
├── coverage-*.xml             # local only — per-suite reports
├── htmlcov/                   # local only — browsable coverage
└── test-coverage-summary.txt  # local only
```

`quality-summary.json` is the one that gates a pull request. It carries a
per-package and per-workspace-scope content hash, and CI recomputes those from
the checkout: if they disagree, your artifacts do not describe the code being
merged and the check fails, naming the packages that need re-validation.

### Where the time went

Every check records a `duration_seconds` beside its status, and the run records
a `total_seconds`, so a slow gate can be diagnosed from the artifact instead of
from a stopwatch and an impression:

```bash
python3 -c "
import json; d = json.load(open('.quality-artifacts/quality-summary.json'))
print(f\"total {d['total_seconds']}s\")
for name, c in sorted(d['checks'].items(), key=lambda kv: -(kv[1]['duration_seconds'] or 0)):
    secs = c['duration_seconds']
    print(f\"  {'skipped' if secs is None else str(secs) + 's':>8}  {name}\")
"
```

A check that did not run records `null` rather than `0` — a skipped check has no
measurement, and `0` would claim it ran instantly. `unit_tests` additionally
carries `workspace_guards_seconds`, the share of its span spent on the
workspace guards under `tests/`; those are folded into the unit-test status
rather than reported as their own check, so without that field their cost is
invisible.

That suite is also run with `--durations=10`, because one number for a whole
suite says it is expensive without saying which part of it is. The ten slowest
guards are listed at the end of
`.quality-artifacts/unit-test-output-workspace.txt`:

```bash
grep -A12 'slowest' .quality-artifacts/unit-test-output-workspace.txt
```

Field order within a check no longer matters. `validate-quality-artifacts.sh`
used to read this file with line-offset greps — `grep -A2` for a status,
`grep -A3` for a skipped flag — which made position load-bearing in a format
that has no order: a field added above the one a window wanted pushed it out,
the grep returned nothing, and the validator rejected an artifact it had merely
failed to read. It parses the file as JSON now, so the producer is free to
record fields in whatever order reads best.

To see what the validator makes of a summary without running the rest of it:

```bash
bin/validate-quality-artifacts.sh --read-summary .quality-artifacts/quality-summary.json
```

That prints one tab-delimited line per check — `CHECK<TAB>name<TAB>status<TAB>skipped<TAB>label`
— and is the same reader the validation path uses, so what it shows is what CI
sees.

**Coverage reports are not committed.** The gate's only use for `coverage.xml`
was its line rate, which it reports as a warning and never fails on, so that
number is recorded as `coverage_percent` in the summary instead. Committing a
multi-megabyte generated report to carry one float cost a merge conflict on
every pull request and a large amount of object history.

## Merging main into a branch

Merging or rebasing onto main requires re-running `bin/dk pr`. That is not
avoidable ceremony: main's packages now hash differently than your artifacts
recorded, so your attestation no longer describes the merged tree — and if main
touched a package yours depends on, your suites genuinely have not run against
that code.

What *was* avoidable is resolving a merge conflict in the artifacts first. The
re-run overwrites them wholesale, so nothing you decide during the resolution
survives it. `.gitattributes` marks `.quality-artifacts/**` as `merge=ours` so
git keeps one side intact instead of interleaving them:

```bash
git merge origin/main    # artifacts resolve automatically
bin/dk pr                # regenerates them against the merged tree
git add .quality-artifacts/ && git commit
```

This cannot let a stale artifact through. The hash comparison above runs against
the working tree, so an artifact that was merged but not regenerated fails
exactly as loudly as a conflicted one would have.

A merge driver is a *name*, though — git silently falls back to a normal text
merge when `merge.ours.driver` is unset, and gives no warning that the attribute
did nothing. `bin/dk` configures it on every invocation. If you drive git
without `dk`, run it once yourself:

```bash
bin/setup-git-config.sh
```

## Test Organization

Tests should be organized with pytest markers:

```python
# Unit test (no external services)
def test_data_model():
    assert DataModel().validate() == True

# Integration test (requires services)
@pytest.mark.integration
def test_elasticsearch_query():
    es = Elasticsearch(['localhost:9200'])
    result = es.search(index='test')
    assert result['hits']['total']['value'] >= 0
```

## Troubleshooting

### Services Won't Start

```bash
# Check if ports are already in use
lsof -i :5432  # PostgreSQL
lsof -i :9200  # Elasticsearch
lsof -i :4566  # LocalStack

# Reset Docker services
docker-compose down -v
docker-compose up -d
```

### Tests Pass Locally but CI Rejects Artifacts

Common causes:
- **Artifacts too old** - Re-run `./bin/dk pr`
- **Merged main without re-running** - Expected; re-run and re-commit
- **Forgot to commit artifacts** - Run `git add .quality-artifacts/`
- **Modified artifacts** - Don't edit files in `.quality-artifacts/`

The failure names the packages needing re-validation, or the workspace scope
that changed. There are three, and only the first dirties any package:

| Scope | Covers | Effect when it changes |
|---|---|---|
| `toolchain` | root `pyproject.toml`, `uv.lock`, `conftest.py`, `mypy.ini`, `pytest.ini`, `.python-version`, and the three scripts that *are* the lint and test steps | every package needs re-validation |
| `workspace_tests` | `bin/`, `tests/`, `.github/workflows/`, `.pylintrc`, `run_api.sh`, `setup-dk.sh`, and the data files a recorded check reads — `.gitignore`, `.gitattributes`, `bin/internal-label-allowlist.txt` | artifacts stale, no package dirtied |
| `docs` | `docs/`, `packages/*/docs/`, `mkdocs.yml`, and the two `.dataknobs/` registries the version and mirror checks read | artifacts stale, no package dirtied |

The last two report a changed scope and no package list — nothing they cover can
move a suite's result, only the verdict recorded about it.

A documentation-only change therefore now requires a gate run, where it
previously did not. That is the point: the gate records three checks over the
documentation trees, and until these files were hashed a change to one left every
hash intact, so the stored verdict was accepted over content that had never
produced it — and CI's docs job, which skips its build when no hash is dirty,
declined to rebuild the site as well.

### Integration Tests Fail

```bash
# Check service connectivity
curl http://localhost:9200/_cluster/health  # Elasticsearch
psql postgresql://postgres:postgres@localhost:5432/dataknobs  # PostgreSQL
curl http://localhost:4566/_localstack/health  # LocalStack

# View service logs
docker-compose logs postgres
docker-compose logs elasticsearch
docker-compose logs localstack
```

### Out of Disk Space

```bash
# Clean up old Docker data
docker system prune -a --volumes

# Remove old test data
rm -rf ~/dataknobs_postgres_data
rm -rf ~/dataknobs_elasticsearch_data
rm -rf ~/dataknobs_localstack_data
```

## Configuration

### Linting and Code Style

DataKnobs uses Ruff for linting and code formatting. The project has a carefully configured set of linting rules that balance code quality with practicality. See the [Linting Configuration](linting-configuration.md) documentation for details on:
- Which error types are ignored and why
- How to run linting checks
- Understanding the remaining important errors

To run linting checks:
```bash
# Check specific package
uv run bin/validate.sh data

# Auto-fix formatting issues
uv run ruff format packages/*/src
```

### Type Checking and Python Compatibility

DataKnobs requires **Python 3.12+** and uses modern type hints. See the [Python Compatibility Guide](python-compatibility.md) for important requirements.

**Key requirements:**
- Use modern type hint syntax (`str | None` instead of `Optional[str]`)
- `from __future__ import annotations` is no longer required for that syntax at the
  3.12 floor, but remains useful for forward references and to avoid runtime
  annotation evaluation
- Run type checking with `uv run mypy` to use project dependencies

To run type checking:
```bash
# Check entire data package
uv run mypy packages/data/src/dataknobs_data

# Check specific file
uv run mypy packages/data/src/dataknobs_data/validation/constraints.py
```

> **Historical note.** This page previously recorded a dated snapshot of test
> and mypy-error counts taken against a Python 3.9 floor. Both the floor and the
> counts have since changed; run the commands above for current numbers rather
> than relying on a transcribed total.

### Environment Variables

The quality check scripts respect these environment variables:

```bash
# Maximum age of artifacts for CI validation (hours)
export MAX_AGE_HOURS=24

# Minimum required code coverage (percentage)
export REQUIRED_COVERAGE=70

# Custom pytest markers
export PYTEST_MARKERS="not slow"
```

### Customizing Checks

To add custom checks, modify `bin/run-quality-checks.sh`:

```bash
# Add your custom check
print_status "Running custom security scan..."
if run_security_scan; then
    print_success "Security scan passed"
else
    print_error "Security scan failed"
    OVERALL_STATUS="FAIL"
fi
```

## Benefits of This Approach

1. **Fast CI/CD** - GitHub Actions runs in < 30 seconds vs 5-10 minutes
2. **Real Integration Testing** - Tests run against actual services, not mocks
3. **Cost Effective** - No cloud service costs for every PR
4. **Developer Ownership** - Developers verify their code works before PR
5. **Audit Trail** - Artifacts provide evidence of test execution

## FAQ

**Q: Why not run tests in CI?**
A: Running PostgreSQL, Elasticsearch, and LocalStack for every PR is expensive and slow. Local testing with artifact validation is faster and more cost-effective.

**Q: What if I don't have Docker?**
A: Docker is required for integration tests. You can still run unit tests without Docker: `uv run pytest -m "not integration"`

**Q: Can I skip integration tests?**
A: For PRs to feature branches, you might skip integration tests. For PRs to `main`, they're required.

**Q: How do I add a new service?**
A: Add it to `docker-compose.override.yml`, update `bin/run-quality-checks.sh` to wait for it, and document it here.

**Q: What if artifacts are accidentally modified?**
A: The signature check reports it, and the content hashes in `quality-summary.json` are what actually fail the build. Re-run `./bin/dk pr` to regenerate valid artifacts.

**Q: Do I have to re-run checks after merging main?**
A: Yes. Your artifacts attest to a tree that no longer exists, and CI compares their hashes against the merged checkout. You do *not* have to resolve an artifact merge conflict first — see [Merging main into a branch](#merging-main-into-a-branch).

## Summary

1. **Before PR:** Run `./bin/dk pr`
2. **Commit:** Include `.quality-artifacts/` in your commit
3. **CI Validates:** Artifacts are checked automatically
4. **Merge:** Only if all checks pass

This process ensures high code quality while keeping CI/CD fast and economical!