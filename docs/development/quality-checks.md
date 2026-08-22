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

Those hashes are taken **before the first check runs**, and re-taken at the end.
The two are different questions and only one of them is the one you want asked:
hashing at the end records whatever is on disk when the run finishes, so a file
checked at minute 1 and edited at minute 2 is attested as its *new* content —
CI then compares disk against the record, finds them equal, and accepts an
artifact describing code no check ever read. Hashing at the start makes that
edit surface as the ordinary "content has changed since quality checks were
run", which is the message you want and a failure rather than a pass.

The re-check at the end compares the two. If they disagree the tree moved while
the run was reading it, no digest describes what was actually checked, and the
run stops without writing an artifact:

```
✗ The tree changed while the checks were running:
  - packages/data
  - workspace/workspace_tests
✗ No digest describes what was checked, so no artifact was written.
✗ Re-run once the tree is settled.
```

Editing your working tree during a `bin/dk pr` is the usual cause. Let it finish,
or re-run after you stop.

If the *initial* hash computation fails outright, the gate stops there rather
than at the end — an empty digest set compares clean against anything, so
signing an artifact over one would attest a comparison that never happened. The
diagnostics tier attests nothing, so `bin/dk check` degrades instead: it warns
that a half went un-re-checked and finishes the run.

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

### How the summary is produced

Each check appends one record to `check-records.jsonl` in the run's output
directory, at the point that ran it — or at the arm that deliberately skipped
it. `bin/quality-summary.py build` merges those records with the run's metadata
and writes the document; `bin/quality-summary.py render` prints the terminal
banner's check rows from the finished document, so the two cannot disagree.

The split is the fix for a defect that shipped twice. The summary used to be a
shell heredoc over status variables initialised near the top of the script, and
**a status variable has a default, and the default is a verdict**: `0` renders
as `"pass"`, so a check no code path assigned reported as one that ran and
passed. There is no default to fall through to now — a check that writes no
record simply is not in the document, and an absent entry cannot be read as a
passing one.

The banner had the same defect one layer further out, and kept it a phase
longer: on any pull request that changed no documentation it printed three
`✓ PASSED` rows beside a summary recording `skipped: true` for all three. It
renders from the document now, and takes only presentation arguments — which
rows to group, not what any row says.

`check-records.jsonl` stays on disk beside the summary. It is git-ignored in
both tiers, and it is worth reading when a run aborted before its summary was
written: it holds every verdict the run had reached.

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

That prints one record per line with fields separated by ASCII Unit Separator
(`\037`) — `CHECK<US>name<US>status<US>skipped<US>exit_code<US>tool<US>label`,
plus an `OVERALL` record and `META` records. The separator is not a tab
because a tab is IFS *whitespace*, and shell `read` collapses runs of it and
discards empty fields, so a check with no `tool` recorded shifted every later
field one place left and blanked the row.

`--read-summary` returns before the rest of the script, which is what makes it
cheap — and also means it exercises none of the validation below it. To run
that part over a summary of your own:

```bash
bin/validate-quality-artifacts.sh --from /path/to/some-run
```

Use it when you want to know what CI would say about an artifact set. The
projection alone cannot tell you: for a while the reader was correct and the two
lines that consumed it were not.

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
| `toolchain` | root `pyproject.toml`, `uv.lock`, `conftest.py`, `pytest.ini`, `.python-version`, and the three scripts that *are* the lint and test steps | every package needs re-validation |
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

## The quality contract

`.dataknobs/quality-contract.json` declares, for each of three tools, which
files it covers and how far from clean each part of the tree is allowed to be.
It is a **ceiling, not evidence**: no run produces it, CI never signs it, and
moving a number is a deliberate visible diff rather than something a rerun does
on your behalf.

Each cell names a path, a tier, a ceiling and a reason:

| Tool | Tiers | Ceiling counts |
|---|---|---|
| `ruff` | `checked` | findings |
| `mypy` | `strict` / `transitional` / `unchecked` | findings |
| `format` | `enforced` | files the formatter would rewrite |

Two properties make it a ratchet rather than a list of excuses, and both are
enforced rather than described:

**Totality.** Every tracked first-party `*.py` lands in exactly one cell per
tool. A file in no cell is one nobody decided about — the state `bin/` was in
for as long as this repository has had a linter, outside every lint invocation
with nothing saying so. A file in two cells is a decision that contradicts
itself.

**Ceilings are compared, not read.** The declaration this replaced recorded its
counts in comment prose, which is enforced in one direction only: an entry
matching nothing failed, while "241 findings" stayed green at 400. A number
nobody compares is one that stops being true without anyone finding out.

**A backlog is frozen, not excused.** Every ceiling equals what the tree
measures, so a cell holding one cannot *grow* — adding a file with a new finding
fails the `contract` check even where the tier's name suggests it is being
tolerated. `transitional` freezes a type-checking backlog at its current size;
it does not licence another. That is the ratchet working: a phase that clears
one finding while another arrives never ends. Write new files clean, or clear
one of the existing findings in the same cell.

**A tier nothing uses is struck.** Ruff has one tier. `deferred` was deleted
once its last cell emptied, and `verify` now fails a declared tier that no cell
holds, so the same thing happens to mypy's `transitional` and `unchecked` when
their turn comes without anyone having to remember.

That is a ratchet, not bookkeeping. Un-covering a directory takes two edits —
drop it from the tool's target set, and re-file its cell in a tier that
tolerates a backlog — and neither is a fault alone, because the two agree with
each other.

What the retreat costs differs by tool. Ruff is measured in one pass over the
whole population regardless of tier, so the gate keeps measuring a retreated
cell and only the *local* half goes: `bin/validate.sh` stops reading the
directory and `bin/fix.sh` offers no remedy, so a pre-push run reports clean
over territory the gate still checks. mypy is worse — an `unchecked` cell is not
measured at all.

It used to compound that by reporting a measurement of 0 against a ceiling of 0,
indistinguishable from a cell that is genuinely clean, which made the retreat
invisible in the artifact as well as unopposed in the tree. `check` now reports
such a cell as `"measured": null` and lists it under `unmeasured`, and its
closing line says how many cells went unread rather than claiming every cell is
within its ceiling. `null` rather than a flag beside the zero, because the zero
was the defect: it summed into any total a reader built, silently and low.

That applies to mypy and not to the other two, because it is the *measurer* that
decides whether a tier silences a cell — mypy is pointed at a target set the
tiers choose, while ruff and the formatter are handed the whole population and
tally it per cell. So a re-tiered ruff cell is still measured, and reporting
`null` for it would be the same defect pointed the other way: an absence
invented where there was a measurement. Which tools read the tier is declared in
`_TIER_GATED_TOOLS`, and pinned against the measurers by a test rather than
restated.

That makes the retreat legible; it does not make it opposed. The declaration is
still the authority on what gets measured, and a cell re-filed into an
unmeasured tier has still stopped being checked — which is why the tier itself
has to go.

A tier that cannot be spelled cannot be the first step. Striking it is not
sufficient on its own, though — a single change that re-adds the word *and* uses
it passes `verify` — so
`test_every_lint_cell_is_one_the_linter_actually_reaches` closes the rest by
comparing every ruff cell against the target set without consulting its tier.

**Every ruff cell is `checked` at a ceiling of zero**, so the linter reads every
tracked first-party `*.py` and finds nothing. The remaining backlog is the type
checker's. One consequence is visible below: the guards over
`bin/quality-contract.py` need a cell that measures *something*, and there is no
longer one in this repository — hence
[the purpose-built cell](#the-purpose-built-cell).

```bash
# Measure the tree against every ceiling (the `contract` check the gate records)
uv run python bin/quality-contract.py check

# Just the declaration's shape — total, well-formed, no stale cells. Milliseconds.
uv run python bin/quality-contract.py verify

# One tool at a time
uv run python bin/quality-contract.py check --tool mypy

# Which cell does each file land in?
uv run python bin/quality-contract.py partition --tool ruff
```

When you clear findings, lower the ceilings you cleared:

```bash
uv run python bin/quality-contract.py update-baseline
```

That command **only lowers**. A cell measuring above its ceiling is reported —
as a warning naming the cell and both numbers — and then left alone, because
raising one is how a backlog grows during the phase that is supposed to be
clearing it, and doing it by rerunning a command is how that happens without
anyone deciding to. Raising a ceiling is a hand edit, so the argument for it
lands in a pull request where someone can read it.

**Lowering it is not optional.** `check` fails on a cell that measures *below*
its ceiling, in the same class as one that measures above it, and prints the
`update-baseline` invocation that resolves it:

```
mypy/packages/data/src is under its ceiling: 545 findings against 549 declared,
so 4 of headroom is left standing. Write the progress down with:
    uv run python bin/quality-contract.py update-baseline --tool mypy --cell packages/data/src
```

Headroom is a regression budget nobody voted for. Four findings of slack left
in a cell is four a later change can reintroduce with every run in between
reporting green, because nothing the check compares will have moved. So the
zero-headroom rule — *a cell that falls below its ceiling is re-baselined in
the same pull request that lowered it* — is enforced rather than remembered.

Two things follow from the arithmetic rather than from a flag. It can only fire
on a cell whose ceiling is above zero, since nothing measures below zero — which
is the `transitional` mypy cells and nothing else, every other cell in the
declaration being pinned at zero. And a scoped run stays scoped:
`bin/validate.sh` asks only about the cells its targets name, so work in one
package is not failed by a merge that lowered another. The whole-tree catch is
the gate's.

The check never rewrites the declaration itself. Auto-lowering would take the
one diff that records progress out of the pull request where somebody reads it,
and a checker that edits what it is measuring against is the shape this harness
refuses everywhere else.

When a ceiling *is* breached, `check` names the files under it, most findings
first, so a count you cannot act on does not send you to a second tool:

```
format/tests exceeds its ceiling: 21 findings against 20 allowed
    tests/test_deep_merge_agreement.py (1)
    tests/test_docs_mirror_check.py (1)
    ... and 11 more
```

### Asking what one file owes: `charge`

A ceiling is a whole-cell property, so every command above answers about cells.
`bin/validate.sh` handed a single filename deliberately measures the whole cell
that file is in, for the same reason. That leaves two questions with no command
behind them, and they are the two a per-file convention is made of: *what does
this file owe?* and *have I paid it?*

That convention is `.claude/rules/touched-file-cleanup.md`: a change that opens
a file clears what it owes and lowers the ceiling in the same pull request,
unless the file owes more than 25. This command is how the rule stays
self-checkable instead of becoming a matter of estimate.

```bash
# One file, or a directory of them
uv run python bin/quality-contract.py charge --tool mypy packages/data/src/dataknobs_data/query.py
uv run python bin/quality-contract.py charge --tool mypy packages/fsm/src/dataknobs_fsm/patterns

# Machine-readable
uv run python bin/quality-contract.py charge --tool mypy <path> --json
```

The cell is still measured **whole**, exactly as `check` measures it; only the
display is filtered. So the number a file is charged is a term of the sum its
ceiling is compared against, and not a second measurement that could disagree
with the first. The cell's own total is printed beside it so that distinction
stays visible:

```
mypy — 3 finding(s) charged to 1 path

<path>/file_processing.py: 3 of N in packages/fsm/src (ceiling N), over 1 tracked file(s)
  <path>/file_processing.py: 3
      <path>/file_processing.py:110: error: Return type "dict[str, Any]" ...  [override]
```

The cell's total is written `N` on purpose. This repository has published a
documentation page carrying counts nobody could date, which is the reason the
census prints the conditions it was taken under; a live ceiling quoted here
would go stale the first time somebody lowered it, and say nothing. The charge
— the `3` — is the number the command exists to produce.

For mypy the message lines come with it, filtered to the same paths. ruff and
the formatter are read from JSON that *is* the tally, so those report counts and
say that they have no message text to quote — rather than printing an empty
block, which would read as a clean file.

Three ways of naming a path are **refused** rather than answered, because the
whole value of this command is that `0` means *paid* and each of these would
render as `0` meaning something else:

| Named | Why a zero would be wrong |
|---|---|
| A path in no cell, or a directory spanning several | There is no single ceiling its findings count toward |
| A path in a tier the tool is not pointed at | Nothing measured it, so its zero is a silence |
| A path naming no tracked `*.py` | A typo would otherwise report the file as clean |

`charge` reports; it does not judge. It exits 0 whatever it finds, because a
command run *before* doing the work must not be confusable with the failure it
was run to avoid.

### Reading the record: `ledger`

Every command above measures the tree as it stands. `ledger` measures nothing —
it reads the declaration at past revisions and reports what its own history
records:

```bash
uv run python bin/quality-contract.py ledger --tool mypy
uv run python bin/quality-contract.py ledger --tool ruff --json

# Only what happened after a boundary — a revision, or a YYYY-MM-DD day
uv run python bin/quality-contract.py ledger --tool mypy --since <sha>
uv run python bin/quality-contract.py ledger --tool mypy --since 2026-08-22
```

```
mypy ledger — N at <sha> (<date>) to N now, over N merge(s)

cleared N over N of N merges (N% paying, N% paying nothing), mean N per merge
raised 0  — no cell has ever ended higher than it started

by population — the convention is the second row, and only the second row
  leg               N over N of N merge(s)
  convention        N over N of N merge(s), mean N per merge, N% paying nothing
```

Every count is written `N` for the same reason the charge sample above is: each
one moves with the next merge, and a live figure quoted on this page would go
stale without anyone finding out. The `0` is not a placeholder — a raised
ceiling has never happened here, and that is a claim the command re-checks over
the whole history every time it runs.

No new artifact backs this. The contract is committed, so `git log` over it
already *is* the time series — and a per-run file would be a snapshot where the
question is a series, conflicting on every run for the trouble.

Five properties are worth knowing before quoting a number out of it.

**The unit is a merge, not a commit.** Measured over this repository's history on
the day the command was written: 21 paying events out of 66 read per commit,
against 11 out of 67 read per first-parent step, because a pull request that
moves a ceiling twice is one pull request. Since the figure being reported is a
*rate over pull requests*, counting per commit double-counts exactly the
population it describes.

**Ceilings are compared per cell, never by sum.** Cells get added, removed, and
split. When a glob cell covering the per-package test trees was replaced by one
cell per package, the ceilings fell by 255 by sum while the cells present in
both revisions moved by 13 — so a sum-and-subtract reading credits the redraw
with clearing 242 findings nobody cleared. Structural movement is reported in
its own section and never counted as progress.

**A raised ceiling is reported, not netted.** Summing signed deltas would let a
cell gaining 40 and a cell losing 40 report as a quiet zero, and *no cell ends
higher than it started* is a property this is meant to be able to answer.

**Each population carries its own rate.** A drain achieved entirely by scheduled
cleanup would satisfy a threshold read off the total line while falsifying
everything that total was quoted to show, so the mean and the idle fraction are
printed per population rather than left to be divided out. Note the two
denominators are different: `merges` is the population, `paying` is the part of
it that moved a ceiling.

**A window boundary is absolute, or refused.** `--since` takes a revision or a
`YYYY-MM-DD` day, and nothing else. `4 weeks ago` names a different window every
time it runs; `HEAD@{4.weeks.ago}` resolves against *this machine's* reflog, so
the same window read elsewhere is a different window. The day form is pinned to
midnight here rather than handed to git, because git fills the fields an
approxidate leaves out from *now*: measured against this repository at 17:20, a
bare `--since=2026-08-22` reported **0** of the 4 merges made that day. A
boundary older than the declaration opens at its first appearance instead, and
the report says that it did.

`ledger` reads `Quality-Leg:` trailers over the commits each merge brought in —
not off the merge commit, which carries the branch's subject and none of its
trailers — to split deliberate cleanup from incidental. **Presence is the
discriminator**, so a misspelled value still counts as a leg; what it loses is
the record of *which* cell the work went to, which is why a value naming no
declared cell is reported as a fault. The command still exits 0: the commit is
already merged, so there is nothing for a failing status to block.

A leg that moved no ceiling never appears among the steps, since it never
touched the declaration — but it is still not ordinary work, so it is counted
in the leg population's `merges` and left out of the convention's denominator.

### Reading a backlog: `census`

A ceiling is denominated in findings, so `check` answers *whether* a cell is
over budget and — since the file names are carried out of the measurement —
*where*. It has never answered *what*. A cell at 657 might be one mechanical
omission repeated six hundred times or six hundred separate judgements, and
those have entirely different plans.

`census` answers that from the same run:

```bash
# Every finding in every measured cell, broken down by the rule it names
uv run python bin/quality-contract.py census --tool mypy
uv run python bin/quality-contract.py census --tool ruff

# One cell at a time, and machine-readable
uv run python bin/quality-contract.py census --tool ruff --cell packages/data/tests
uv run python bin/quality-contract.py census --tool mypy --json
```

The output lists every cell the run covered — including the ones that measured
nothing, because "this cell is clean" and "the run never reached this cell" are
different facts and a table showing only what was found renders them
identically:

```
mypy census — N finding(s) under pyproject.toml

per cell
  packages/<name>/src: N  (transitional, ceiling N)
      assignment: ...
      arg-type: ...

per rule, across the cells above
  assignment: ...
```

"The cells above", not every cell read: the type checker follows imports, so a
scoped run reads well past the cells it was pointed at. Those findings are
attributed to the cell they are in and left out of this total, which is the
right scoping and would be the wrong heading for it.

For the same reason, naming a cell in a tier no tool reads is refused rather
than answered — the run would leave it out of the table, and a cell missing from
a census cannot be told from one that measured nothing. The refusal names the
flag below that reads it.

Two flags widen the question, and both are type-checker-only — `census` refuses
them for `ruff`, which has neither an unmeasured tier nor per-module
configuration sections, rather than accepting them and quietly doing nothing:

| Flag | What it changes |
|---|---|
| `--include-unmeasured` | reads the `unchecked` cells too. Their ceiling of zero is not a measurement of zero — nothing points the type checker at them, and `verify` insists the ceiling stays zero precisely so that nothing reads their silence as a count |
| `--without-overrides` | measures with the `[[tool.mypy.overrides]]` sections that relax strictness over **first-party** code removed. The sections waiving missing stubs for third-party libraries are left in place: their findings are the absence of type annotations in somebody else's library, which is neither our backlog nor ours to fix |

Which modules count as first-party is read from `mypy_path` rather than listed,
so a package added to the workspace is covered on the day it appears. Each
pattern in a section is classified on its own: a section naming both kinds, or a
pattern that does not begin with a module name, is refused rather than resolved
— removing it would measure somebody else's missing stubs as our backlog, and
keeping it would leave our own strictness relaxed through a run taken to remove
exactly that.

The stripped configuration is generated per run as `.mypy-census.toml` at the
repository root — it has to be at the root, because `mypy_path` is a list of
*relative* paths and the same file elsewhere resolves a different tree — and
deleted afterwards. A file already at that path is refused rather than
overwritten: two censuses in one checkout share it, and the first to finish
deletes it out from under the second. It is also gitignored, so an interrupted
run leaves a stray file rather than a second type-checker configuration for a
later commit to pick up. That run gets its own cache under `.mypy_cache/census`,
so a measurement taken under a configuration nobody has adopted cannot be served
from a cache populated under the declared one.

Three things a census is not:

- **Not a verdict.** A census that ran exits 0 however large the backlog it
  found. The one command whose purpose is to read a backlog would otherwise look
  like a failing check, and a caller would learn to ignore its status. A census
  that *refused* — an unusable contract, a flag the tool has no use for — exits
  non-zero, because that one is a report about the request rather than the tree.
- **Not a ratchet move.** It never touches `.dataknobs/quality-contract.json`.
  A cell measuring under its ceiling is an `update-baseline` decision, made
  deliberately in a pull request; a measurement that also moved the thing it
  measured would leave nobody able to say what the tree looked like beforehand.
- **Not available for `format`.** That tool's unit is files it would rewrite,
  not rules broken, so a per-rule census of it is a category error rather than a
  smaller version of this one. It is refused with that sentence rather than
  returning an empty table, since an empty table is also what a clean tree looks
  like.

Run `uv sync --all-packages` first. A bare `uv sync` under-installs the
workspace and inflates the type checker's findings across several cells, which
reads as a clean tree failing its own gate — and in a census it reads as a
backlog that is not there.

No counts are recorded on this page. A number in prose is a number nobody
compares, which is the failure the contract's ceilings exist to have ended; a
census is re-run at the commit it is quoted against.

### The purpose-built cell

`quality-fixture/` is a small tree that is **clean under `pyproject.toml` and
dirty under its own `quality-fixture/ruff.toml`**, which selects rules this
repository declines to. So it carries a ceiling of zero in the contract like
every other ruff cell, adds nothing to any backlog, and still measures something
when read under its own configuration.

It exists because clearing the last ruff backlog retired this repository as a
test fixture. Several guards over `bin/quality-contract.py` need a cell holding
findings — one inflates a ceiling and checks that `update-baseline` lowers it
back, one pushes a cell under its ceiling and reads the breach report, one
checks that a census and a measurement agree. Over a tree that measures zero
they compare two empty tallies, which agree.

`--contract PATH` is how those guards reach it, and it is the only reason the
option exists: the properties are properties of the *command* — a census must
report a backlog and still exit 0 — so an in-process call, which has no exit
status, cannot check them.

```bash
# What the fixture measures, under the configuration that can see it
uv run python bin/quality-contract.py census --tool ruff \
    --cell quality-fixture/dense --contract <a declaration naming it>
```

The declaration is built by `tests/_workspace.py` from
`.dataknobs/quality-contract.json` — two edits to the real one: the ruff
configuration becomes the fixture's, and the single `quality-fixture` cell
becomes its two halves. Derived rather than written, because `verify` requires a
total partition of every tracked `*.py` and a second hand-kept copy of that
would go stale.

The premise is asserted, in both directions, by
`test_the_purpose_built_cell_is_dirty_to_itself_and_clean_to_the_gate`. Adopt
one of the fixture's rules repo-wide and the failure names the premise rather
than arriving as an unexplained lint error in a directory nobody remembers the
purpose of.

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

# Auto-fix lint findings and formatting
./bin/fix.sh
```

Formatting is checked, not suggested: `bin/validate.sh` fails on a file the
formatter would rewrite. `./bin/fix.sh` is what repairs it, and `bin/dk format`
runs the formatter alone. All three read the root `pyproject.toml`, and all
three resolve their file list from one declaration — `format_targets` in
`bin/package-discovery.sh` — so a green `dk format` means a green format check.

That list is **not** the one the linter uses, and what remains of the difference
is one directory per package. `bin/validate.sh` lints `packages/*/src` and each
package's `tests`, `examples`, `scripts` and `benchmarks`, plus the workspace
directories — every ruff cell is `checked`, so every one of them has to be in
front of the linter. The contract holds `format` to a ceiling of 0 on *all* of
its cells, so the formatter additionally reaches each package's `docs` — a
standing target rather than a live one, since no tracked `*.py` lives under any
of them today. It is there so that a Python file landing in one is formatted
from the day it arrives. Scoping to a package (`bin/validate.sh data`) narrows
both lists to that package.

Each script prints either resolved list without running anything —
`--print-targets` for the linter's, `--print-format-targets` for the
formatter's — which is also how `tests/test_toolchain_consistency.py` checks
them against the contract. Four probes, and the reason there are four is that
each list has a check side and a fix side, and both directions can fail: a
check reading less than the contract enforces is a green verdict over
unexamined files, and a fix reaching less than the check flags is a red gate
with no local remedy.

`./bin/fix.sh --print-targets` is the newest of the four and closes the last of
those corners. A bare `./bin/fix.sh` now resolves **exactly** the linter's list:
promoting `examples`, `scripts` and `benchmarks` widened both sides in the same
change, which is what the contract's ceilings require —
`test_every_lint_ceiling_is_reachable_by_the_fix` fails a cell the check
compares and the fix cannot reach, because a red gate with no local remedy is a
finding a developer has nowhere to take.

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

A check has to make three hops, and `tests/test_quality_gate_accounting.py`
enforces each. It must capture its own exit status; that status must reach
`compute_overall_status`, which decides the local exit code; and it must be
recorded, because CI validates the committed artifacts rather than re-running
the gate, so a check missing from `quality-summary.json` is invisible to CI by
construction. The workflow lint once made none of the three: it printed
`✗ Workflow lint failed` and the gate went on to report `PASS`.

In `bin/run-quality-checks.sh`:

```bash
SECURITY_SCAN_STATUS=0        # beside the other statuses

print_status "Running custom security scan..."
_check_start=$(date +%s)
if run_security_scan; then
    print_success "Security scan passed"
else
    SECURITY_SCAN_STATUS=$?   # hop 1: capture it
    print_error "Security scan failed"
fi
record_check security_scan "$SECURITY_SCAN_STATUS" \
    --tool run-security-scan.sh --duration "$(elapsed_since "$_check_start")"
```

Then add `SECURITY_SCAN_STATUS` to the test in `compute_overall_status` — hop 2.
Assigning `OVERALL_STATUS` directly does nothing: that variable is computed from
the statuses, and recomputed again before the exit code is chosen.

Nothing else needs editing. `record_check` is hop 3, and both readers enumerate
whatever the document holds — the terminal banner derives a label from the
check's name, and `bin/diagnose-quality-failures.sh` offers `bin/<tool>` as the
remedy when `--tool` names an executable there.

A check that can be skipped passes `--skipped true|false` from the arm that
knows, and `--duration null` when it did not run. One that cannot be skipped —
nothing gates it — omits the field rather than always reporting `false`.

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