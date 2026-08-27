# CI/CD Pipeline

This document describes the GitHub Actions workflows that run against this
repository. Every workflow named here exists in `.github/workflows/`; nothing
below is aspirational.

## Table of Contents

- [The unusual part: tests run on your machine](#the-unusual-part-tests-run-on-your-machine)
- [Workflow inventory](#workflow-inventory)
- [Pull request checks](#pull-request-checks)
- [Post-merge checks](#post-merge-checks)
- [Documentation deployment](#documentation-deployment)
- [Releases](#releases)
- [Dependency updates and CVE auditing](#dependency-updates-and-cve-auditing)
- [Conventions](#conventions)
- [Troubleshooting](#troubleshooting)

## The unusual part: tests run on your machine

**No workflow in this repository runs the test suite.** The integration suites
need PostgreSQL, Elasticsearch, and LocalStack, and standing those up for every
push is slow and expensive. So the heavy work happens locally and CI verifies
that it happened.

You run `bin/dk pr`, which executes the suites and writes an attestation into
`.quality-artifacts/`. You commit that alongside your code. On the pull request,
`quality-validation.yml` re-derives a content hash for every package and
workspace scope from the checkout and compares them against the hashes your
artifacts recorded. If they disagree, your artifacts describe different code
than the branch being merged, and the check fails naming the packages that need
re-validation.

```bash
bin/dk pr                 # changed packages only (the usual case)
bin/dk pr --all           # every package, in parallel
git add .quality-artifacts/
```

Two consequences worth internalizing:

- **A green pull request means your local run was green** and covered the code
  being merged. It does not mean a GitHub runner executed anything.
- **Merging main into a branch requires re-running `bin/dk pr`**, because your
  attestation no longer describes the merged tree.

[Quality Checks](quality-checks.md) covers the mechanism in full, including
what is committed, what is not, and why merging main no longer produces an
artifact conflict.

## Workflow inventory

| Workflow | Trigger | Purpose |
|---|---|---|
| `quality-validation.yml` | pull request → `main`, `develop` | Validates the committed quality artifacts; builds docs |
| `docs-mirror-check.yml` | pull request (doc paths) | Package and site doc trees stay in sync |
| `docs-version-check.yml` | pull request (version paths) | Doc version tables match the package registry |
| `ci.yml` | push to `main` | Post-merge packaging sanity check |
| `docs.yml` | push to `main` (doc paths), manual | Builds and deploys the docs site to GitHub Pages |
| `release.yml` | release published, manual | Builds and publishes packages to PyPI |
| `dependency-update.yml` | weekly Monday schedule, manual | Upgrades Python dependencies and audits for CVEs |

Dependabot (`.github/dependabot.yml`) handles GitHub Actions version updates
separately, grouped into a single weekly pull request.

## Pull request checks

### Quality Validation

`quality-validation.yml` is the gate. It has four jobs:

**`check-changes`** uses a paths filter to decide whether the pull request
touches code or documentation. The `code` filter deliberately includes
`.github/workflows/quality-validation.yml` itself — without that, a pull
request whose only change deletes the filter patterns would match nothing, skip
the sole quality job, and report green.

**`validate-quality-artifacts`** runs when code changed. It executes
`bin/validate-package-references.py`, then `bin/validate-quality-artifacts.sh`,
which:

1. Confirms the required artifact files are present
2. Runs `bin/package-hashes.py validate` — the check that actually gates
3. Reads the recorded check statuses out of `quality-summary.json`
4. Reports the recorded coverage percentage (advisory; never fails the build)
5. Verifies the artifact signature

On failure it comments on the pull request with the command to run locally.

**`build-docs`** runs when documentation changed, and skips its own expensive
steps when `bin/package-hashes.py validate` reports no quality-relevant source
change.

**`all-checks-complete`** always runs and produces the single consistent status
check, failing if either substantive job failed.

### Documentation Mirror Check

Every package doc has a counterpart under `docs/packages/`, and the two must
stay in agreement — silent drift is what once let the rendered site teach an API
that did not exist. `bin/docs-mirror-check.py` verifies each pair against its
classification in `.dataknobs/docs-mirror-manifest.json`. See the
[Documentation Guide](documentation-guide.md) and the manifest itself for the
available classifications.

It also enforces that both trees spell a document the same way. A package doc
must be named `lower-hyphen.md` (`README.md` excepted), and a paired doc must
carry the same filename on both sides — otherwise a bare link to it resolves in
at most one of the trees it is served from, which is how 89 package-tree links
came to be broken while the rendered site stayed clean. Relative `.md` links are
resolved case-**sensitively**, because `Path.exists()` is not: on macOS it
answers *yes* for `configuration.md` when only `CONFIGURATION.md` is present.

Every relative `.md` link must resolve in every tree its document is served
from, and the check fails when one does not. A link broken by spelling names the
rename that fixes it; a link whose target is absent under any spelling names the
three remedies instead — an absolute site URL, publishing the target into the
package tree, or a prose mention — because the two trees nest some documents
differently and some targets are site-native, so no relative path reaches those
from both. See `.claude/rules/dual-docs.md` for which remedy fits which case.

### Documentation Version Check

Package versions live in `pyproject.toml` files and in `.dataknobs/packages.json`.
`bin/docs-update-versions.sh --check` confirms the version tables in
`docs/index.md` and `docs/installation.md` still match. Run it without `--check`
to fix them.

## Post-merge checks

`ci.yml` runs on pushes to `main` and does one thing: `uv build` for every
package. Pull-request validation already proves the packages import and the
tests pass, so this exists solely to catch packaging-only regressions — a file
missing from an sdist, broken `pyproject.toml` metadata — that only surface when
a distribution is actually built.

It deliberately does *not* cancel in-flight runs. When two pull requests merge
back to back, each merge's sanity check completes.

## Documentation deployment

`docs.yml` builds the MkDocs site and deploys it to GitHub Pages. It triggers on
pushes to `main` that touch `docs/**`, `packages/*/src/**/*.py`, `mkdocs.yml`,
the workflow itself, `uv.lock`, or the root `pyproject.toml`. Source files are
included because the API reference is generated from docstrings.

Build it the same way locally before pushing:

```bash
uv run mkdocs build --strict     # warnings become errors
uv run mkdocs serve              # preview at http://127.0.0.1:8000
```

## Releases

`release.yml` triggers when a GitHub release is published, or manually via
`workflow_dispatch` with a package choice (any single package, or `all`) and a
TestPyPI toggle that defaults to on.

Publishing uses PyPI trusted publishing — the job requests `id-token: write` and
uploads with `uv publish`, so no API token is stored in the repository.

The local side of a release (version bumps, tags, release notes) is driven by
`bin/release-helper.sh`. See [Release Process](release-process.md) for the full
sequence.

## Dependency updates and CVE auditing

`dependency-update.yml` runs weekly on Monday and can be dispatched manually. It
upgrades Python dependencies and then audits **two** resolutions with
`osv-scanner`:

- **The upgraded resolve** (`uv.lock`) — the highest versions all workspace
  constraints permit.
- **The floor resolve** (`uv-lowest.lock`) — every direct dependency pinned to
  its declared lower bound, which is what a fresh consumer install can land on
  when no inherited pin forces something higher.

Both sets of findings go into the pull request body. A finding against the floor
resolve means a declared lower bound in some `pyproject.toml` needs raising, not
that the lockfile needs regenerating.

The workflow handles Python dependencies only. Updating workflow files would
require `workflows: write`, which is intentionally not granted; GitHub Actions
versions are Dependabot's job instead.

## Conventions

**Actions are pinned to commit SHAs, not tags.** Every `uses:` reference carries
a full SHA with the version in a trailing comment. A mutable ref (`@v4`,
`@master`) can be repointed if the action's repository is compromised, and the
next run picks it up silently.

**The three pull-request workflows cancel superseded runs**, via a concurrency
group keyed on workflow and ref, so pushing again supersedes the in-flight run.
`ci.yml` and `dependency-update.yml` set `cancel-in-progress: false` — for
`ci.yml` that is the back-to-back-merge case described above. `docs.yml` and
`release.yml` declare no concurrency group.

**Permissions are declared at the narrowest useful scope.** `ci.yml` takes
`contents: read`; `quality-validation.yml` adds `pull-requests: write` for its
failure comment; `docs.yml` adds `pages: write` and `id-token: write` for the
Pages deployment. `release.yml` declares none at the workflow level and requests
`id-token: write` on its publishing job instead. The remaining workflows declare
none and inherit the repository default.

## Troubleshooting

### "Quality Check Validation Failed"

Your committed artifacts do not match the code in the pull request. The job
output names either the packages needing re-validation, or the workspace scope
(`toolchain`, `workspace_tests`, `docs`) that changed. The last two dirty no
individual package, so those cases report a changed scope and an empty package
list.

```bash
bin/dk pr
git add .quality-artifacts/ && git commit --amend --no-edit
```

The usual causes are merging main without re-running, and editing a file after
the run that produced the artifacts.

### The quality job did not run at all

`check-changes` decided nothing quality-relevant changed. Confirm your paths
match the `code` filter in `quality-validation.yml`. The filter is mirrored by
`WORKSPACE_QUALITY_INPUTS` in `bin/changed-packages.py`, and
`tests/test_toolchain_consistency.py` asserts the two agree — a disagreement
there is a bug, not something to work around.

### A documentation check fails on a doc you did not touch

The mirror guard reports every failure in the package it is checking, not only
the ones in your diff — an unclassified doc, a transclude replaced by a hand
copy, a link that does not resolve. Each error names its own remedy; there is no
blanket fix and no `--fix` flag, since a classified pair is one text at two
paths rather than two copies to reconcile. Run it locally for the full list:

```bash
python3 bin/docs-mirror-check.py
bin/docs-update-versions.sh --check
```

### Reproducing a workflow locally

Each job is a thin wrapper over a script in `bin/`, so run the script directly:

```bash
./bin/validate-quality-artifacts.sh      # the pull-request gate
uv run python bin/validate-package-references.py
uv run mkdocs build --strict
python3 bin/docs-mirror-check.py
```

## Resources

- [Quality Checks](quality-checks.md) — the artifact mechanism in full
- [Testing Guide](testing.md) — running suites and services locally
- [Dependency Updates](dependency-updates.md) — the weekly upgrade and audit
- [Release Process](release-process.md) — version bumps through publication
- [dk Command](dk-command.md) — the developer entry point
- [GitHub Actions documentation](https://docs.github.com/en/actions)
