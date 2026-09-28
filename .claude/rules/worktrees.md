# Worktrees — parallel branches, one gate

## The rule

> **Develop and validate in as many worktrees as you like. Merge one at a
> time, and a branch's final `bin/dk pr` runs on a branch that already
> contains the current `main`.**

Everything below is how to keep that true without machinery: one shared set
of services, one ordering rule for the final run, and a GitHub setting that
refuses the merge when the ordering was skipped.

## Why the final run needs ordering

**The developer box is the gate, not GitHub.** `bin/dk pr` runs the checks
here and commits `.quality-artifacts/`; CI only *verifies* those artifacts
(`bin/validate-quality-artifacts.sh`). It does so against GitHub's trial merge
of the branch into `main`, comparing content hashes — package source and
tests, plus the workspace, docs and toolchain scopes that
`bin/changed-packages.py` declares — against the hashes the run recorded. A
change on `main` to anything the gate reads therefore makes the artifacts
stale in CI.

**That verification runs only when the branch is pushed.** A merge to `main`
does not re-run it for the pull requests still open. So without an ordering
rule:

1. Branch A passes the gate and merges.
2. Branch B's check went green against the old `main` and stays green.
3. B merges a combination no gate run ever saw.

Working one branch at a time made this impossible by construction, which is
why nothing needed to say it until worktrees made parallel branches ordinary.

**The "Protect main" ruleset now enforces the ordering.** Its required status
check has *require branches to be up to date before merging* on, so the merge
button refuses a branch that does not contain the current `main`. Updating the
branch re-runs CI against the new trial merge, which reports the artifacts
stale until the gate is re-run. Skipping a step fails loudly instead of
merging silently.

## The final run, in order

After any merge to `main`, every other branch that is about to merge goes
through this again. A green check from before that merge says nothing about
the combination.

```bash
cd ../dataknobs-<branch>
git fetch origin
git merge origin/main          # a merge, not a rebase — see below
bin/dk pr                      # syncs packages itself, then the full gate
git add .quality-artifacts && git commit -m "ran quality checks"
git push                       # asking first, as for every push
```

- **Merge, don't rebase, and don't use GitHub's "Update branch" button.**
  `.gitattributes` marks `.quality-artifacts/** merge=ours`, and
  `bin/setup-git-config.sh` sets the `merge.ours.driver` that makes the
  attribute mean anything (`bin/dk` runs it on every invocation). The setting
  lives in `.git/config`, which every worktree shares, so a local merge keeps
  the branch's artifacts without a conflict and the gate run replaces them.
  GitHub's server-side merge ignores custom merge drivers, so its "Update
  branch" button does not get that treatment.
- **`bin/dk pr` runs `uv sync --all-packages` itself**, so a lock file that
  changed on `main` is picked up with no separate step, and a sync that fails
  stops the gate with uv's error rather than running the checks against a
  stale environment. Anything else run
  first in a fresh worktree — `bin/validate.sh`, a targeted `pytest` — needs
  the sync by hand: a bare `uv sync` under-installs, and the extra mypy
  findings read as `main` failing its own gate.
- **Intermediate gate runs need no coordination.** Only the run whose
  artifacts will be merged has to follow this sequence.

## Where a worktree lives

A sibling of the main checkout, named after its branch:

```bash
git fetch origin
git worktree add --no-track -b <branch> ../dataknobs-<branch> origin/main
```

`origin/main` rather than local `main`, which may be behind. `--no-track`
because a branch started from a remote-tracking ref otherwise takes it as its
upstream: `git pull` then pulls `main`, and a bare `git push` refuses because
the names differ. The first `git push -u origin <branch>` sets the right one.

- **Outside the checkout, never inside it**, so nothing the main checkout's
  tooling or `git status` walks can see it.
- **The branch name is the repository's usual descriptive slug**, and the
  directory suffix is the same slug, so either one names the other.

## Services are shared, and owned by nobody

There is one set of dev services (Postgres, Elasticsearch, LocalStack, Redis)
for every checkout. The top-level `name: dataknobs` in `docker-compose.yml` is
what makes a worktree resolve to the same compose project as the main
checkout; before it, a worktree's `manage-services.sh ensure` could not see the
running containers and collided with them on the host ports.

- **A worktree neither starts nor stops services.** `bin/dk pr` finds them
  running and uses them. Start them with `bin/dk up` from any checkout when
  they are down.
- **Leave them running between runs.** The automatic teardown in the test
  scripts has never fired, because the flag it checks is written under a
  different PID. Should it ever be fixed, it must not stop services another
  checkout's run is using.
- **Collisions are avoided by hand, not by locking.** The one known
  interference is Elasticsearch: each test session deletes `test_*` indices
  older than 300 seconds (`DK_ES_TEST_INDEX_MAX_AGE_SECONDS` overrides it),
  so a run in one checkout can remove a long-lived index a concurrent run in
  another still holds. Do not start a run while a long Elasticsearch-heavy one
  is under way in a different checkout.

## Artifacts belong to the worktree that produced them

`.quality-artifacts/` is tracked, so each worktree has its own copy. Commit it
from the worktree the gate ran in; a run in one worktree is evidence about
nothing in another.

Committing code while a run is in progress is harmless as long as the commit
does not change the content being tested. The commit recorded in
`environment.json` is captured when the run starts and is a label only —
nothing checks it — so it will name the parent of the commit you just made.

## Removing a worktree

Once its pull request has merged:

1. Check it holds nothing: `git status --short` is empty and
   `git log --oneline origin/main..HEAD` prints nothing.
2. `git worktree remove ../dataknobs-<branch>` — **not** `rm -rf`. Deleting
   the directory leaves git's worktree record behind, and until
   `git worktree prune` runs, git refuses to check that branch out anywhere
   else.
3. `git branch -d <branch>`, which refuses a branch that is not merged.
4. `git fetch --prune`, to drop the tracking ref of the deleted remote branch.

Gitignored files left in the worktree — caches, `.venv`, run diagnostics under
`.quality-artifacts/` — go with it; none is recorded anywhere else.

## Git operations still ask first

Creating a worktree is local and reversible. Committing, pushing, opening a
pull request, deleting a branch and pruning each need the user to say so, as
they always have — and a yes for one worktree's cleanup does not carry to the
next one.
