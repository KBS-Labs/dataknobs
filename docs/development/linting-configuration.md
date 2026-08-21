# Linting Configuration Guidelines

## Overview

This page explains **how** the linting and type-checking configuration is
organized and how to interrogate it. It deliberately does not restate **what**
the configuration says: `pyproject.toml` is the authority on which rules are
enforced and why, and `bin/quality-contract.py explain` is how you ask.

That split is the lesson of this page's own history, recorded below rather than
quietly corrected. For what a given package owes today, ask
`bin/quality-contract.py check` — the ceilings in
`.dataknobs/quality-contract.json` are the live work list, and the one
per-package checklist that used to be linked here is now filed under
`packages/data/docs/history/` as the record of a finished cleanup.

Policy for changing any of it: `.claude/rules/lint-policy-authority.md`.

## Error Categories and Decisions

**The classification lives in `pyproject.toml`, beside the rules it classifies.**
Every entry in the `[tool.ruff.lint] ignore` list carries a `[category]` marker
on its own comment, and `tests/test_lint_policy.py` compares the two on every
run. This page used to carry that classification by hand, and the two drifted in
the way only a duplicated declaration can — the section that stood here listed
six rules under "Important Errors to Keep (NOT ignored)" that the configuration
declines (`PLR1714`, `PLR5501`, `PLW2901`, `RUF005`, `TC001`, `TC003`), told you
`UP038` was ignored when ruff has removed that rule outright, and omitted 54 of
the 83 declines entirely. A hand-maintained copy is what the markers replaced,
so it is not restated here.

The four categories:

| Category | Claim | What the guard requires |
|---|---|---|
| `presentational` | no finding can be a behaviour difference | an argument wherever ruff marks one of its fixes *unsafe*, since that is ruff contradicting the claim |
| `covered-elsewhere` | another tool or rule enforces the property | the cover named, and a named rule must be one ruff actually enforces |
| `behavioural` | findings can be real; declined deliberately | the argument, as comment lines under the entry |
| `provisional` | findings can be real and the decline is **not** argued | the count, compared against what ruff reports today |

`provisional` is the category whose total has a target of zero, and it is at
zero: the three entries that held it (`B905`, `PLW2901`, `RUF012` — 80
findings) were read site by site and are enforced.

That is a state to keep rather than a rule against filing there. Declining a
rule you have not read is still legitimate, and this is still where it goes —
the category exists so that doing so is visible and countable rather than
dressed up as an argument. Adding an entry is a decision to state in the pull
request; the alternative to stating it is the same decline filed as
`behavioural` with an invented argument.

The word stays in the vocabulary even with nothing in it, which is the
opposite of what happens to a quality-contract tier when its last cell empties
(see [Quality checks](quality-checks.md)). An empty tier is the first move of
a retreat, so removing it makes that move unspellable; an empty decline
category blocks nothing by going away, and only costs the next unread decline
its honest name.

### Asking about one finding

Do not infer a rule's disposition from this page or from reading the TOML. Ask:

```bash
bin/quality-contract.py explain D203
bin/quality-contract.py explain SIM115 packages/utils/src/dataknobs_utils/xml_utils.py
```

It answers one of four verdicts — reported, declined globally, waived for this
file, or not selected — with the reason attached, and always exits 0. The
enabled set comes from ruff itself rather than from a second reading of the
config, so it cannot disagree with what the linter does.

To see the whole list by category, with what each decline stands in front of:

```bash
bin/quality-contract.py explain --audit --measure
```

## Remaining Important Errors

Enforced rules whose findings are worth reading first when a package has a
backlog. Every code below was checked against the configuration with
`explain`; ask it again rather than trusting this list, which is the same kind
of hand-maintained copy the section above describes going wrong.

### Critical Bugs (Must Fix)
- **F811**: Redefinition of unused variable - Can mask real bugs
- **F821**: Undefined name - Will cause runtime errors
- **PLE0704**: Bare raise not in exception handler - Invalid Python
- **B904**: Raise without `from` in except - Loses exception context

### Code Quality Issues (Should Fix)
- **F841**: Local variable assigned but never used - Dead code
- **F401**: Unused imports (except in __init__.py) - Dead code
- **F402**: Import shadowing - Can cause confusion
- **PLW0127**: Self-assignment - Use # noqa: PLW0127 when intentional for documentation

### Security

**ruff does not check security in this repository.** A "Security" heading here
used to say `S3` — "various security issues" — was always important to address.
`flake8-bandit` (`S`) is in no `select` family, so not one `S` rule can fire,
and the entry named a prefix rather than a rule in any case. It is the same
defect as the `UP038` claim above, in the section that pass did not reach: a
hand-written statement about the configuration that nothing compared against
the configuration.

Security is enforced by review against `.claude/rules/security.md`, and by
`bin/lint-shell.sh` for the shell scripts. Selecting `S` is a config change
with its own argument, not something this page can assert.

## MyPy Type Checking Configuration

### Common Error Categories
- **attr-defined**: Accessing undefined attributes, often on None
- **no-untyped-def**: Missing type annotations
- **union-attr**: Union type attribute access without guards
- **assignment**: Type incompatibilities in assignments
- **arg-type**: Wrong argument types passed to functions
- **no-any-return**: Returning Any from typed functions
- **unreachable**: Dead code - indicates logic errors
- **import-untyped**: Missing type stubs for third-party libraries
- **import-not-found**: Optional dependencies not installed

### Configuration Strategy
1. **Third-party libraries**: Add to ignore list when stubs unavailable
2. **Complex modules**: Relax strictness for gradual migration
3. **Optional dependencies**: Ignore imports for feature-specific libraries
4. **Legacy code**: Use per-module overrides to disable strict checking

### Priority for Fixes
1. **Unreachable code** - Always indicates logic errors
2. **None attribute access** - Add proper type guards
3. **Type mismatches** - Fix as you modify code
4. **Missing annotations** - Add gradually, prioritize public APIs

### How the verdict is reached

`bin/validate.sh` does not count mypy's findings and does not match its output
for the word "error". It asks `.dataknobs/quality-contract.json` which cells its
targets fall in and delegates the comparison to `bin/quality-contract.py`: a cell
fails when it measures **above its ceiling**, not when it measures above zero.
See [MyPy Configuration](./mypy-configuration.md).

For a path in no cell there is no ceiling, so any finding fails, and that verdict
follows mypy's exit status. It used to follow a `grep` over mypy's output
instead — and because the script sets `pipefail` and mypy exits non-zero exactly
when it has findings, a real type error made the pipeline non-zero and sent the
check down its *success* branch. Every type error was reported as "Type checks
passed".

## Running Validation

```bash
# Run linting checks
uv run bin/validate.sh [package-name]

# Run type checking
uv run mypy packages/[package-name]/src

# Show per-rule error counts instead of the errors themselves
uv run bin/validate.sh [package-name] --stats
```

### What runs with no arguments

Every package's `src`, plus the code that belongs to no package: `tests/`
(the workspace guards), `bin/` (the scripts that decide whether a pull
request passes), `src/`, and the root `conftest.py`. Anything outside that
set is declared, with its size, in `DEFERRED_FROM_DEFAULT_LINT` in
`tests/test_toolchain_consistency.py` — which compares the list against every
tracked `*.py`, so a new directory of Python fails there rather than silently
joining the set nothing checks.

`--workspace` adds the no-package half rather than replacing what you named, so
`bin/validate.sh data --workspace` checks `packages/data/src` *and* that set,
and with nothing else named it checks that set alone. The quality gate passes it
on every run that validates anything — narrowing to the changed packages used to
drop this half silently, which meant a pull request touching any package
validated `packages/*/src` and nothing more.

`--print-targets` prints the resolved list and exits without running a check,
which is how the guard in `tests/test_toolchain_consistency.py` learns what is
covered. It asks rather than reading the script, because a target appended
inside a conditional reads as unconditional in the source.

## Shell Scripts

Everything above concerns Python. The repository's own shell scripts — 46 of
them, including every script on the path that decides whether a pull request
passes — are checked separately by `bin/lint-shell.sh`, which runs `shellcheck`
and is a recorded gate check in its own right (`shell_lint` in
`quality-summary.json`).

Run it directly, or as `dk shell`:

```bash
bin/lint-shell.sh                    # check everything
bin/lint-shell.sh --print-targets    # which scripts are checked
bin/lint-shell.sh --print-strict     # which are held to zero findings
bin/lint-shell.sh --print-baseline   # which are held to errors only
bin/lint-shell.sh --print-promotable # baseline scripts that are now clean
bin/lint-shell.sh --check-file bin/dev.sh          # one file, at its own tier
bin/lint-shell.sh --check-file bin/dev.sh strict   # ...and would it pass promotion
```

Naming a tier answers "would this script survive promotion", showing the
findings rather than only the verdict — `--print-promotable` for a single file.
It is also how `tests/test_shell_lint_coverage.py` reaches the code that turns a
tier into a severity floor: the run loop calls the same function, so the tiers
are asserted about the real check rather than a re-implementation of it.

### Two tiers, and they ratchet

| Tier | Held to | Which scripts |
|---|---|---|
| strict | zero findings at `info` and above | the scripts that decide whether a run passes, plus every script already clean |
| baseline | `error` severity only | everything else, while its warnings are worked down |

A baseline script that reaches zero **must** be promoted into the strict tier:
`tests/test_shell_lint_coverage.py` fails while one is eligible. So the deferred
set only ever shrinks, and nobody has to remember to revisit it.

Demoting a script instead of fixing it is not available — the same test pins the
scripts that decide the verdict, and `lint-shell.sh` refuses to run if a strict
entry names a file that does not exist (a stale entry would otherwise drop that
script quietly into the baseline tier).

### Two things worth knowing before you fix a finding

**`mapfile` is not available.** ShellCheck's suggested remedy for `SC2207` —
the most common warning class here — is `mapfile`, which is a bash 4 builtin.
These scripts run on the stock macOS `/bin/bash`, which is 3.2. Use a
`while IFS= read -r` loop instead; there are several in `bin/` to copy.

**Not every finding is a defect.** `SC2086` flags an unquoted expansion, and
several are load-bearing: `VALIDATE_ARGS` holds `"$PACKAGES --workspace"` and
*must* word-split. Quoting it would make `validate.sh` look for a target named
`bots --workspace`, find nothing, and validate nothing while still reporting
success. Those sites carry a `# shellcheck disable=SC2086` with the reason
beside them; add to that comment rather than removing the waiver.

### Why the floor is `info`

ShellCheck's `style` severity is stylistic preference with no defect content
(`SC2001`, `SC2129`, `SC2006`). `info` and above is "this may be a bug" — and
`SC2086` is reported as `info`, so a floor of `warning` would exclude the single
class the check most exists to catch. Raising the floor to `style` is the next
turn of the ratchet, not a separate decision.

## What a package owes today

Not a hand-maintained checklist. Packages used to keep one under
`docs/linting-errors-checklist.md`, and exactly one was ever written — it is
filed under `packages/data/docs/history/linting-and-type-checking/` as the
record of a finished cleanup, and nothing has been added to it since.

The live work list is the contract:

```bash
bin/quality-contract.py check                         # every cell against its ceiling
bin/quality-contract.py census --tool ruff --cell packages/data/tests
```

The ceilings live in `.dataknobs/quality-contract.json`. A ceiling only ever
falls: lowering one is what fixing findings looks like, and raising one has to
be argued for in a pull request rather than absorbed by re-running
`update-baseline`. That ratchet is what makes the numbers a work list rather
than a status report, and it is why nothing here has to be checked off by
hand.
