# Linting Configuration Guidelines

## Overview
This document explains the rationale behind the linting and type checking configuration for the Dataknobs project. It serves as a reference for understanding which error types are considered important versus cosmetic, and why certain rules are ignored.

The actual configuration is in `pyproject.toml`. For specific errors that need to be fixed in each package, see the package-specific checklists (e.g., `packages/data/docs/linting-errors-checklist.md`).

## Recent Cleanup (August 2025)
A comprehensive linting cleanup was performed, reducing Ruff errors in the data package from ~40 to 10. Key achievements:
- Fixed all functional issues (undefined names, import shadowing, unused variables)
- Modernized NumPy random generation (NPY002)
- Fixed loop variable overwrites that revealed a bug in vector search
- Moved type-checking imports appropriately (TC001/003/004)
- Established clear guidelines for stylistic vs functional errors

## Error Categories and Decisions

### 1. Important Errors to Keep (NOT ignored)

#### Critical Bugs
- **F811**: Redefinition of unused variable - Can mask real bugs
- **F821**: Undefined name - Runtime errors
- **B904**: Raise without `from` inside except - Loses exception context
- **PLE0704**: Bare raise not in exception handler - Invalid Python

#### Code Quality
- **F401**: Unused imports (except in __init__.py)
- **F402**: Import shadowing
- **B007**: Unused loop variable not prefixed with underscore
- **B027**: Empty method without abstract decorator (use # noqa: B027 for intentional empty implementations)
- **PLR1714**: Consider merging multiple comparisons
- **PLR5501**: Consider using elif
- **RUF005**: Consider iterable unpacking
- **NPY002**: Replace legacy np.random.rand - use modern np.random.default_rng()
- **SIM101**: Multiple isinstance calls - merge for clarity
- **PYI056**: Use += for __all__ modifications - better type checker support
- **PLW2901**: Loop variable overwritten - can indicate logic errors
- **TC001/TC003/TC004**: Type checking imports - proper placement for performance

#### Security
- **S3**: Various security issues

### 2. Ignored Error Categories

#### Whitespace/Formatting (Auto-fixable)
- **W291, W293**: Whitespace issues - Cosmetic, can be auto-fixed
- **E501**: Line too long - Already configured at 100 chars

#### Documentation
- **D105, D107**: Missing docstrings in special methods - Often self-explanatory
- **D200, D415, D417**: Docstring formatting - Minor style issues

#### Type Annotations
- **ANN204**: Missing return type for `__init__` - Always returns None
- **ANN001, ANN003**: Missing type annotations - Often obvious from context
- **ANN201, ANN202, ANN205**: Missing return types - Can be inferred

#### Import Location
- **PLC0415**: Import at top-level - Sometimes needed for:
  - Lazy loading for performance
  - Avoiding circular dependencies
  - Conditional imports

#### Code Simplification (Stylistic Preference)
- **SIM102**: Combine nested if - Sometimes clearer as nested
- **SIM103**: Return negated condition directly - Sometimes clearer with explicit if/else
- **SIM108**: Use ternary operator - Can reduce readability
- **SIM118**: Use `key in dict` instead of `key in dict.keys()` - Explicit .keys() can be clearer
- **PLW3301**: Nested max calls - More readable when nested for complex expressions
- **RUF006**: Store asyncio.create_task reference - Only needed if task cancellation is required

#### Complexity Metrics
- **PLR0911**: Too many returns - Already limited to 6
- **PLR0912**: Too many branches - Already limited to 12
- **PLR0915**: Too many statements - Already limited to 50

#### Unused Arguments
- **ARG001, ARG002, ARG004**: Unused arguments - Often required by:
  - Interface contracts
  - Callback signatures
  - Override methods

#### Type System Updates
- **UP035, UP038**: Modern type syntax - Gradual migration
- **UP028**: Yield from - Not always clearer

## Remaining Important Errors

After configuration, focus on these error types that indicate real issues:

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
- **S3**: Various security issues - Always important to address

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

`bin/validate.sh` fails a run when mypy exits non-zero, and prints whatever mypy
said. It does not match mypy's output for the word "error": it previously piped
mypy into `grep` and tested the pipeline, and because the script sets `pipefail`
and mypy exits non-zero exactly when it has findings, a real type error made the
pipeline non-zero and sent the check down its *success* branch. Every type error
was reported as "Type checks passed". Note that `mypy.ini` disables most error
codes, so the set of findings that can fail a run is narrower than mypy's
default — the config decides what counts, and the exit status decides the
verdict.

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

## Package-Specific Checklists

Each package should maintain its own error checklist documenting specific issues to address:
- Location: `packages/[package-name]/docs/linting-errors-checklist.md`
- Format: Checkbox list organized by priority
- Updates: As errors are fixed, check them off and remove when complete

Example: `packages/data/docs/linting-errors-checklist.md`
