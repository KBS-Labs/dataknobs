# MyPy Configuration

## One configuration

Type checking is configured in exactly one place: `[tool.mypy]` in the root
`pyproject.toml`. It is strict — `disallow_untyped_defs`, `check_untyped_defs`,
`warn_return_any`, `warn_unused_ignores`, `warn_unreachable` and the rest are all
on.

There used to be a second, `mypy.ini`, which disabled around twenty error codes.
`bin/validate.sh` read that one by default while the quality contract measured
under the root one, so the two answered "is this clean?" differently and the
weaker answer was the one a developer saw locally. A finding introduced in a
package with a backlog passed validation and failed CI. That file is gone.

Run it directly with no arguments beyond the target:

```bash
uv run mypy packages/data/src
```

## How a strict configuration works against a tree that is not strict-clean

The tree is not clean under this configuration, and adopting it did not require
it to be. `.dataknobs/quality-contract.json` splits the repository into *cells* —
one per package source tree, plus `tests`, `bin`, `src` and `conftest.py` — and
gives each a tier and a **ceiling**: the number of findings it is allowed to
have.

| Tier | Meaning |
|------|---------|
| `strict` | type-checked and clean under the root config |
| `transitional` | type-checked, with a backlog that can only shrink |
| `unchecked` | outside the type-checker's target set entirely |

Every ceiling equals what the tree currently measures, so there is no headroom:
new code has to arrive clean, and the only way a ceiling moves is down.

```bash
# What every cell measures, against what it is allowed
bin/quality-contract.py check --tool mypy

# After clearing findings, lower the ceilings to match
bin/quality-contract.py update-baseline --tool mypy
```

`update-baseline` **lowers only**. A cell that measures above its ceiling is left
alone and reported, so the argument for raising one lands in a pull request
rather than in a re-run.

## How `bin/validate.sh` reaches a verdict

It does not count findings itself. It asks the contract which cells its targets
fall in, then delegates the comparison:

```bash
bin/validate.sh utils            # measures the packages/utils/src cell
bin/validate.sh packages/utils/src/dataknobs_utils/file_utils.py
```

The second form measures the whole containing cell, not the one file. A ceiling
is a whole-cell property, so a partial count compared against it would not be a
verdict.

A path in **no** cell — a scratch file outside the repository, say — has no
ceiling to be within, so any finding there fails. Imports are not followed for
those, so the population's backlog does not leak into the verdict.

Paths under a cell the contract marks `unchecked` are reported as skipped rather
than silently dropped.

## Per-module overrides

`[[tool.mypy.overrides]]` sections in the root `pyproject.toml` relax a rule or
waive missing imports for a named module pattern.

**A section that matches no module fails the contract check.** It suppresses
nothing, and is one of two things: a waiver whose spelling is wrong — so the
findings it was written for are still being reported — or one whose subject is
gone. Both read as "handled" to anyone looking at the config. mypy reports it as
a *note*, which leaves the exit status untouched, so `bin/quality-contract.py`
reads the note and fails on it.

Three sections were dead when that check was introduced, and two of them were
waivers for findings still being reported: `dataknobs_legacy.*` (the package's
importable module is `dataknobs`) and `python_nmap.*` (the library actually
imported is `nmap3`, from a different distribution).

A section may legitimately match nothing — an `ignore_missing_imports` override
for a library imported inside a `try/except ImportError`, for instance. Say so in
a comment on the line above and the check accepts it:

```toml
[[tool.mypy.overrides]]
module = [
    # Imported inside a try/except ImportError by bin/check-services.py, as the
    # more accurate of its two PostgreSQL liveness probes. Stubs are not carried
    # for a driver nothing in the workspace depends on.
    "psycopg2.*",
]
ignore_missing_imports = true
```

## Adding type stubs

Prefer stubs to an `ignore_missing_imports` waiver: stubs give real checking,
the waiver gives `Any`. Weigh the addition against the dependency rules in
`.claude/rules/dependency-management.md` — stub packages are ordinary
dependencies.

```toml
[dependency-groups]
dev = [
    "types-requests>=2.31.0",
]
```

## Search path

`mypy_path` in the root config lists the package source trees, so a first-party
import resolves to the source under check rather than to an installed copy.

Both directions fail the same way, and both are guarded in
`tests/test_toolchain_consistency.py`:

- **A declared entry that does not exist** contributes no modules. Every import
  through it falls back to `ignore_missing_imports`, the symbols come back as
  `Any`, and the run still reports success.
- **A type-checked package that is not declared** is the same failure reached
  from the other side: its modules resolve to `Any` wherever they are imported
  by name, so the findings that would have been reported are never computed. The
  package still measures a number, and the number is lower than the truth. The
  guard compares the search path against the mypy cells in
  `.dataknobs/quality-contract.json`, because a package the contract
  type-checks is exactly the population that must be resolvable.

## IDE integration

Point the type checker at the repository root and let it discover
`pyproject.toml`; no `--config-file` argument is needed.

### VS Code

```json
{
  "mypy-type-checker.args": ["--config-file=pyproject.toml"]
}
```

### PyCharm

Settings → Tools → Python Integrated Tools, set MyPy as the type checker. No
extra arguments.

## Current status

- **Python floor**: 3.12 (`requires-python = ">=3.12"`; mypy's `python_version`
  matches it, so the checker evaluates against the same interpreter the project
  runs on)
- **Per-cell counts**: run `bin/quality-contract.py check --tool mypy`

> This page deliberately quotes no error totals. It used to carry several, taken
> against a Python 3.9 floor and a configuration that no longer exists, and they
> outlived every one of those conditions. A number nobody compares is a number
> that stops being true without anyone finding out — which is precisely what the
> ceilings above are for.

## Related documentation

- [Quality Checks](./quality-checks.md)
- [Linting Configuration](./linting-configuration.md)
- [Python Compatibility Guide](./python-compatibility.md)
- [Contributing](./contributing.md)
