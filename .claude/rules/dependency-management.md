# Dependency Management

## Open-Source Selection Criteria

Before adding any new dependency, ALL criteria must be met:

| Criterion | Requirement |
|-----------|-------------|
| **Maturity** | 1.0+ release or 2+ years of production use |
| **Maintenance** | Active commits within the last 6 months |
| **Adoption** | Widely used in the ecosystem |
| **License** | Permissive only **for libraries we import** — see below; a tool we merely execute is governed separately |
| **Scope** | No excessive transitive dependencies |

Every row except **License** applies to development tooling as well; only the
License row splits by category.

**Prefer libraries already in the dependency tree** before adding new ones. Check `pyproject.toml` and `uv.lock` first.

Since dataknobs is infrastructure consumed by many projects, every dependency we add becomes a transitive dependency for all consumers. Be conservative.

## License Rules (Non-Negotiable)

This is especially critical for dataknobs — as a library, the licenses of what
we *ship* propagate to every consuming project.

### Two Categories, Opposite Answers

Copyleft propagates through **linking**, not through **running**. That single
distinction decides everything below, and it splits our third-party software
into two populations that need opposite rules:

| | **Category A — libraries we import** | **Category B — tools we execute** |
|---|---|---|
| What it is | anything reachable by `import` from shipped code: every `[project.dependencies]` entry and every optional extra | a binary or CLI invoked as a subprocess — linters, type checkers, formatters, test runners, the shell itself |
| Propagation risk | **Real.** This is the linking case copyleft is written for. | **None.** GPLv3 §2: *"This License explicitly affirms your unlimited permission to run the unmodified Program."* |
| License bar | **Permissive only** — the lists below | No license bar; the four conditions in *Category B* instead |

**The Permitted / Prohibited lists below are Category A.** They always were —
the prohibition's own wording is *"copyleft that would require open-sourcing
proprietary downstream code,"* which a tool we exec cannot do. This section
states that explicitly so the next person reaching for a GPL dev tool does not
have to re-derive it.

**Category is decided by how we use a thing, not by what it is.** The same
package can fall on either side: `pylint` as an `import` is prohibited; `pylint`
as a subprocess is fine, and is already in the dev group.

### Permitted Licenses

MIT, Apache 2.0, BSD (2-clause and 3-clause), ISC, Unlicense, CC0

### Prohibited Licenses

Any copyleft license that would require open-sourcing proprietary downstream code:
- GPL (all versions), AGPL, SSPL, EUPL
- LGPL (when statically linked or boundary is unclear)
- MPL 2.0 (unless strictly file-scoped and verified)

### Rules (Category A)

- **MIT is the gold standard.** If a library is not MIT-licensed, verify its license explicitly before adopting.
- **Transitive dependencies matter.** A permissive library that depends on a GPL library still creates a GPL obligation. Check the full dependency tree.
- **When no permissive alternative exists:** Build it ourselves (in the appropriate dataknobs package) or enhance an existing dataknobs construct. Do not adopt the copyleft dependency.
- **Verify, don't assume, in both directions.** Read the installed metadata
  (`importlib.metadata.metadata(pkg)["License-Expression"]`) rather than
  trusting a recollection — including when the answer would be inconvenient.

### Category B — Tools We Execute

A tool we run over our source is not a dependency of our source. Running a
linter no more licenses the code than compiling with GCC licenses the binary —
and the GCC Runtime Library Exception exists precisely because *linking* the
runtime was the risk while *running* the compiler never was. The GPL does not
claim a program's output, and a linter emits diagnostics, not code.

This is not a new allowance. The toolchain has always contained copyleft tools,
and could not function without them:

| Tool | License | How it is already here |
|---|---|---|
| `bash` | GPLv2 (macOS 3.2) / GPLv3 (Linux 5.x) | executes all 41 `bin/*.sh`, including the entire quality gate |
| `git`, `make` | GPL-2.0 | toolchain |
| `pylint` | GPL-2.0-or-later | root `[dependency-groups] dev`, resolved into `uv.lock` |
| `astroid` | LGPL-2.1-or-later | `pylint`'s dependency |
| `shellcheck` | GPL-3.0-or-later | `bin/lint-workflows.sh`, wired into the gate |

A Category B tool needs no permissive license, but **all four conditions must
hold**:

1. **Never import its API.** `import pylint` from shipped code makes it
   Category A and the derivative-work question becomes real. Exec-only is the
   boundary, and it is the only one that actually matters.
2. **Never in a runtime dependency group.** Not in any package's
   `[project.dependencies]`, not in an optional extra. Dev groups only —
   consumers never install those, so nothing reaches a consumer's environment.
3. **Never vendored or redistributed.** Do not commit the binary, and do not
   bundle it into anything we publish.
4. **If it is ever bundled** into a published artifact (a container image, an
   installer), GPL §6 obligations attach **to that artifact** — a written offer
   of source. This never changes dataknobs' own MIT license; it creates a
   distribution obligation on that one artifact. Prefer a tool the CI runner or
   the developer already has, so the question does not arise.

**Prefer the platform binary over a PyPI wrapper that bundles it** (e.g.
`shellcheck-py`). The wrapper puts a copyleft binary into every resolved dev
environment for no benefit here, and moves the tool one step closer to
condition 3.

**A Category B tool must fail loudly when absent.** A check that skips silently
because its tool is missing reports green while testing nothing, which is worse
than having no check — it also reports success. Gate its presence explicitly and
fail; do not warn and continue.

## Upgrade Over Duplicate

When existing code (in dataknobs or dependencies) handles ~80% of the use case:

- **Extend it** with a parameter or optional behavior — don't create `_v2` variants
- **Add to the existing module** — don't create a parallel one in a new file
- Ensure backward compatibility and test both old and new behavior

This applies within dataknobs packages too: if `dataknobs-utils` has an HTTP helper that's close to what you need, enhance it rather than building a new one in `dataknobs-llm`.

## Anti-Patterns

Stop and revisit the reuse hierarchy if you find yourself:

- **Copy-pasting** from another module and tweaking — extract a shared function
- **Creating `_v2` / `_new` variants** — upgrade the original
- **Wrapping** a library call with no added value — call it directly
- **Duplicating boilerplate** across packages — extract to `dataknobs-common` or `dataknobs-utils`
- **Re-implementing stdlib** — `datetime`, `pathlib`, `itertools`, `functools`, `collections` cover more than you think
- **Adding a dependency** for something a current dependency already handles
- **Duplicating across dataknobs packages** — if two packages need the same utility, it belongs in `dataknobs-common` or `dataknobs-utils`
