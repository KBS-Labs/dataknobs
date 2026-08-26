# Changelog

All notable changes to the dataknobs-structures package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries for v1.0.15 and earlier were reconstructed from the release tags when
this file was added; they record what each release actually changed in this
package, which for most of them was nothing but the version number.

## Unreleased

## v1.0.17 - 2026-08-26

### Changed

- This package's tests joined the linted set at a ceiling of zero, alongside
  the sources that graduated in v1.0.16. No source change and no
  consumer-visible change: this is a maintenance release, cut so the workspace
  carries one version set rather than because anything here changed.

## v1.0.16 - 2026-08-11

### Changed

- **This package's sources are now checked under strict typing with a ceiling
  of zero findings**, graduating from the transitional tier it shared with the
  packages still carrying waivers. The single finding standing in the way was
  fixed rather than waived: the fallback `node_name_fn` that `Tree.build_dot`
  defines when the caller passes none is now annotated `(n: Tree) -> str`,
  matching the signature the parameter already declared.

  This package ships a `py.typed` marker, so what it exports is what a
  downstream type checker reads. The tier is the guarantee that stays true —
  a new unannotated definition now fails the gate instead of spending the
  package's one remaining allowance.

- Removed this package's `.python-version` pin, which still named 3.10 after
  `requires-python` moved to 3.12 in v1.0.6. The two disagreed, and the pin was
  the stale one; the workspace-level version governs. No consumer-visible
  change — `requires-python` is what an installer reads.

### Added

- This changelog. Entries for v1.0.15 and earlier were reconstructed from the
  release tags at the same time, which is what the note above the first entry
  records.

## v1.0.15 - 2026-07-29

## v1.0.14 - 2026-07-20

## v1.0.13 - 2026-07-15

## v1.0.12 - 2026-06-29

## v1.0.11 - 2026-05-26

## v1.0.10 - 2026-05-19

## v1.0.9 - 2026-05-13

## v1.0.8 - 2026-05-06

## v1.0.7 - 2026-04-23

## v1.0.6 - 2026-04-04

### Changed

- **`requires-python` raised from `>=3.10` to `>=3.12`.** Breaking for an
  installation on 3.10 or 3.11, which resolves to v1.0.5 from this release
  onward.

## v1.0.5 - 2026-01-14

### Fixed

- `build_tree_from_string` called `pyparsing`'s deprecated camelCase names
  (`nestedExpr`, `parseString`). Both now use the snake_case spellings
  (`nested_expr`, `parse_string`) that pyparsing 3.x documents.

## v1.0.4 - 2026-01-05

### Added

- **A `py.typed` marker**, so a downstream type checker reads this package's
  annotations instead of treating it as untyped (PEP 561).

### Fixed

- Declared the wheel's package directory, which the build backend needs to
  find `src/dataknobs_structures`.

## v1.0.3 - 2025-12-13

### Fixed

- `__version__` disagreed with the version declared in `pyproject.toml`. The
  two are now synchronized.

## v1.0.2 - 2025-11-19

### Changed

- Expanded the docstrings across `tree`, `record_store`, `document`,
  `conditional_dict`, and the package's `__init__` with worked examples. No
  API change: the same names are exported, with the same signatures.

## v1.0.1 - 2025-08-31

### Changed

- Applied the formatter and lint fixes across the package. No behavior change.

## v1.0.0 - 2025-08-11

### Added

- Initial release: `tree`, `record_store`, `document`, and `conditional_dict`.
