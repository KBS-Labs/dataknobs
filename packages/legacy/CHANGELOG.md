# Changelog

All notable changes to the dataknobs (legacy) package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v0.2.2 - 2026-09-02

### Changed

- **The pinned sibling versions move to the ones released today**:
  `dataknobs-common` to 3.2.0 and `dataknobs-xization` to 2.2.1.
  `dataknobs-structures` and `dataknobs-utils` are unchanged. No source change
  here beyond `__version__`, and nothing a consumer of this package calls
  behaves differently.

  This package pins its siblings with `==` rather than `>=`, so its version
  tracks theirs whether or not its own code moves — which is what most of its
  releases have been. The pin is not cosmetic this time: `dataknobs-data`
  0.11.0 requires `dataknobs-common>=3.2.0`, so a v0.2.1 still asking for
  `dataknobs-common==3.1.0` cannot be resolved in an environment that also
  holds a package from this release. Installing `dataknobs` alongside any
  freshly released sibling needs this version.

## v0.2.1 - 2026-08-26

### Fixed

- **`from dataknobs.<pkg>.<module> import Name` now resolves — the import form
  pre-split code actually contains.** Each shim re-exported a modular package's
  submodules by importing them, which binds them as attributes and is enough for
  `from dataknobs.structures import tree`, but not for the dotted form: Python
  resolves a dotted module path through `sys.modules`, not through the parent's
  attributes. The submodules are now registered under the legacy namespace, so
  both spellings work. For a package that exists only to provide backward
  compatibility, the dotted form was the one that mattered most. The shared
  helper lives in `dataknobs._aliasing` because all three shims had the same
  shape and therefore the same gap.

## v0.2.0 - 2026-08-11

### Removed
- `dataknobs.flask_api`, and with it the `flask` dependency it alone required.
  The module imported a `create_app` that this package does not define, so
  `import dataknobs.flask_api` raised `ImportError`; nothing could have depended
  on it.

## v0.1.11 - 2026-07-29

## v0.1.10 - 2026-07-20

## v0.1.9 - 2026-07-15

## v0.1.8 - 2026-06-29

## v0.1.7 - 2026-06-02

## v0.1.6 - 2026-05-20

## v0.1.5 - 2026-05-18

## v0.1.4 - 2026-05-09

### Security
- Bumped minimum `flask` requirement from `>=3.0.0` to `>=3.1.3`
  to exclude versions affected by GHSA-68rp-wp8r-4726.
