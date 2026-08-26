# Changelog

All notable changes to the dataknobs (legacy) package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
