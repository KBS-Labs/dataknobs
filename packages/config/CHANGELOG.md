# Changelog

All notable changes to the dataknobs-config package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- `RequiredEnvVarError`, a `ValueError` subclass raised by
  `substitute_env_vars` (and every loader that calls it) when a required
  ``${VAR}`` or ``${VAR:?msg}`` is unset. Carries `var_name`,
  `bash_form`, and `explicit_message` so callers can branch on the
  failure shape without parsing message text. Existing `except
  ValueError` / `pytest.raises(ValueError)` callers keep working.

### Changed
- `EnvironmentConfig.load()` and `EnvironmentConfig.from_dict()` now apply
  `${VAR}` / `${VAR:default}` substitution by default, matching the
  behaviour of `InheritableConfigLoader.load()`. Pass the new keyword-only
  `substitute_vars=False` to opt out (e.g., to inspect raw refs). Required
  `${VAR}` refs without a default raise `ValueError` at load time.
- `substitute_env_vars` is now the canonical environment-variable
  substitution helper across the package. It accepts three keyword-only
  options: `type_coerce` (default `False`; coerce whole-value `${VAR}`
  placeholders to `int` / `float` / `bool`), `expand_user_paths` (default
  `True`; applies `os.path.expanduser` to substituted values), and
  `substitute_keys` (default `True`; substitutes `${VAR}` references in
  dict keys as well as values). The regex recognises bash-style
  `${VAR:-default}` and `${VAR:?error_msg}` in addition to the legacy
  `${VAR:default}` form. `Config._load_dict` calls `substitute_env_vars`
  with `type_coerce=True, expand_user_paths=False, substitute_keys=False`.
- `type_coerce=True` no longer treats `"0"` / `"1"` as booleans. Only the
  unambiguous bool words `true` / `false` / `yes` / `no` (case-insensitive)
  coerce to `bool`; numeric `"0"` and `"1"` coerce to `int`. This affects
  every caller that opts in to `type_coerce=True`, including
  `Config._load_dict`, the `VariableSubstitution` shim, and direct
  `substitute_env_vars(..., type_coerce=True)` callers.

### Deprecated
- `VariableSubstitution` is now a thin compatibility shim over
  `substitute_env_vars(data, type_coerce=True, expand_user_paths=False,
  substitute_keys=False)` and emits `DeprecationWarning` on construction.
  Use `substitute_env_vars` directly. The class will be removed in a
  future release.

## v0.3.11 - 2026-05-06

## v0.1.0 - 2025-01-12

### Added
- Initial release of dataknobs-config package
- Core `Config` class for managing modular configurations
- Support for YAML and JSON file formats
- Atomic configuration management with type/name-based access
- String reference system (`xref:`) for cross-referencing configurations
- Environment variable override system with bash-compatible naming
- Global settings and defaults management
- Path resolution for relative paths in configurations
- Object construction support with class instantiation and factory patterns
- Object caching for improved performance
- Comprehensive test suite with 91% code coverage
- Full type annotations with mypy support
- Detailed documentation and usage examples

### Features
- **Modular Design**: Organize configurations by type with atomic units
- **File Loading**: Load from YAML, JSON, or Python dictionaries
- **Cross-References**: Link configurations using `xref:type[name]` syntax
- **Environment Overrides**: Override any config value via environment variables
- **Path Resolution**: Automatic resolution of relative paths
- **Object Building**: Optional object construction from configurations
- **Settings Management**: Global and type-specific defaults
- **Extensible**: Clean interfaces for custom builders and factories

### Technical Details
- Python 3.8+ support
- Dependency: PyYAML >= 6.0
- Development dependencies include pytest, mypy, ruff, and types-PyYAML
- Follows PEP 8 style guidelines
- 100% type annotated codebase
