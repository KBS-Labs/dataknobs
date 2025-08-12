# Changelog

All notable changes to the dataknobs-config package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-12

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