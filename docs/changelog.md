# Changelog

All notable changes to Dataknobs packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### dataknobs-data

#### Added
- Enhanced `upsert` method signature to accept just a Record object
  - All database backends now support `upsert(record)` in addition to `upsert(id, record)`
  - Automatically uses Record's built-in ID management (storage_id > id field > generated UUID)
  - Maintains full backward compatibility with existing code
  - Simplifies FSM storage implementation and other use cases

## New Releases - 2025-08-31

### Dataknobs project

#### Added and Fixed
- Global developer tools and project documentation

### dataknobs-common [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-structures [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-xization [1.0.1]

#### Fixed
- Auto lint and formatting fixes

### dataknobs-data [0.2.0]

#### Added
- Added SQLite backend
- Added VectorStore abstraction
  - As an integrated feature in Databases
  - As a stand-alone abstraction

#### Fixed
- All ruff lint and mypy errors

## Releases - 2025-08-18

### Dataknobs project

### Added
- New modular package structure
- `dataknobs-structures` - Core data structures
- `dataknobs-utils` - Utility functions
- `dataknobs-xization` - Text processing
- `dataknobs-common` - Shared components
- Migration guide from legacy package

### Changed
- Migrated from Poetry to uv package manager
- Split monolithic package into focused modules
- Improved test coverage and organization

### Deprecated
- Legacy `dataknobs` package (use modular packages instead)

### dataknobs-data [0.1.0] - Initial Release 🎉

#### Added
- **Multiple Storage Backends**: Memory, File, PostgreSQL, Elasticsearch, and S3 support
- **Async-First Architecture**: Native async/await support with connection pooling
- **Advanced Query System**: Rich operators with boolean logic (AND/OR/NOT)
- **Pandas Integration**: Seamless DataFrame conversion and batch operations
- **Ergonomic Field Access**: Dictionary-style (`record["field"]`) and attribute-style (`record.field`) access
- **Schema Validation**: Built-in validation and migration utilities
- **Streaming Operations**: Efficient read/write streaming for large datasets
- **Factory Pattern**: Dynamic backend selection via configuration
- **Example Projects**: Complete sensor dashboard demonstration app
- **Connection Pooling**: Automatic pool management for PostgreSQL and Elasticsearch

### dataknobs-config [0.2.0]

#### Added
- **Factory Registration System**: Register and manage factories at runtime
  - `register_factory()` - Register custom factory instances
  - `unregister_factory()` - Remove registered factories  
  - `get_registered_factories()` - List all registered factories
- **Cleaner Configurations**: Reference factories by name instead of module paths
- **Runtime Substitution**: Swap factories at runtime (useful for testing)

### dataknobs-utils [1.1.0]

#### Added
- **PostgreSQL Enhancements**:
  - `port` parameter for `PostgresDB` class
  - Parameterized query support in `execute()` method
- **Improved Security**: SQL injection protection via parameter binding

### dataknobs-legacy [0.0.16]

#### Changed
- Updated imports to use new modular package structure
- Improved compatibility layer for smooth migration

### Developer Experience Improvements

#### Added
- **`dk` Developer Tool**: Unified command-line interface for development
  - `dk test` - Run tests with automatic service orchestration
  - `dk quality-checks` - Run comprehensive quality checks
  - `dk docs` - Build and serve documentation
  - `dk build` - Build distribution packages
- **Enhanced Testing Infrastructure**:
  - Automatic Docker service management for integration tests
  - Parallel test execution support
  - Improved coverage reporting
  - Test debugging utilities
- **Documentation Improvements**:
  - Comprehensive package documentation
  - Real-world example projects
  - Migration guides

## Legacy Package [0.0.15] - Pre-2025

### Added
- Initial tools, features, and functionality

---

For more details on each release, see the [GitHub Releases](https://github.com/KBS-Labs/dataknobs/releases) page.
