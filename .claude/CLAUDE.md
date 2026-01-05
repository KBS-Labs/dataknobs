# Dataknobs Project

General-purpose library providing boilerplate and common functionality for Python projects.

## Project Structure

This is a UV workspace monorepo:

```
dataknobs/
├── packages/           # Individual packages
│   ├── common/        # dataknobs-common: Base utilities
│   ├── config/        # dataknobs-config: Configuration management
│   ├── data/          # dataknobs-data: Data stores, vector stores
│   ├── llm/           # dataknobs-llm: LLM integrations
│   ├── bots/          # dataknobs-bots: Bot framework
│   ├── fsm/           # dataknobs-fsm: State machines
│   ├── structures/    # dataknobs-structures: Data structures
│   ├── utils/         # dataknobs-utils: General utilities
│   └── xization/      # dataknobs-xization: NLP/annotation
├── docs/              # MkDocs documentation (site-wide)
│   └── packages/      # Package docs for mkdocs
└── packages/*/docs/   # Package-specific documentation
```

## Development Commands

**Always use `bin/dk` for development tasks** - it ensures proper environment setup.

```bash
# Quality checks
bin/dk pr              # Full PR preparation checks
bin/dk check           # Quick quality check (dev mode)
bin/dk check data      # Quick check specific package

# Testing
bin/dk test            # Run all tests
bin/dk test data       # Test specific package
bin/dk test --last     # Re-run failed tests only
bin/dk testquick       # Fast tests without coverage

# Fixing issues
bin/dk fix             # Auto-fix style issues
bin/dk format          # Format code

# Code validation (ruff + mypy)
bin/validate.sh              # Validate all packages
bin/validate.sh data         # Validate specific package
bin/validate.sh -f           # Validate and auto-fix
bin/validate.sh -f data      # Auto-fix specific package

# Documentation
bin/dk docs            # Serve docs locally (live reload)
bin/dk docs-build      # Build documentation
bin/dk docs-check      # Check for doc issues

# Services (for integration tests)
bin/dk up              # Start dev services
bin/dk down            # Stop dev services
bin/dk logs            # View service logs

# Diagnostics
bin/dk diagnose        # Analyze last failure
bin/dk coverage        # Show coverage report

# Shortcuts: dk t (test), dk f (fix), dk d (diagnose), dk q (quick check)
```

Run `bin/dk help` for full command reference.

## Key Utilities Available

### Data Stores (dataknobs-data)
- `InMemoryDataStore` - Testing and lightweight use
- `FileDataStore` - File-based persistence
- `PostgreSQLDataStore` - Production database
- `S3DataStore` - Cloud storage

### Configuration (dataknobs-config)
- `ConfigLoader` - Load from YAML/JSON/env
- `ConfigValidator` - Schema validation

### Common Patterns (dataknobs-common)
- Base classes and interfaces
- Logging utilities
- Error handling patterns

## Before Adding New Functionality

1. Check if similar functionality exists in one of the packages
2. Consider whether new code belongs in dataknobs or is project-specific
3. If adding to dataknobs, determine the appropriate package
