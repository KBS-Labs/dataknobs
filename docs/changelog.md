# Changelog

All notable changes to Dataknobs packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release - 2026-02-21

### dataknobs-bots [0.4.8]

#### Added
- enhanced artifact registry to support content field filtering
- conversational intent detection for wizard state transition
- an artifact corpus abstraction
- wizard transform helpers for corpus operations
- a generic rate limiter

#### Fixed
- fixed serialization bugs
- fixed async deficiencies
- fixed wizard initialization from config
- fixed wizard state tracking and flow

### dataknobs-common [1.3.1]

#### Added
- a generic rate limiter

#### Fixed
- fixed serialization bugs

### dataknobs-data [0.4.8]

#### Added
- a dedup checker utility

### dataknobs-fsm [0.1.10]

#### Changed
- updated documentation
- miscellaneous fixes and small enhancements

#### Fixed
- refactored to leverage the common package's generic rate limiter
- fix to pass function reference params to transform functions

### dataknobs-llm [0.3.6]

#### Added
- a parallel llm executor

#### Fixed
- fixed disconnected rate limit checking
- refactored to leverage the common package's generic rate limiter
- fixed incomplete async fsm integration layer


## Release - 2026-02-17

### dataknobs-bots [0.4.7]

#### Added
- added conversational intent detection for wizard state transitions


## Release - 2026-02-16

### dataknobs-bots [0.4.6]

#### Added
- added a summary memory option
- added deterministic code generators to be used (and eventually created) by bots
- added artifact provenance and rubric evaluation
- added rubrics extraction


## Release - 2026-02-14

### dataknobs-utils [1.2.4]

#### Fixed
- fixed transitive dependencies

### dataknobs-xization [1.2.5]

#### Fixed
- fixed transitive dependencies

### dataknobs-data [0.4.7]

#### Fixed
- fixed intermittent test failures
- fixed transitive dependencies

### dataknobs-bots [0.4.5]

#### Fixed
- fixed transitive dependencies


## Release - 2026-02-11

### dataknobs-common [1.3.0]

#### Added
- added standalone transition validation functionality in common for general use
- promoted configurable retry logic utilities from fsm to common for general use

### dataknobs-fsm [0.1.9]

#### Fixed
- promoted configurable retry logic utilities from fsm to common for general use

### dataknobs-llm [0.3.5]

#### Fixed
- fixed to properly handle kwargs

### dataknobs-bots [0.4.4]

#### Added
- configbot toolkit

#### Changed
- enhanced tool dependency resolution


## Release - 2026-02-09

### dataknobs-data [0.4.6]

#### Fixed
- linting errors

### dataknobs-fsm [0.1.8]

#### Fixed
- fixed faulty divergent path bug

### dataknobs-bots [0.4.3]

#### Added
- lm context generation and transition data derivation features


## Release - 2026-02-06

### dataknobs-bots [0.4.2]

#### Added
- wizard subflow support
- templated wizard responses
- stage label support
- per-stage `extraction_scope` override
- schema-aware data normalization

#### Fixed
- consistent wizard metadata on all response paths
- wizard state reset on restart
- settings injection from wizard config
- template response persistence through serialization

### dataknobs-fsm [0.1.7]

#### Added
- subflow engine support
- multi-transform arc execution

#### Fixed
- tuple truthiness handling in condition evaluation
- exec() scope bug
- subflow network stack popping on completion

### dataknobs-llm [0.3.4]

#### Fixed
- floating point precision in schema extraction numeric fields


## Release - 2026-01-29

### dataknobs-llm [0.3.3]

#### Added
- added assumption tracking in SchemaExtractor

### dataknobs-bots [0.4.1]

#### Added
- added artifacts, reviews, task injection, focus guards, and config versioning enhancements


## Release - 2026-01-28

### dataknobs-config [0.3.4]

#### Added
- added template variable substitution utility

#### Changed
- updated documentation

### dataknobs-utils [1.2.3]

#### Fixed
- fixed ruff errors

### dataknobs-llm [0.3.2]

#### Added
- added testing utilities
- adding missing close methods

### dataknobs-bots [0.4.0]

#### Added
- add ReAct reasoning to wizard reasoning
- strip schema defaults, and add skip-default handling
- adds for auto-ingestion
- adding missing close methods

#### Changed
- updated documentation
- improved hardcoded/default prompt

#### Fixed
- fixed ruff errors


## Release - 2026-01-23

### dataknobs-common [1.2.1]

#### Fixed
- Tightened dependencies

### dataknobs-xization [1.2.4]

#### Fixed
- Tightened dependencies

### dataknobs-fsm [0.1.6]

#### Added
- Added observability functionality

### dataknobs-llm [0.3.1]

#### Added
- Observability functionality
- Context injection into tools

#### Fixed
- Fixed missing optional dependencies

### dataknobs-bots [0.3.1]

#### Added
- Observability functionality
- Custom function resolution


## Release - 2026-01-14

### dataknobs-bots [0.3.0]

#### Added
- Dynamic Registration
  - DataKnobsRegistryAdapter for pluggable config storage
  - CachingRegistryManager with TTL and event invalidation
  - Hot-reload infrastructure (HotReloadManager, RegistryPoller)
  - HTTPRegistryBackend for REST API config sources
  - Knowledge storage backends (InMemory, File, S3)
  - KnowledgeIngestionManager for file→vector ingestion
- Wizard Reasoning
  - Wizard Reasoning Strategy for FSM-backed guided conversational flows
  - WizardFSM - Thin wrapper around AdvancedFSM with wizard-specific conveniences (navigation, stage metadata, state serialization)
  - WizardConfigLoader - Translates user-friendly wizard YAML to FSM configuration at load time
  - WizardHooks - Lifecycle hooks for stage events: on_enter, on_exit, on_complete, on_restart, on_error
  - Navigation Commands - Built-in support for "back"/"go back", "skip", and "restart" navigation
  - Stage Features - Per-stage prompts, JSON Schema validation, suggestions, help text, can_skip, can_go_back, and stage-scoped tools
  - Response Metadata - Wizard progress tracking, current stage info, and available actions
  - Two-Phase Validation - Extraction confidence check followed by JSON Schema validation with graceful degradation
  - State Persistence - Wizard state stored in ConversationManager.metadata for cross-turn persistence

#### Changed
- integration and factory update

#### Fixed
- fixed self-deprecation warnings in tests

### dataknobs-llm [0.3.0]

#### Added
- SchemaExtractor - LLM-based structured data extraction from natural language using JSON Schema
- ExtractionConfig - Configuration for extraction provider, model, and confidence threshold
- ExtractionResult - Result object with extracted data, confidence score, and validation errors
- Multi-Provider Support - Extraction works with Ollama (dev), Anthropic, and OpenAI providers
- Per-Stage Model Override - Stages can specify different extraction models for varying complexity

### dataknobs-common [1.2.0]

#### Added
- EventBus abstraction with Memory/Postgres/Redis backends

### dataknobs-structures [1.0.5]

#### Fixed
- Updated pyparsing API calls to use non-deprecated names (nested_expr, parse_string)

## Release - 2026-01-05

To all packages except legacy, added py.typed markers to enable PEP 561 type checking support for downstream consumers.
Patched versions:
- dataknobs-common [1.1.3]
- dataknobs-config [0.3.3]
- dataknobs-structures [1.0.4]
- dataknobs-utils [1.2.2]
- dataknobs-xization [1.2.3]
- dataknobs-data [0.4.5]
- dataknobs-fsm [0.1.5]
- dataknobs-llm [0.2.4]
- dataknobs-bots [0.2.6]

## Release - 2025-12-26

### dataknobs-xization [1.2.2]

#### Added
- JSON chunking
- Knowledge base ingestion

### dataknobs-data [0.4.4]

#### Added
- Hybrid search types
- Backend hybrid search integration

### dataknobs-bots [0.2.5]

#### Added
- RAGKnowledgeBase hybrid search enhancements


## Release - 2025-12-16

### dataknobs-config [0.3.2]

#### Added
- multi-layered environment-aware configuration support

### dataknobs-bots [0.2.4]

#### Added
- multi-layered environment-aware configuration support
- BotRegistry enhancements
- Deprecated BotManager -- use BotRegistry instead


## Release - 2025-12-15

### dataknobs-data [0.4.3]

#### Added
- Added pgvector backend

### dataknobs-llm [0.2.3]

#### Added
- Implemented per-request LLM overrides

### dataknobs-bots [0.2.3]

#### Added
- Implemented per-request LLM overrides
- Added copy() method to BotContext

## Release - 2025-12-13

### dataknobs-common [1.1.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-config [0.3.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-structures [1.0.3]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-utils [1.2.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-xization [1.2.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-data [0.4.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-fsm [0.1.4]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-legacy [0.1.1]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-llm [0.2.2]

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.

### dataknobs-bots [0.2.2]

#### Added
- Connected streaming responses; Added middleware access to full response.

#### Fixed
- Fixed version sync:ing between pyproject.toml and __version__ attributes.


## Release - 2025-12-08

### dataknobs-common [1.1.1]

#### Added
- Added testing utilities

### dataknobs-data [0.4.1]

#### Fixed
- Validation constraint fixes

### dataknobs-llm [0.2.1]

#### Changed
- Prompt definition and management enhancements

### dataknobs-bots [0.2.1]

#### Changed
- Leverage the LLM package prompt enhancements
- Added progress tracking and logging middleware
- Added a Multi-Tenant Bot Manager
- Added API exception and dependency management
- Added examples, documentation, and tests

## Release - 2025-11-05

### dataknobs-bots [0.1.0]

#### Added
- created new bots package

### dataknobs-llm [0.1.1]

#### Fixed
- fixed option types and logging for the OllamaProvider

## Release - 2025-11-04

### dataknobs-llm [0.1.0]

#### Added
- created new llm package

### dataknobs-xization [1.1.0]

#### Added
- added markdown chunking utilities

#### Changed
- updated documentation

#### Fixed
- fixed ruff and mypy validation errors; moved md_cli.py to xization/scripts

### dataknobs-data [0.3.2]

#### Fixed
- fixed get_nested_value bug for metadata fields
- fixed intermittent test failures

### dataknobs-fsm [0.1.2]

#### Changed
- moved llm modules and llm-based examples to the llm package

## Release - 2025-10-08

### dataknobs-data [0.3.1]

#### Changed
- Dependency security updates
- Fixed psql backend construction to accept connection_string
- Fixed sql search results to include record storage_id
- various lint and test fixes 

### dataknobs-fsm [0.1.1]

#### Changed
- Dependency security updates

#### Fixed
- updated documentation


## Release - 2025-09-20

### dataknobs-fsm [0.1.0]
- Initial Release

### dataknobs-data [0.3.0]

#### Added
- Fixed ID management in filters; added 'NOT_LIKE' operator
- Enhanced `upsert` method signature to accept just a Record object
  - All database backends now support `upsert(record)` in addition to `upsert(id, record)`
  - Automatically uses Record's built-in ID management (storage_id > id field > generated UUID)
  - Maintains full backward compatibility with existing code

#### Changed
- Enhanced upsert to take just a record and use its ID

#### Fixed
- Fixed to properly skip tests in the absence of services.
- Fixed to properly address services from within the development docker container.

## Releases - 2025-08-31

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
