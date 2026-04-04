# Changelog

All notable changes to Dataknobs packages will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release - 2026-04-03

### dataknobs-common [1.3.8]

#### Added
- added backing for redis

#### Fixed
- more bug fixes


## Release - 2026-04-01

### dataknobs-common [1.3.7]

#### Fixed
- bug fixes

### dataknobs-llm [0.5.5]

#### Added
- migrated create_embedding_provider to the llm package from bots

### dataknobs-bots [0.6.9]

#### Added
- added routing_transforms

#### Fixed
- redesigned wizard generate to separate business logic and extraction from presentation
- fixed load_markdown_text api to be public
- migrated create_embedding_provider to the llm package


## Release - 2026-03-31

### dataknobs-common [1.3.6]

#### Added
- added expression engine abstraction

### dataknobs-bots [0.6.8]

#### Added
- added hybrid reasoning mode, composing grounded and react reasoning
- migrated expression impls to common package's engine abstraction
- added wizard transforms for conditional/logical, collections, regex, and general-purpose


## Release - 2026-03-30

### dataknobs-bots [0.6.7]

#### Added
- added grounded reasoning strategy with configurable search result
  synthesis and deterministic retrieval (PR #216)
- added standalone extraction grounding utility for reuse across
  reasoning strategies (PR #216)
- added per-stage first-render confirmation control for wizard
  flows (PR #215)
- added composite memory fallback and embedding provider factory
  improvements (PR #211)
- added wizard loader validation for stage configuration (PR #211)

#### Fixed
- fixed tool/middleware error propagation and timeout handling in
  turn lifecycle (PR #213)
- fixed process hanging and error swallowing during bot
  creation (PR #214)
- fixed metadata dropping in grounded reasoning pipeline (PR #216)
- fixed thinking mode interference with extraction (PR #216)

### dataknobs-llm [0.5.4]

#### Added
- added extraction grounding utility for validating extracted
  values against field schemas (PR #216)
- added retrieval intent types for structured source
  queries (PR #216)

#### Fixed
- fixed Anthropic messages bug by standardizing LLM adapter pattern
  across all providers (PR #212)
- fixed Ollama provider model matching to be strict (PR #211)

### dataknobs-data [0.4.13]

#### Added
- added grounded source abstraction with database, topic index,
  and cluster index implementations (PR #216)
- added cross-source normalization and result processing
  utilities (PR #216)

#### Fixed
- fixed `LIKE`/`NOT_LIKE` filter operators to be
  case-insensitive (PR #216)

### dataknobs-fsm [0.1.14]

#### Fixed
- fixed async HTTP provider session cleanup to drain SSL transport
  callbacks before event loop shutdown (PR #214)

### dataknobs-config [0.3.8]

#### Fixed
- fixed `substitute_env_vars` to also substitute environment
  variables in dictionary keys

### dataknobs-xization [1.3.0]

#### Added
- added HTML-to-markdown converter with structure-preserving
  table and list handling (PR #209)

#### Fixed
- replaced `chunk_overlap` with priority-based boundary splitting
  in `MarkdownChunker` (paragraph, sentence, word) (PR #220)

### dataknobs-common [1.3.5]

#### Fixed
- testing utility with markdown chunk_overlap parameter removal (PR #220)

### Infrastructure / CI

- added vulnerability auditing with `osv-scanner` (PR #210)
- updated `nltk` and `torch` for CVE remediation (PR #210)
- simplified dependency-update workflow to Python-only (PR #218)
- bumped GitHub Actions in the `github-actions` group (PR #219)


## Release - 2026-03-23

### dataknobs-bots [0.6.6]

#### Added
- added `BotTestHarness` and `WizardConfigBuilder` testing utilities for
  standardized bot test setup (PR #184)
- added `TurnState` per-turn cross-middleware communication and bridged
  LLM + state middleware (PR #186)
- added `from_config` direct injection capability for providers and
  middleware (PR #186)
- added wizard extraction field grounding to validate extracted values
  against field schemas (PR #181)
- added extraction scope escalation strategy for multi-field extraction
  retries (PR #183)
- added wizard extractor field derivations for computed/dependent
  fields (PR #187)
- added enum-based extraction normalization in hints framework (PR #188)
- added extraction recovery pipeline for retrying failed
  extractions (PR #189)
- added custom merge filter protocol for wizard data merging (PR #190)
- added boolean extraction recovery with negation handling (PR #192)
- added security hardening for `context_transform` and summary memory
  injection resistance (PR #186)

#### Fixed
- fixed auto_advance and override logic and landing stage extraction
  from transition messages (PR #175)
- fixed `skip_extraction` lifecycle and stale `_message` injection in
  wizard reasoning (PR #176)
- fixed `store_trace` and `verbose` forwarding through ReAct wizard
  reasoning (PR #177)
- fixed wizard extraction from polluted prompts by managing raw user
  content (PR #178)
- fixed partial wizard data accumulation across multi-turn
  extraction (PRs #179, #180)
- fixed strategy tools gap — reject non-enum values in tool
  registration (PR #191)
- unified hook migration, deprecating legacy hooks (PR #186)

### dataknobs-llm [0.5.3]

#### Added
- added `turn_data` transient state on `ConversationState` for per-turn
  cross-middleware communication (PR #186)
- added `turn_data` bridging into `ToolExecutionContext` so tools can
  access per-turn plugin data (PR #186)
- added `strict_tools` mode on `EchoProvider` to catch missing tool
  definitions in tests (PR #191)
- added `ConfigurableExtractor` and `scripted_schema_extractor` testing
  utilities (PR #184)

#### Fixed
- improved extraction prompts — explicit omission rules, boolean
  negation handling, better error messages (PR #178)

### dataknobs-fsm [0.1.13]

#### Fixed
- fixed `InMemoryStorage` to use separate databases for history and step
  records, avoiding namespace collisions (PR #185)
- added explicit `owns_databases` parameter on `UnifiedDatabaseStorage`
  for ownership control of injected databases (PR #185)

### dataknobs-config [0.3.7]

#### Fixed
- fixed `substitute_env_vars` to use `os.path.expanduser()` instead of
  `Path.expanduser()`, preventing URL corruption (collapsing `://` to
  `:/`) (PR #185)

### dataknobs-data [0.4.12]

#### Fixed
- fixed PgVectorStore `add_vectors` to upsert all columns (content,
  domain_id, document_id, chunk_index) on ID conflict, preserving
  `created_at` timestamp (PR #185)

### dataknobs-config [0.3.7]

#### Fixed
- miscellaneous bug fixes

### dataknobs-xization [1.2.6]

#### Fixed
- quality review fixes

### Infrastructure / CI

- pinned all GitHub Actions to SHAs for supply chain security (PR #193)
- added Dependabot configuration for automated action updates (PR #193)
- bumped `peter-evans/create-pull-request`, `actions/upload-artifact`,
  `actions/upload-pages-artifact` (PR #204)
- added workflow syntax and pinned SHA validation checks (PR #200)
- updated dependency update workflow to wrap Dependabot PRs in addition
  to the Monday morning schedule (PR #207)


## Release - 2026-03-16

### dataknobs-llm [0.5.2]

#### Added
- added embedding provider factory support for config-driven embedding
provider creation
- added caching embedding provider with pluggable backends (memory, SQLite)
- added provider visibility — summary memory can expose its LLM provider for
  registration

#### Fixed
- fixed SQL dot-notation queries in storage backends
- fixed error handling consistency across chat implementations

### dataknobs-bots [0.6.5]

#### Added
- added pluggable conversation storage via config (`storage_class` key)
- added public wizard advance API for non-conversational wizard progression
- added provider registry on DynaBot for enumerating and managing all
  LLM/embedding providers
- added composite memory strategy combining multiple memory backends
- added embedding provider factory support in memory and knowledge base config

#### Fixed
- fixed error handling consistency across chat and stream_chat

### dataknobs-fsm [0.1.12]

#### Added
- added storage injection — FSM storage backends can be provided externally 
  instead of created internally
- added metadata filtering in query_histories() with dot-notation support

#### Fixed
- refactored AdvancedFSM for shared sync/async execution core, eliminating
  code duplication

### dataknobs-data [0.4.11]

#### Added
- added Postgres database auto-create — databases are created automatically if
  they don't exist

#### Fixed
- fixed SQL dot-notation queries for nested field access in filters


## Release - 2026-03-10

### dataknobs-bots [0.6.4]

#### Added
- added post-stream middleware hook

#### Fixed
- fixed flow to enable wizard message mode behavior (skip states w/out requiring a user response)
- fixed deictic resolution bug
- fixed bug in wizard undo fsm state restoration
- fixed middleware bugs


## Release - 2026-03-09

### dataknobs-utils [1.2.5]

#### Fixed
- resiliency fix for transient elasticsearch errors

### dataknobs-bots [0.6.3]

#### Fixed
- fixed skip navigation and config casing bugs


## Release - 2026-03-06

### dataknobs-common [1.3.3]

#### Fixed
- fixed bugs, including 1 security injection risk

### dataknobs-bots [0.6.2]

#### Added
- Added conversation undo/rewind capability

## Release - 2026-03-05

### dataknobs-common [1.3.2]

#### Added
- added json safety functions and aids for serialization strictness

### dataknobs-config [0.3.6]

#### Fixed
- fixed passing capabilities data through config

### dataknobs-data [0.4.10]

#### Fixed
- fixed async elasticsearch database to override count() for filtered queries

### dataknobs-fsm [0.1.11]

#### Fixed
- improved transition control and data/context management

### dataknobs-llm [0.5.1]

#### Added
- added call tracker utility
- added thinking mode detection
- added LLM capture/replay testing harness support

#### Fixed
- improved conversation management and storage
- improved parallel execution configuration
- fixed provider functionality gaps
- fixed llm message serialization

### dataknobs-bots [0.6.1]

#### Added
- added wizard turn context for separating transient from persistent data
- added greeting for non-wizard bots
- added multi-llm capability validation (e.g., extractor -vs- main llm)
- added bots capture/replay testing utilities

#### Fixed
- fixed fsm context management
- fixed reasoning strategy lifecycle and streaming contracts
- fixed greet initial context


## Release - 2026-03-03

### dataknobs-llm [0.5.0]

#### Added
- added persistence of system prompt overrides to metadata
- added name param to add_message for tool result messages
- added 'tool', 'assistant', and 'function' role support
- added tool_calls to LLMMessage
- added conversation export_to_dict
- added accessor for collecting all conversation nodes

#### Fixed
- fixed chat -vs- chat stream code divergence
- fixed provider tool usage bugs
- fixed to deep-copy tc.parameters in metadata capture to prevent aliasing
- fixed conversation storage bugs

### dataknobs-bots [0.6.0]

#### Added
- added artifact bank abstractions with tools
- added restart_wizard tool
- added wizard artifact catalog lifecycle tools

#### Fixed
- fixed wizard and react reasoning flow, context injection, tools, and bugs
- fixed bugs and sync/async divergences
- fixed conversation metadata update timing
- fix to refresh system prompt on data change through tools


## Release - 2026-02-26

### dataknobs-config [0.3.5]

#### Added
- added validation of $requires against capabilities metadata

### dataknobs-data [0.4.9]

#### Fixed
- resource management fixes

### dataknobs-llm [0.4.0]

#### Added
- improved storage and retrieval for visibility (w/ bots)
- broadened conversation search capabilities
- added llm resource specs and layered enforcement strategies (w/ bots, config)
- added delete conversations by filter
- added metadata accessor
- added a conversation branching helper method
- updated tool-using strategy across llm providers, including deprecation

#### Fixed
- fixes to inject system context variables, including current_date (including performing template rendering without rag — old bug)
- fixed gaps in persisting conversation metadata (w/ bots)
- fixed conversation_id initialization, eliminating wasteful conversation root node (w/ bots)
- fixed to allow injected capabilities
- fixed resource management

### dataknobs-bots [0.5.0]

#### Added
- improved storage and retrieval for visibility (w/ llm)
- added llm resource specs and layered enforcement strategies (w/ llm, config)
- added per-message wizard state snapshots, config validation warnings, and debug logging; updated documentation
- added bot greeting
- added configurable wizard navigation
- added memory bank abstraction

#### Fixed
- fixed dynabot stream_chat to return all information, not just the text — BREAKING CHANGE in return value
- fixed resource leaks (multiple instances)
- fixed gaps in persisting conversation metadata (w/ llm)
- fixed to detect and break duplicate tool calls; fix post-break logic
- fixed resource cleanup bugs (w/ data)
- fixed conversation_id initialization, eliminating wasteful conversation root node (w/ llm)
- fixed wizard reasoning vs conversation manager interface disconnect
- fixed to centralize wizard metadata (across all wizard modes)
- refactored tests to remove bug-obscuring mocks (WizardTestManager)
- fixed state counting bugs using centralized code
- fixed stream_chat's defects/divergence from chat
- fixed conversation tree to properly build branches


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
