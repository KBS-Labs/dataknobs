# Dataknobs Project

General-purpose library providing infrastructure, abstractions, and common functionality for Python projects. **This project IS the shared infrastructure** - enhancements here benefit all downstream projects.

## Core Mandates

These apply to all work in this project. See the global `~/.claude/CLAUDE.md` for full details.

1. **Documentation verification is mandatory** - verify every claim against actual code before finalizing
2. **No workarounds** - fix root causes; temporary mitigations require documented plans
3. **Abstraction-first design** - interfaces/protocols before implementations; configurable via config
4. **Mock prohibition in tests** - use EchoProvider, SyncMemoryDatabase, etc. (see below)
5. **Ollama-first for LLM defaults** - default to local models; commercial providers via config
6. **Security constraints are non-negotiable** - input validation, HTTP safety, path traversal prevention, sensitive data protection (see `rules/security.md`)
7. **Dependency management** - permissive licenses only, selection criteria enforced, no duplication (see `rules/dependency-management.md`)

### This Project's Special Role

When working in dataknobs, remember:
- **New infrastructure belongs here**, not in consuming projects
- If a consuming project needs functionality that doesn't exist, **add it to the appropriate dataknobs package**
- Testing constructs (EchoProvider, memory databases, InMemoryEventBus) live here for reuse across all projects
- Observability gaps should be filled here, not worked around downstream

### Infrastructure Leverage Hierarchy (Within DataKnobs)

1. **FIRST: Use existing code in another dataknobs package** - check before reimplementing
2. **SECOND: Enhance an existing dataknobs package** - extend, don't duplicate
3. **THIRD: Other reliable open-source** - only if not appropriate for dataknobs
4. **LAST: Build from scratch** - in the most appropriate dataknobs package

## Project Structure

UV workspace monorepo (Python 3.10+):

```
dataknobs/
├── packages/
│   ├── common/        # dataknobs-common: Registries, events, exceptions, serialization
│   ├── config/        # dataknobs-config: Config management, environment bindings, factories
│   ├── data/          # dataknobs-data: Database backends (7), vector stores, query, streaming
│   ├── llm/           # dataknobs-llm: LLM providers, prompts, tools, RAG adapters
│   ├── bots/          # dataknobs-bots: Configuration-driven AI agents
│   ├── fsm/           # dataknobs-fsm: Finite state machines
│   ├── structures/    # dataknobs-structures: Core data structures
│   ├── utils/         # dataknobs-utils: File, JSON, HTTP, SQL utilities
│   └── xization/      # dataknobs-xization: NLP normalization/annotation
├── docs/              # MkDocs documentation (site-wide)
└── .dataknobs/        # Version registry (packages.json)
```

See `~/.claude/rules/dataknobs-reference.md` for the complete verified lookup table of all classes and import paths.

## Development Commands

**Always use `bin/dk` for development tasks.**

```bash
# Quality checks
bin/dk pr              # Full PR preparation (required before push)
bin/dk check           # Quick quality check
bin/dk check data      # Check specific package

# Testing
bin/dk test            # Run all tests
bin/dk test data       # Test specific package
bin/dk test --last     # Re-run failed tests only

# Fixing
bin/dk fix             # Auto-fix style issues
bin/validate.sh -f     # Validate + auto-fix all packages
bin/validate.sh data   # Validate specific package

# Documentation
bin/dk docs            # Serve docs locally
bin/dk docs-build      # Build documentation

# Services (integration tests)
bin/dk up / down       # Start/stop dev services (Docker)
```

Run `bin/dk help` for full command reference.

## Testing Constructs (Provided by This Project)

These exist for use by dataknobs tests AND all consuming projects:

| Need | Construct | Package |
|---|---|---|
| LLM provider | `EchoProvider` - scripted responses, call history | `dataknobs-llm` |
| LLM response fixtures | `text_response()`, `tool_call_response()`, `ResponseSequenceBuilder` | `dataknobs-llm` |
| RAG adapter | `InMemoryAdapter`, `InMemoryAsyncAdapter` | `dataknobs-llm` |
| Sync database | `SyncMemoryDatabase` | `dataknobs-data` |
| Async database | `AsyncMemoryDatabase` | `dataknobs-data` |
| Vector store | `MemoryVectorStore` | `dataknobs-data` |
| Event bus | `InMemoryEventBus` | `dataknobs-common` |
| Pytest markers | `@requires_ollama`, `@requires_faiss`, `@requires_redis` | `dataknobs-common` |

If a new testing construct is needed, **add it to the appropriate dataknobs package** for cross-project reuse.

## Before Adding New Functionality

1. **Check if it exists** in another dataknobs package
2. **Determine the right package** for new code (common, config, data, llm, utils, etc.)
3. **Design the abstraction first** - protocol/interface before implementation
4. **Provide a reasonable default implementation** (e.g., in-memory, Ollama-based)
5. **Add tests using real constructs** from this project (not mocks)
6. **Update documentation in BOTH locations** (package docs + MkDocs site docs)
7. **Verify documentation** against the actual code

## Definition of Done

Each task is complete when ALL of the following are true:

1. All tasks implemented and acceptance criteria verified
2. No new code duplicates existing functionality (reuse hierarchy followed)
3. New code is modular, injectable, and reusable
4. All function parameters and return types have type hints (modern syntax: `list[str]` not `List[str]`, `X | None` not `Optional[X]`)
5. Type checker and linter pass with no errors (`bin/dk check <package>`)
6. Tests pass using real constructs, not mocks (`bin/dk test <package>`)
7. Security constraints satisfied (input validation, no leaked credentials, safe HTTP/file ops)
8. No `TODO`, `FIXME`, placeholder comments, or commented-out code left behind
9. Documentation updated in both locations and verified against code
