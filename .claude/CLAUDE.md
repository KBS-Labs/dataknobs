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
| **DynaBot wizard tests** | **`BotTestHarness`** - preferred single-object test setup (see below) | `dataknobs-bots` |
| **Wizard config building** | **`WizardConfigBuilder`** - fluent builder replacing inline config dicts | `dataknobs-bots` |
| Extraction (bypass LLM) | `ConfigurableExtractor` - scripted results, call tracking | `dataknobs-llm` |
| Extraction (real pipeline) | `scripted_schema_extractor()` - real SchemaExtractor + EchoProvider | `dataknobs-llm` |
| LLM provider | `EchoProvider` - scripted responses, call history | `dataknobs-llm` |
| LLM response fixtures | `text_response()`, `tool_call_response()`, `ResponseSequenceBuilder` | `dataknobs-llm` |
| Provider injection | `inject_providers()` - wire providers + extractors into DynaBot | `dataknobs-bots` |
| Capture/replay | `CaptureReplay` - replay recorded LLM conversations | `dataknobs-bots` |
| RAG adapter | `InMemoryAdapter`, `InMemoryAsyncAdapter` | `dataknobs-llm` |
| Sync database | `SyncMemoryDatabase` | `dataknobs-data` |
| Async database | `AsyncMemoryDatabase` | `dataknobs-data` |
| Vector store | `MemoryVectorStore` | `dataknobs-data` |
| Event bus | `InMemoryEventBus` | `dataknobs-common` |
| Rate limiter | `InMemoryRateLimiter` - sliding window, per-category rates, weighted ops | `dataknobs-common` |
| Pytest markers | `@requires_ollama`, `@requires_faiss`, `@requires_redis` | `dataknobs-common` |

If a new testing construct is needed, **add it to the appropriate dataknobs package** for cross-project reuse.

### Optional-Dependency Backend Testing — MANDATORY

Backends that depend on optional packages (asyncpg for `PostgresEventBus`, pyrate-limiter for `PyrateRateLimiter`, etc.) **must have behavioral tests with the real dependency** — not just import/init tests, and not tests using hand-crafted fakes that bypass real code paths. Import-only tests verify the module loads; fakes verify call structure; neither tells you whether the class actually works.

**Requirements for optional-dependency backends:**

1. **The optional dependency must be available in the dev/test environment** — add it to dev dependencies so behavioral tests can run against the real library
2. **Behavioral tests must call the class's primary methods** (e.g., `subscribe()`/`publish()` for an event bus, `try_acquire()`/`acquire()` for a rate limiter) with the real dependency installed
3. **Use `@requires_package("dep_name")` or service-specific markers** (`@requires_postgres`, `@requires_redis`) so tests skip gracefully when the dependency or service is unavailable
4. **Integration tests with real services** (via `bin/dk up` / `bin/test.sh`) should cover the end-to-end path when the service is available
5. **Do not substitute fakes for the real dependency** — a `FakeConnection` that appends to a list has the same problem as `MagicMock`: it doesn't test real behavior. The `PostgresEventBus` bug (sync fake hiding missing `await`) is the canonical example.

**Anti-pattern:** A class with 300 lines of async methods where the only tests check that the module imports and that `__init__` raises a clean error when the dependency is missing. This guarantees that the first consumer to actually use the class discovers all the runtime bugs.

### DynaBot Testing — MANDATORY Patterns

When writing tests for DynaBot behavior, **always use `BotTestHarness`** — for ALL bot tests, not just wizard tests. The harness handles `from_config()`, provider injection, context creation, tool registration, and middleware wiring.

**Non-wizard example** (tool execution, middleware, streaming, from_config DI, etc.):

```python
from dataknobs_bots.testing import BotTestHarness
from dataknobs_llm.testing import text_response, tool_call_response

async with await BotTestHarness.create(
    bot_config={
        "llm": {"provider": "echo", "model": "test"},
        "conversation_storage": {"backend": "memory"},
        "reasoning": {"strategy": "simple"},
    },
    main_responses=[
        tool_call_response("my_tool", {"query": "test"}),
        text_response("Here are the results"),
    ],
    tools=[my_tool],
    middleware=[my_middleware],
) as harness:
    result = await harness.chat("search for test")
    assert result.response == "Here are the results"
    # For streaming: harness.bot.stream_chat("msg", harness.context)
```

**Wizard example** (use `wizard_config=` and `WizardConfigBuilder`):

```python
from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

config = (WizardConfigBuilder("test")
    .stage("gather", is_start=True, prompt="Tell me your name.")
        .field("name", field_type="string", required=True)
        .transition("done", "data.get('name')")
    .stage("done", is_end=True, prompt="All done!")
    .build())

async with await BotTestHarness.create(
    wizard_config=config,
    main_responses=["Got it!"],
    extraction_results=[[{"name": "Alice"}]],
) as harness:
    await harness.chat("My name is Alice")
    assert harness.wizard_data["name"] == "Alice"
    assert harness.wizard_stage == "done"
```

**Anti-patterns to AVOID in bot tests:**

| Anti-Pattern | Why It's Wrong | Use Instead |
|---|---|---|
| Manual `DynaBot.from_config()` + `EchoProvider` + `BotContext` | Duplicates harness boilerplate, bypasses lifecycle management | `BotTestHarness.create(bot_config=...)` |
| Direct `WizardReasoning()` + `reasoning.generate(manager)` | Bypasses `from_config()`, middleware, raw_content pipeline | `BotTestHarness.create()` + `harness.chat()` |
| `bot._conversation_managers` access | Couples to internal cache implementation | `bot.get_wizard_state()` or `harness.wizard_data` |
| `strategy._extractor = extractor` | Private attribute injection | `strategy.set_extractor(extractor)` or `inject_providers(bot, extractor=ext)` |
| `_get_wizard_state()` / `_get_wizard_data()` per-file helpers | Duplicated internal metadata access | `harness.wizard_stage` / `harness.wizard_data` |
| `MagicMock(spec=ConversationManager)` | Mocks hide integration bugs | `BotTestHarness` creates real bots via `from_config()` |
| Inline 40-line wizard config dicts | Verbose, error-prone, copy-pasted | `WizardConfigBuilder` fluent API |

**Exception:** Tests that verify DynaBot/WizardReasoning internal logic (`_evaluate_condition`, `_can_auto_advance`, transform flows) are legitimate unit tests and may use direct instantiation. These test specific internal methods, not behavioral flows.

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
