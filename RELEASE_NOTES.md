# Release Notes

For all current and ongoing release notes, see [docs/changelog.md](docs/changelog.md).

---

## Historical: Modular Architecture Migration (2025)

The monolithic `dataknobs` package was split into focused, independent packages:

- `dataknobs-common` — Registries, events, exceptions, serialization
- `dataknobs-config` — Configuration management, environment bindings
- `dataknobs-data` — Database backends, vector stores, query system
- `dataknobs-llm` — LLM providers, prompts, tools, RAG adapters
- `dataknobs-bots` — Configuration-driven AI agents
- `dataknobs-fsm` — Finite state machines
- `dataknobs-structures` — Core data structures
- `dataknobs-utils` — File, JSON, HTTP, SQL utilities
- `dataknobs-xization` — NLP normalization/annotation

The legacy `dataknobs` package is no longer maintained.
