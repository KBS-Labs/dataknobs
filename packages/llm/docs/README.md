# Dataknobs LLM Documentation

**Package**: `dataknobs_llm`
**Version**: 0.1.0
**Status**: Production Ready

---

## Documentation Index

### Getting Started
- **[User Guide](USER_GUIDE.md)** - Comprehensive guide with examples
  - Installation
  - Quick start
  - Prompt library system
  - Conversation management
  - Middleware
  - Complete examples

### Production Use
- **[Best Practices](BEST_PRACTICES.md)** - Patterns for production deployments
  - Prompt design guidelines
  - Template organization
  - Validation strategy
  - RAG best practices
  - Conversation management
  - Middleware patterns
  - Storage and performance
  - Error handling
  - Testing
  - Production deployment

### Technical Reference
- **[Schema Versioning](SCHEMA_VERSIONING.md)** - Storage schema management
  - Version format and history
  - Automatic migration
  - Adding new versions
  - Migration examples
  - Error handling
  - Best practices

---

## Quick Links

### Key Files
- `../src/dataknobs_llm/prompts/` - Prompt library implementation
- `../src/dataknobs_llm/conversations/` - Conversation management
- `../src/dataknobs_llm/llm/` - LLM providers
- `../tests/` - Comprehensive test suite

### Status Documents
- `../PROMPT_LIBRARY_STATUS.md` - Implementation status and progress
- `../../tmp/active/conversation-manager-design.md` - Design documentation

---

## Features Overview

### Prompt Library
- Template-based prompt management
- Variable substitution and conditionals
- Validation with configurable levels
- RAG integration
- Template composition and inheritance
- Filesystem and config-based libraries

### Conversation Management
- Multi-turn conversations
- Tree-based branching
- Persistent storage (multiple backends)
- Resume across sessions
- Metadata tracking

### Middleware System
- Request/response processing
- Built-in middleware:
  - Logging
  - Content filtering
  - Response validation
  - Metadata injection
- Custom middleware support
- Onion model execution

### Storage
- Abstract storage interface
- Dataknobs backend adapter
- Support for: Memory, File, SQLite, PostgreSQL
- Schema versioning with automatic migration
- Tree serialization for branching

---

## Getting Help

1. **Read the documentation** - Start with the User Guide
2. **Check examples** - See complete examples in User Guide
3. **Review tests** - Explore `../tests/` for patterns
4. **Check status** - See PROMPT_LIBRARY_STATUS.md for implementation details
5. **Report issues** - https://github.com/kbs-labs/dataknobs/issues

---

## Development Status

✅ **All 6 phases complete**:
1. Core Infrastructure
2. Resource Adapters
3. Prompt Library Implementations
4. Prompt Builders
5. Template Composition
6. LLM Integration & Conversation Management

✅ **516 tests passing** (76% coverage)
✅ **Documentation complete**
✅ **Production ready**

---

## What's Next

### Optional Future Enhancements
- Variable transformations (filters)
- Advanced conditional logic
- Prompt versioning and A/B testing
- RAG caching layer
- Performance optimization
- Migration guide for other libraries

See PROMPT_LIBRARY_STATUS.md for full details.

---

**Last Updated**: 2025-10-29
