# Dataknobs Release Notes

## 🎆 Dataknobs Modular Architecture Release

**Release Date**: [TBD]
**Version**: 1.0.0 (New Packages) / 0.0.15 (Legacy)

### 🎉 Major Announcement

We're excited to announce the release of Dataknobs' new modular architecture! The monolithic `dataknobs` package has been split into focused, independent packages that you can install based on your specific needs.

### 📦 New Packages

#### dataknobs-structures (v1.0.0)
- **Description**: Core data structures including Tree, Document, RecordStore, and ConditionalDict
- **Install**: `pip install dataknobs-structures`
- **Key Features**:
  - Hierarchical tree structures with efficient traversal
  - Document management with metadata
  - Record storage with indexing
  - Conditional dictionaries with validation

#### dataknobs-utils (v1.0.0)
- **Description**: Comprehensive utility functions for JSON, files, Elasticsearch, and more
- **Install**: `pip install dataknobs-utils`
- **Key Features**:
  - JSON manipulation and extraction
  - File processing utilities
  - Elasticsearch integration
  - LLM prompt utilities
  - SQL and pandas helpers

#### dataknobs-xization (v1.0.0)
- **Description**: Text processing, normalization, and tokenization tools
- **Install**: `pip install dataknobs-xization`
- **Key Features**:
  - Text normalization
  - Masking tokenizer
  - Annotation framework
  - Authority management
  - Lexicon support

#### dataknobs-common (v1.0.0)
- **Description**: Shared components and base functionality
- **Install**: `pip install dataknobs-common`
- **Note**: Automatically installed as a dependency of other packages

### 🔄 Legacy Package Update (v0.0.15)

The original `dataknobs` package has been updated to provide backward compatibility:
- **Compatibility Layer**: Re-exports all functionality from new packages
- **Deprecation Notices**: Will show warnings starting in v0.0.16 (Month 6)
- **Migration Path**: Continues to work but users should migrate to new packages

### ✨ Key Improvements

#### Performance
- 🚀 **Faster Installation**: Install only what you need
- ⚡ **Better Dependency Resolution**: Using `uv` package manager
- 📦 **Smaller Package Size**: Each package is focused and lightweight

#### Developer Experience
- 🔧 **Better IDE Support**: Improved type hints and autocomplete
- 📖 **Comprehensive Documentation**: New MkDocs site with full API reference
- 🧪 **Extensive Testing**: 100% test coverage across all packages
- 🎯 **CI/CD Pipeline**: Automated testing and deployment

#### Architecture
- 🏗️ **Modular Design**: Clean separation of concerns
- 🔗 **Loose Coupling**: Packages can evolve independently
- 📦 **Semantic Versioning**: Clear version management per package

### 🔄 Migration Guide

Migrating from the legacy package is straightforward:

1. **Install new packages**:
   ```bash
   pip install dataknobs-structures dataknobs-utils dataknobs-xization
   ```

2. **Update imports**:
   ```python
   # Old
   from dataknobs.structures.tree import Tree
   
   # New
   from dataknobs_structures import Tree
   ```

3. **Run migration script**:
   ```bash
   python migrate_dataknobs.py /path/to/project
   ```

Full migration guide: [docs/migration-guide.md](docs/migration-guide.md)

### 📋 Breaking Changes

#### Renamed Functions
- `normalize_text` → `basic_normalization_fn`
- `MaskingTokenizer` → `TextFeatures`

#### Removed Functions
- `set_value` from json_utils (use dict operations)
- `read_file`, `write_file` from file_utils (use generators)

### 🔮 What's Next

#### Coming Soon
- **Month 3**: User feedback integration
- **Month 6**: Deprecation warnings in legacy package
- **Month 12**: Legacy package end-of-life

#### Roadmap
- Enhanced LLM integration utilities
- Vector database support
- Streaming data processing
- GraphQL utilities
- Advanced text analysis features

### 📖 Documentation

- **Documentation Site**: [Run locally with `./bin/dev.sh docs`]
- **Migration Guide**: [docs/migration-guide.md](docs/migration-guide.md)
- **API Reference**: Complete API documentation for all packages
- **Examples**: [docs/examples/](docs/examples/)

### 👥 Contributors

This major release was made possible by the Dataknobs community. Special thanks to all contributors who provided feedback, reported issues, and submitted pull requests.

### 🔧 Development

#### New Development Workflow
```bash
# Setup development environment
./bin/dev.sh setup

# Run tests
./bin/dev.sh test

# Run linting
./bin/dev.sh lint

# Build packages
./bin/dev.sh build

# Serve documentation
./bin/dev.sh docs
```

### 🔗 Links

- **GitHub**: https://github.com/kbs-labs/dataknobs
- **PyPI**: https://pypi.org/project/dataknobs-structures/
- **Issues**: https://github.com/kbs-labs/dataknobs/issues
- **Discussions**: https://github.com/kbs-labs/dataknobs/discussions

### 📢 Deprecation Notice

The legacy `dataknobs` package will be deprecated according to this timeline:
- **Now - Month 3**: Soft launch, gathering feedback
- **Month 3-6**: Active migration period
- **Month 6-9**: Deprecation warnings
- **Month 12**: End of support

Please plan your migration accordingly.

### 🎆 Thank You!

Thank you for being part of the Dataknobs community. We're excited about this new chapter and look forward to building amazing things together with the new modular architecture!

---

## Previous Releases

### v0.0.14 (Legacy)
- Last version before modular split
- Full functionality in monolithic package
- Poetry-based dependency management

### v0.0.13 and earlier
- See git history for details
