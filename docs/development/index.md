# Development Guide

Welcome to the Dataknobs development documentation. This section provides comprehensive information for developers who want to contribute to, extend, or understand the internal workings of the Dataknobs ecosystem.

## Overview

Dataknobs is a modular Python ecosystem for AI knowledge base structures and text processing. The project is organized as a monorepo with multiple interconnected packages that work together to provide comprehensive data processing capabilities.

## Quick Start with dk Command

!!! tip "New Developer? Start Here!"
    The `dk` command is your unified interface for all development tasks. Install it with `./setup-dk.sh` and use simple commands like `dk pr` to prepare for pull requests or `dk test` to run tests.
    
    **[→ Learn the dk Command](dk-command.md)**

## Release Management

!!! success "Simplified Release Process"
    The release process has been streamlined with automated tools that handle version bumping, changelog generation, and publishing. Use `dk release` for an interactive guided process or check the **[Release Process Guide](release-process.md)** for detailed documentation and FAQ.
    
    **Quick commands:**
    - `dk release` - Interactive complete release
    - `dk release-check` - See what changed
    - `dk release-bump` - Update versions
    - `dk release-notes` - Generate changelog

## Getting Started

If you're new to Dataknobs development:

1. **[Developer Workflow (dk)](dk-command.md)** - 🚀 **Start here** - The easy way to develop
2. **[Contributing Guide](contributing.md)** - Learn how to contribute
3. **[Configuration System](configuration-system.md)** - Understand the DataKnobs configuration patterns
4. **[UV Virtual Environment Guide](uv-environment.md)** - How to work with UV package manager
5. **[Quality Checks Process](quality-checks.md)** - Developer-driven quality assurance
6. **[Architecture Overview](architecture.md)** - Understand the system design
7. **[Testing Guide](testing.md)** - Learn about our testing approach
8. **[Integration Testing & CI](integration-testing-ci.md)** - Integration testing in CI/CD pipeline
9. **[CI/CD Pipeline](ci-cd.md)** - Understand our deployment process

## Development Topics

### Core Development
- **[Contributing Guide](contributing.md)** - How to contribute code, documentation, and report issues
- **[Configuration System](configuration-system.md)** - DataKnobs configuration patterns and best practices
- **[Adding Config Support](adding-config-support.md)** - Step-by-step guide to add configuration support to packages
- **[UV Virtual Environment Guide](uv-environment.md)** - Working with UV package manager and virtual environments
- **[Quality Checks Process](quality-checks.md)** - Running quality checks locally before PRs
- **[Architecture Overview](architecture.md)** - System architecture and design principles
- **[Documentation Guide](documentation-guide.md)** - How to write and maintain documentation

### Testing
- **[Testing Guide](testing.md)** - Testing strategies, frameworks, and best practices
- **[Testing Commands](testing-guide.md)** - Practical guide to running tests with the new test infrastructure
- **[Integration Testing & CI](integration-testing-ci.md)** - Integration testing with real services and CI/CD quality gates

### Operations
- **[CI/CD Pipeline](ci-cd.md)** - Continuous integration and deployment processes
- **[Release Process](release-process.md)** - 📦 **Streamlined release workflow** with automated tools and comprehensive FAQ

## Project Structure

```
dataknobs/
├── packages/          # Individual packages
│   ├── common/          # Shared utilities
│   ├── structures/      # Core data structures
│   ├── utils/           # Utility functions
│   ├── xization/        # Text processing
│   ├── data/            # Database abstractions
│   ├── fsm/             # FSM processing
│   └── legacy/          # Legacy compatibility
├── docs/              # Documentation
├── tests/             # Integration tests
├── docker/            # Docker configurations
├── bin/               # Scripts and tools
└── resources/         # Shared resources
```

## Development Environment

### Prerequisites

- **Python**: 3.10 or higher
- **Package Manager**: UV (fast Python package manager)
- **Version Control**: Git
- **Docker**: For running PostgreSQL, Elasticsearch, and LocalStack services

### Quick Setup with UV

```bash
# Clone the repository
git clone https://github.com/yourusername/dataknobs.git
cd dataknobs

# Install all dependencies
uv sync --all-packages

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Run quality checks before PRs (includes integration tests)
./bin/run-quality-checks.sh

# Or run specific test types with the new test infrastructure
./bin/test.sh                          # Run all tests (unit + integration)
./bin/test.sh -t unit                  # Unit tests only
./bin/test.sh -t integration           # Integration tests with services
./bin/test.sh data                     # Test specific package
./bin/run-integration-tests.sh -s      # Start services for manual testing
```

### Development Services

#### Docker-based Services

Most development services run via Docker:

```bash
# Start all services
docker-compose up -d postgres elasticsearch localstack

# Check service status
docker-compose ps

# Stop services
docker-compose down
```

#### Ollama (Local Installation Required)

Unlike other services, **Ollama runs locally** due to hardware requirements (GPU access).

**Installation:**
- **macOS**: `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download from https://ollama.ai/download

**Starting Ollama:**
```bash
ollama serve
```

**Verifying Ollama:**
```bash
./bin/check-ollama.sh
# Or manually:
curl http://localhost:11434/api/tags
```

**Running Tests Without Ollama:**
```bash
export TEST_OLLAMA=false
dk test
# Or use quick test mode (skips all integration tests)
dk testquick
```

For more details, see the [UV Virtual Environment Guide](uv-environment.md) and [Quality Checks Process](quality-checks.md).

## Package Overview

### dataknobs-common
**Purpose**: Shared utilities and base classes used across all packages.

**Key Components**:
- Base classes and interfaces
- Common configuration management
- Standardized logging
- Error handling framework

### dataknobs-config
**Purpose**: Modular configuration system with environment variable support.

**Key Components**:
- YAML/JSON configuration loading
- Environment variable substitution (`${VAR:default}`)
- Factory registration for dynamic object creation
- Cross-reference resolution
- Layered configuration merging

### dataknobs-data
**Purpose**: Unified data abstraction layer for consistent operations across storage backends.

**Key Components**:
- Multiple backend support (Memory, File, PostgreSQL, Elasticsearch, S3)
- Unified `Record` and `Query` abstractions
- Factory pattern for dynamic backend selection
- Transaction management (Single, Batch, Manual)
- Vector store integration
- Streaming support for large datasets

### dataknobs-fsm
**Purpose**: Finite State Machine framework for workflow orchestration and data processing.

**Key Components**:
- Three API levels: SimpleFSM (sync), AsyncSimpleFSM (async), AdvancedFSM (debugging)
- Data handling modes (COPY, REFERENCE, DIRECT) for different performance/safety tradeoffs
- Built-in resource management (databases, files, HTTP, LLMs, vector stores)
- Streaming support with backpressure handling
- YAML/JSON configuration with inline transforms
- Step-by-step debugging with breakpoints and execution hooks

### dataknobs-structures
**Purpose**: Core data structures for hierarchical and document-based data.

**Key Components**:
- Tree data structure with advanced navigation
- Document and text processing classes
- Record storage and retrieval
- Conditional dictionary implementations

### dataknobs-utils
**Purpose**: Utility functions for various data processing tasks.

**Key Components**:
- File operations and I/O utilities
- Elasticsearch integration
- JSON processing tools
- LLM prompt management
- Database and statistical utilities

### dataknobs-xization
**Purpose**: Text normalization, tokenization, and processing.

**Key Components**:
- Text normalization functions
- Character-level analysis
- Tokenization and masking
- Lexical variation generation

## Development Workflow

### 1. Issue Creation
- Use GitHub issues to track bugs, features, and improvements
- Follow issue templates for consistency
- Label issues appropriately
- Assign to milestones when relevant

### 2. Branch Management
- **main**: Stable, production-ready code
- **develop**: Integration branch for features
- **feature/***: Individual feature development
- **bugfix/***: Bug fixes
- **hotfix/***: Critical production fixes

### 3. Code Development
- Follow [Python PEP 8](https://pep8.org/) style guidelines
- Write comprehensive docstrings
- Include type hints
- Add appropriate tests
- Update documentation

### 4. Testing
- Write unit tests for all new functionality
- Run integration tests with real services (PostgreSQL, Elasticsearch)
- Ensure all tests pass with `./bin/run-quality-checks.sh`
- Achieve minimum code coverage targets (70% overall, 90% for new code)
- Test across supported Python versions (3.10+)

### 5. Review Process
- Create pull request with detailed description
- Request review from maintainers
- Address feedback and make necessary changes
- Ensure all checks pass

## Code Standards

### Python Style
- Follow PEP 8 coding style
- Use black for code formatting
- Use isort for import organization
- Use pylint for code quality checks

### Documentation
- Write clear, comprehensive docstrings
- Follow Google docstring style
- Include usage examples
- Update README files as needed

### Testing
- Aim for >90% code coverage
- Write both unit and integration tests
- Use descriptive test names
- Include edge cases and error conditions

### Type Hints
- Use type hints for all public functions
- Import types from typing module
- Use Union, Optional, and generics appropriately

## Tools and Utilities

### Code Quality
```bash
# Format code
black packages/

# Sort imports
isort packages/

# Check style
flake8 packages/

# Type checking
mypy packages/

# Security scanning
bandit -r packages/
```

### Testing
```bash
# Run all tests (unit + integration) with new infrastructure
./bin/test.sh

# Run unit tests only
./bin/test.sh -t unit

# Run integration tests with services
./bin/test.sh -t integration

# Test specific package
./bin/test.sh data                     # All tests for data package
./bin/test.sh -t unit config          # Unit tests for config package
./bin/test.sh -t integration data      # Integration tests for data package

# Advanced options
./bin/test.sh -v                      # Verbose output
./bin/test.sh -k test_s3              # Run tests matching pattern
./bin/test.sh -x                      # Stop on first failure
./bin/test.sh -n -t integration       # Run integration tests without starting services

# Service management
./bin/run-integration-tests.sh -s     # Start services only
./bin/run-integration-tests.sh -k     # Keep services running after tests

# Legacy pytest commands (still available)
pytest                                 # Run all tests
pytest -m "not integration"            # Unit tests only
pytest --cov=packages/                # Run with coverage
pytest packages/structures/tests/     # Run specific package tests

# Run quality checks (linting + tests + coverage)
./bin/run-quality-checks.sh
```

### Documentation
```bash
# Build documentation
mkdocs serve

# Generate API docs
mkdocs build

# Test documentation links
mkdocs build --strict
```

## Performance Considerations

### Memory Management
- Use generators for large dataset processing
- Implement proper resource cleanup
- Monitor memory usage in tests
- Consider lazy loading for large data structures

### Processing Efficiency
- Profile code performance regularly
- Use appropriate data structures
- Implement caching where beneficial
- Consider parallel processing for CPU-intensive tasks

### Scalability
- Design for horizontal scaling
- Use streaming processing for large files
- Implement proper error handling and recovery
- Consider database connection pooling

## Security Guidelines

### Input Validation
- Validate all user inputs
- Sanitize data before processing
- Use parameterized queries for databases
- Implement proper authentication and authorization

### Data Handling
- Encrypt sensitive data at rest
- Use HTTPS for all network communications
- Implement proper logging (avoid logging sensitive data)
- Follow data retention policies

### Dependencies
- Regularly update dependencies
- Use security scanning tools
- Pin dependency versions
- Review new dependencies for security issues

## Debugging and Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure packages are installed in development mode
   - Check Python path configuration
   - Verify virtual environment activation

2. **Test Failures**
   - Run tests individually to isolate issues
   - Check for test data dependencies
   - Verify mock configurations

3. **Performance Issues**
   - Use profiling tools to identify bottlenecks
   - Check memory usage patterns
   - Review algorithm complexity

### Debugging Tools

```python
# Python debugger
import pdb; pdb.set_trace()

# Performance profiling
import cProfile
cProfile.run('your_function()')

# Memory profiling
from memory_profiler import profile
@profile
def your_function():
    pass
```

## Communication and Community

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code contributions and reviews

### Guidelines
- Be respectful and constructive
- Provide clear and detailed information
- Follow up on your contributions
- Help others when possible

## Resources

### Documentation
- [User Guide](../user-guide/index.md) - End-user documentation
- [API Reference](../api/index.md) - Detailed API documentation
- [Examples](../examples/index.md) - Usage examples and tutorials

### External Resources
- [Python Documentation](https://docs.python.org/)
- [PEP 8 Style Guide](https://pep8.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [MkDocs Documentation](https://www.mkdocs.org/)

## Getting Help

If you need help with development:

1. Check existing documentation and examples
2. Search GitHub issues for similar problems
3. Create a new issue with detailed information
4. Join community discussions for general questions

## Next Steps

Ready to contribute? Start with:

1. Read the [Contributing Guide](contributing.md)
2. Set up your development environment
3. Pick a "good first issue" from GitHub
4. Make your first contribution!

We welcome contributions of all types - code, documentation, testing, and community support. Thank you for helping make Dataknobs better!
