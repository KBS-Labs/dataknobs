# Contributing to Dataknobs

We welcome contributions to the Dataknobs project! This guide will help you get started with contributing code, documentation, bug reports, and feature requests.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all participants with respect and courtesy
- **Be inclusive**: Welcome newcomers and encourage diverse perspectives
- **Be constructive**: Focus on what is best for the community
- **Be patient**: Remember that people have different skill levels and backgrounds
- **Be collaborative**: Work together to resolve conflicts and reach consensus

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- **Python 3.8+** installed
- **Git** for version control
- **GitHub account** for submitting contributions
- Basic understanding of Python and software development practices

### Find an Issue to Work On

1. Browse our [GitHub Issues](https://github.com/yourusername/dataknobs/issues)
2. Look for issues labeled `good first issue` if you're new to the project
3. Check issues labeled `help wanted` for areas where we need assistance
4. Comment on the issue to let others know you're working on it

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Help us identify and fix issues
- **New features**: Add functionality that benefits users
- **Documentation**: Improve guides, tutorials, and API docs
- **Tests**: Increase code coverage and test quality
- **Performance improvements**: Optimize existing code
- **Examples**: Add usage examples and tutorials

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/dataknobs.git
cd dataknobs

# Add upstream remote
git remote add upstream https://github.com/original/dataknobs.git
```

### 2. Create Development Environment

#### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all packages
uv sync --all-packages

# Install the dk command for easy development
./setup-dk.sh
```

#### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install packages in development mode
pip install -e packages/common
pip install -e packages/structures
pip install -e packages/utils
pip install -e packages/xization
```

### 3. Verify Setup

```bash
# Using the dk command (recommended)
dk test           # Run tests
dk check          # Quick quality check
dk diagnose       # If something fails

# Or using traditional commands
pytest            # Run tests
ruff check packages/  # Check code style
mypy packages/    # Run type checking
```

### 4. Development Workflow with dk

The `dk` command simplifies your development workflow:

```bash
# Quick development cycle
dk check data     # Quick check while developing
dk fix            # Auto-fix style issues
dk test data      # Test your changes

# Before submitting PR
dk pr             # Full quality checks
dk diagnose       # If checks fail
```

See the [dk Command Guide](dk-command.md) for full details.

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

1. **Clear title**: Briefly describe the issue
2. **Description**: Detailed explanation of the problem
3. **Reproduction steps**: How to reproduce the issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Environment**: Python version, OS, package versions
7. **Code samples**: Minimal example demonstrating the issue

**Bug Report Template:**

```markdown
## Bug Description
Brief description of the issue.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
Describe what you expected to happen.

## Actual Behavior
Describe what actually happened.

## Environment
- Python version: 3.9.7
- Dataknobs version: 1.0.0
- OS: Ubuntu 20.04

## Code Sample
```python
# Minimal code example
from dataknobs_structures import Tree
tree = Tree("test")
# Issue occurs here
```
```

### Requesting Features

When requesting features:

1. **Use case**: Explain why this feature is needed
2. **Detailed description**: What the feature should do
3. **Proposed API**: How users would interact with it
4. **Alternatives considered**: Other approaches you've thought of
5. **Implementation notes**: Any technical considerations

**Feature Request Template:**

```markdown
## Feature Description
Brief description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
How should this feature work? Include API examples.

```python
# Example of proposed API
from dataknobs_utils import new_feature
result = new_feature.process_data(data)
```

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Any other relevant information.
```

### Making Code Changes

#### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/issue-description
```

#### 2. Make Your Changes

- Write clean, readable code
- Follow existing code patterns
- Add appropriate comments
- Update docstrings for public APIs

#### 3. Add Tests

```python
# Example test structure
import pytest
from dataknobs_structures import Tree

class TestYourFeature:
    def test_basic_functionality(self):
        """Test the basic functionality of your feature."""
        # Arrange
        tree = Tree("test")
        
        # Act
        result = tree.your_new_method()
        
        # Assert
        assert result is not None
        assert isinstance(result, expected_type)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        tree = Tree(None)
        with pytest.raises(ValueError):
            tree.your_new_method()
```

#### 4. Update Documentation

- Update docstrings for new/modified functions
- Add usage examples
- Update README if needed
- Add entries to CHANGELOG if appropriate

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use isort for import organization
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public APIs

### Code Formatting

```bash
# Format code with Black
black packages/

# Sort imports with isort
isort packages/

# Check formatting
black --check packages/
isort --check-only packages/
```

### Docstring Style

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.
    
    Example:
        Basic usage example:
        
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    # Implementation here
    return True
```

### Type Hints

**Important**: All files with type hints must include `from __future__ import annotations` for Python 3.9 compatibility. See the [Python Compatibility Guide](./python-compatibility.md) for details.

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

# Good examples (modern style with future annotations)
def process_files(file_paths: list[Path]) -> dict[str, Any]:
    """Process multiple files and return results."""
    pass

def get_value(data: dict[str, Any], key: str, default: str | None = None) -> str | None:
    """Get value from dictionary with optional default."""
    pass

# For complex types, create type aliases
DocumentData = dict[str, str | int | list[str]]
ProcessingResult = dict[str, bool | str | list[DocumentData]]
```

## Testing Guidelines

### Test Structure

Organize tests to match the package structure:

```
tests/
├── unit/                    # Unit tests
│   ├── structures/
│   ├── utils/
│   └── xization/
├── integration/           # Integration tests
└── fixtures/              # Test fixtures and data
```

### Writing Good Tests

```python
import pytest
from unittest.mock import Mock, patch
from dataknobs_utils import file_utils

class TestFileUtils:
    """Test file utility functions."""
    
    def test_filepath_generator_basic(self):
        """Test basic filepath generation."""
        # Use descriptive test names
        # Test the happy path first
        pass
    
    def test_filepath_generator_empty_directory(self):
        """Test filepath generation with empty directory."""
        # Test edge cases
        pass
    
    def test_filepath_generator_nonexistent_path(self):
        """Test filepath generation with nonexistent path."""
        # Test error conditions
        with pytest.raises(FileNotFoundError):
            list(file_utils.filepath_generator("/nonexistent/path"))
    
    @patch('os.walk')
    def test_filepath_generator_with_mock(self, mock_walk):
        """Test filepath generation with mocked filesystem."""
        # Mock external dependencies when needed
        mock_walk.return_value = [("/test", [], ["file1.txt", "file2.txt"])]
        
        result = list(file_utils.filepath_generator("/test"))
        
        assert len(result) == 2
        assert "/test/file1.txt" in result
        assert "/test/file2.txt" in result
```

### Test Coverage

```bash
# Run tests with coverage
pytest --cov=packages/ --cov-report=html

# View coverage report
open htmlcov/index.html

# Aim for >90% coverage
pytest --cov=packages/ --cov-fail-under=90
```

### Integration Tests

```python
# tests/integration/test_pipeline.py
import tempfile
from pathlib import Path
from dataknobs_utils import file_utils
from dataknobs_xization import normalize
from dataknobs_structures import Tree

def test_complete_text_processing_pipeline():
    """Test complete text processing pipeline integration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("getUserName() & validateInput")
        
        # Test file reading
        content = next(file_utils.fileline_generator(str(test_file)))
        assert content == "getUserName() & validateInput"
        
        # Test normalization
        normalized = normalize.expand_camelcase_fn(content)
        assert "get User Name" in normalized
        
        # Test tree structure
        tree = Tree(normalized)
        assert tree.data == normalized
```

## Documentation

### API Documentation

We use MkDocs with mkdocstrings for API documentation:

```python
def new_function(param: str) -> str:
    """Brief description of the function.
    
    Longer description with examples and usage notes.
    
    Args:
        param: Description of the parameter.
    
    Returns:
        Description of the return value.
    
    Example:
        >>> result = new_function("test")
        >>> print(result)
        'processed: test'
    """
    return f"processed: {param}"
```

### User Documentation

When adding new features, update:

1. **User Guide**: Add usage examples
2. **API Reference**: Ensure docstrings are complete
3. **Examples**: Add practical examples
4. **README**: Update if the change affects installation or basic usage

### Documentation Style

- Use clear, concise language
- Provide practical examples
- Include code snippets that work
- Explain not just "how" but "why"
- Use proper Markdown formatting

## Submitting Changes

### Pre-submission Checklist

Before submitting your pull request:

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings written/updated
- [ ] CHANGELOG updated (if applicable)

### Running Pre-commit Checks

```bash
# Using dk command (recommended)
dk pr              # Run full PR quality checks
dk diagnose        # If checks fail, see what went wrong
dk fix             # Auto-fix style issues
dk test --last     # Re-run only failed tests

# Or manually run individual checks
uv run ruff check packages/    # Style check
uv run ruff format packages/   # Format code
uv run pylint packages/*/src   # Linting
uv run mypy packages/          # Type checking
uv run pytest                  # Run tests
```

### Commit Messages

Use conventional commit messages:

```
type(scope): brief description

Longer description if needed.

- Bullet point changes
- Another change

Fixes #123
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code formatting changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(structures): add tree traversal method

Add breadth-first traversal option to Tree.find_nodes()
method to improve search performance for shallow targets.

- Add traversal parameter with 'dfs' and 'bfs' options
- Update tests and documentation
- Maintain backward compatibility

Fixes #45"
```

### Creating Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create pull request on GitHub:
   - Use the pull request template
   - Provide clear title and description
   - Link related issues
   - Add screenshots if relevant

3. Pull request template:
   ```markdown
   ## Description
   Brief description of the changes.
   
   ## Changes Made
   - List of changes
   - Another change
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing performed
   
   ## Documentation
   - [ ] Docstrings updated
   - [ ] User guide updated
   - [ ] Examples added
   
   ## Related Issues
   Fixes #123
   Closes #456
   ```

## Review Process

### What Reviewers Look For

1. **Code Quality**
   - Follows style guidelines
   - Clear and readable
   - Proper error handling
   - Efficient algorithms

2. **Testing**
   - Adequate test coverage
   - Tests actually test the feature
   - Edge cases covered
   - No flaky tests

3. **Documentation**
   - Clear docstrings
   - Updated user documentation
   - Examples work as expected

4. **Compatibility**
   - Doesn't break existing APIs
   - Works across supported Python versions
   - Handles backward compatibility

### Addressing Feedback

- Respond to comments promptly
- Ask questions if feedback is unclear
- Make requested changes
- Update tests and documentation as needed
- Mark conversations as resolved when addressed

### Approval Process

- At least one maintainer approval required
- All checks must pass
- No unresolved conversations
- Documentation updated

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Pull Requests**: Code contributions and reviews

### Getting Help

1. Check existing documentation
2. Search GitHub issues
3. Ask in GitHub Discussions
4. Create new issue if needed

### Recognition

We recognize contributors through:

- Contributor list in README
- Release notes acknowledgments
- GitHub contributor statistics
- Special recognition for significant contributions

### Becoming a Maintainer

Active contributors may be invited to become maintainers based on:

- Quality and quantity of contributions
- Understanding of the codebase
- Helpfulness to community members
- Commitment to project values

## Resources

- [Development Guide](index.md) - Main development documentation
- [Architecture Overview](architecture.md) - System design
- [Testing Guide](testing.md) - Detailed testing information
- [Python Style Guide](https://pep8.org/) - PEP 8 coding standards
- [Semantic Versioning](https://semver.org/) - Versioning guidelines

## Questions?

If you have questions about contributing:

1. Check the [Development Guide](index.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/dataknobs/issues)
3. Start a [GitHub Discussion](https://github.com/yourusername/dataknobs/discussions)
4. Create a new issue if your question hasn't been addressed

Thank you for contributing to Dataknobs! 🎉