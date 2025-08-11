# Testing Guide

This guide covers testing strategies, frameworks, and best practices for the Dataknobs project. We use comprehensive testing to ensure code quality, reliability, and maintainability across all packages.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Types](#test-types)
- [Testing Framework](#testing-framework)
- [Test Organization](#test-organization)
- [Writing Tests](#writing-tests)
- [Test Coverage](#test-coverage)
- [Running Tests](#running-tests)
- [Continuous Integration](#continuous-integration)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Testing Philosophy

### Core Principles

1. **Test-Driven Development (TDD)**: Write tests before implementing features when possible
2. **Comprehensive Coverage**: Aim for high test coverage while focusing on quality over quantity
3. **Fast Feedback**: Tests should run quickly to provide rapid feedback during development
4. **Reliability**: Tests should be deterministic and not flaky
5. **Maintainability**: Tests should be easy to understand, modify, and extend

### Testing Pyramid

We follow the testing pyramid approach:

```
    /\        
   /E2E\      <- End-to-End (Few)
  /____\      
 /Integration\ <- Integration Tests (Some)
/__________\   
/   Unit    \ <- Unit Tests (Many)
____________
```

- **Unit Tests (70%)**: Fast, focused tests for individual functions/classes
- **Integration Tests (20%)**: Test interaction between components
- **End-to-End Tests (10%)**: Test complete workflows and user scenarios

## Test Types

### Unit Tests

Test individual functions, methods, or classes in isolation:

```python
# tests/unit/structures/test_tree.py
import pytest
from dataknobs_structures import Tree

class TestTreeBasicOperations:
    """Test basic Tree operations."""
    
    def test_tree_creation(self):
        """Test creating a tree with data."""
        tree = Tree("root")
        assert tree.data == "root"
        assert tree.parent is None
        assert tree.children is None
    
    def test_add_child(self):
        """Test adding a child to a tree."""
        root = Tree("root")
        child = root.add_child("child")
        
        assert child.data == "child"
        assert child.parent == root
        assert len(root.children) == 1
        assert root.children[0] == child
    
    def test_tree_depth(self):
        """Test tree depth calculation."""
        root = Tree("root")
        child = root.add_child("child")
        grandchild = child.add_child("grandchild")
        
        assert root.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2
    
    def test_invalid_data_raises_error(self):
        """Test that invalid data raises appropriate error."""
        with pytest.raises(ValueError, match="Data cannot be None"):
            Tree(None)
```

### Integration Tests

Test interactions between multiple components:

```python
# tests/integration/test_text_processing_pipeline.py
import tempfile
from pathlib import Path
from dataknobs_utils import file_utils
from dataknobs_xization import normalize
from dataknobs_structures import Tree, Document

class TestTextProcessingPipeline:
    """Test integration of text processing components."""
    
    def test_file_to_tree_pipeline(self):
        """Test complete pipeline from file to tree structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            test_file = Path(temp_dir) / "test.txt"
            test_content = "getUserName() & validateInput"
            test_file.write_text(test_content)
            
            # File processing
            file_content = next(file_utils.fileline_generator(str(test_file)))
            assert file_content == test_content
            
            # Text normalization
            normalized = normalize.expand_camelcase_fn(file_content)
            normalized = normalize.expand_ampersand_fn(normalized)
            
            # Tree creation
            doc = Document(normalized)
            tree = Tree(doc)
            
            assert "get User Name" in tree.data.text
            assert "and" in tree.data.text
    
    def test_batch_processing_integration(self):
        """Test processing multiple files in batch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            files = {
                "file1.txt": "firstName & lastName",
                "file2.txt": "getUserData() function",
                "file3.txt": "processInput & validateOutput"
            }
            
            for filename, content in files.items():
                (Path(temp_dir) / filename).write_text(content)
            
            # Process all files
            results = []
            for filepath in file_utils.filepath_generator(temp_dir):
                if filepath.endswith('.txt'):
                    content = next(file_utils.fileline_generator(filepath))
                    normalized = normalize.basic_normalization_fn(content)
                    results.append(normalized)
            
            assert len(results) == 3
            assert all(isinstance(result, str) for result in results)
```

### End-to-End Tests

Test complete user workflows:

```python
# tests/e2e/test_user_workflows.py
import tempfile
import json
from pathlib import Path
from dataknobs_utils import file_utils, elasticsearch_utils
from dataknobs_xization import normalize
from dataknobs_structures import Tree, Document

class TestUserWorkflows:
    """Test complete user workflows."""
    
    def test_document_analysis_workflow(self):
        """Test complete document analysis workflow."""
        # This would test a realistic user scenario
        # from document input to final analysis output
        pass
    
    def test_search_indexing_workflow(self):
        """Test document indexing and search workflow."""
        # This would test the complete search pipeline
        pass
```

## Testing Framework

### pytest Configuration

We use pytest as our primary testing framework. Configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
    "--cov=packages",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=85",
]
markers = [
    "slow: marks tests as slow (deselect with '-m "not slow"')",
    "integration: marks tests as integration tests",
    "e2e: marks tests as end-to-end tests",
    "unit: marks tests as unit tests",
]
```

### Test Fixtures

Use fixtures for reusable test setup:

```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path
from dataknobs_structures import Tree, Document
from dataknobs_utils import file_utils

@pytest.fixture
def sample_tree():
    """Create a sample tree for testing."""
    root = Tree("root")
    child1 = root.add_child("child1")
    child2 = root.add_child("child2")
    grandchild = child1.add_child("grandchild")
    return root

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        "This is a sample document with some text.",
        metadata={"title": "Sample", "author": "Test"}
    )

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_text_files(temp_directory):
    """Create sample text files for testing."""
    files = {
        "file1.txt": "Sample text content 1",
        "file2.txt": "Sample text content 2",
        "file3.txt": "Sample text content 3"
    }
    
    file_paths = []
    for filename, content in files.items():
        file_path = temp_directory / filename
        file_path.write_text(content)
        file_paths.append(file_path)
    
    return file_paths
```

### Parametrized Tests

Test multiple scenarios efficiently:

```python
import pytest
from dataknobs_xization import normalize

class TestNormalizationFunctions:
    """Test text normalization functions."""
    
    @pytest.mark.parametrize("input_text,expected", [
        ("firstName", "first Name"),
        ("lastName", "last Name"),
        ("getUserID", "get User ID"),
        ("XMLParser", "XML Parser"),
        ("HTMLElement", "HTML Element"),
    ])
    def test_expand_camelcase(self, input_text, expected):
        """Test camelCase expansion with various inputs."""
        result = normalize.expand_camelcase_fn(input_text)
        assert result == expected
    
    @pytest.mark.parametrize("input_text,expected", [
        ("A & B", "A and B"),
        ("Research & Development", "Research and Development"),
        ("X&Y", "X and Y"),
        ("  A  &  B  ", "  A  and  B  "),
    ])
    def test_expand_ampersand(self, input_text, expected):
        """Test ampersand expansion with various inputs."""
        result = normalize.expand_ampersand_fn(input_text)
        assert result == expected
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py              # Global fixtures
├── unit/                    # Unit tests
│   ├── conftest.py          # Unit test fixtures
│   ├── structures/
│   │   ├── test_tree.py
│   │   ├── test_document.py
│   │   └── test_record_store.py
│   ├── utils/
│   │   ├── test_file_utils.py
│   │   ├── test_json_utils.py
│   │   └── test_elasticsearch_utils.py
│   └── xization/
│       ├── test_normalize.py
│       └── test_masking_tokenizer.py
├── integration/            # Integration tests
│   ├── test_pipeline.py
│   └── test_package_interop.py
├── e2e/                    # End-to-end tests
│   └── test_user_workflows.py
└── fixtures/               # Test data and fixtures
    ├── sample_data/
    └── mock_responses/
```

### Naming Conventions

- **Test files**: `test_*.py`
- **Test classes**: `Test*` (e.g., `TestTreeOperations`)
- **Test methods**: `test_*` (e.g., `test_add_child_updates_parent`)
- **Fixtures**: Descriptive names without `test_` prefix

## Writing Tests

### Test Structure (AAA Pattern)

```python
def test_tree_add_child():
    """Test adding a child to a tree."""
    # Arrange: Set up test data and conditions
    root = Tree("root")
    child_data = "child"
    
    # Act: Execute the code being tested
    child = root.add_child(child_data)
    
    # Assert: Verify the results
    assert child.data == child_data
    assert child.parent == root
    assert len(root.children) == 1
```

### Good Test Characteristics

1. **Clear and Descriptive Names**:
   ```python
   # Good
   def test_tree_find_nodes_returns_empty_list_when_no_matches():
       pass
   
   # Bad
   def test_find():
       pass
   ```

2. **Single Responsibility**:
   ```python
   # Good - tests one behavior
   def test_tree_depth_calculation_for_root_node():
       root = Tree("root")
       assert root.depth == 0
   
   def test_tree_depth_calculation_for_child_node():
       root = Tree("root")
       child = root.add_child("child")
       assert child.depth == 1
   
   # Bad - tests multiple behaviors
   def test_tree_operations():
       root = Tree("root")
       assert root.depth == 0
       child = root.add_child("child")
       assert child.depth == 1
       # ... more assertions
   ```

3. **Independent and Isolated**:
   ```python
   # Good - each test is independent
   class TestFileUtils:
       def test_file_reading(self, temp_directory):
           test_file = temp_directory / "test.txt"
           test_file.write_text("test content")
           # Test reading logic
       
       def test_file_writing(self, temp_directory):
           test_file = temp_directory / "output.txt"
           # Test writing logic
   ```

### Testing Error Conditions

```python
import pytest
from dataknobs_structures import Tree

class TestTreeErrorConditions:
    """Test error conditions and edge cases."""
    
    def test_tree_creation_with_none_data_raises_error(self):
        """Test that creating tree with None data raises ValueError."""
        with pytest.raises(ValueError, match="Data cannot be None"):
            Tree(None)
    
    def test_add_child_with_invalid_position_raises_error(self):
        """Test that invalid child position raises IndexError."""
        root = Tree("root")
        with pytest.raises(IndexError):
            root.add_child("child", child_pos=10)
    
    def test_find_nodes_with_invalid_function_raises_error(self):
        """Test that invalid acceptance function raises TypeError."""
        root = Tree("root")
        with pytest.raises(TypeError):
            root.find_nodes("not a function")
```

### Mocking and Patching

```python
from unittest.mock import Mock, patch, MagicMock
import pytest
from dataknobs_utils import elasticsearch_utils

class TestElasticsearchUtils:
    """Test Elasticsearch utilities with mocking."""
    
    @patch('dataknobs_utils.requests_utils.RequestHelper')
    def test_elasticsearch_index_creation(self, mock_request_helper):
        """Test Elasticsearch index creation."""
        # Setup mock
        mock_helper = Mock()
        mock_request_helper.return_value = mock_helper
        mock_helper.request.return_value.succeeded = True
        
        # Create index
        table_settings = elasticsearch_utils.TableSettings(
            "test_index", {}, {}
        )
        index = elasticsearch_utils.ElasticsearchIndex(
            None, [table_settings]
        )
        
        # Verify mock was called
        mock_request_helper.assert_called_once()
        mock_helper.request.assert_called()
    
    def test_file_processing_with_mock_filesystem(self):
        """Test file processing with mocked filesystem."""
        with patch('os.walk') as mock_walk:
            # Setup mock return value
            mock_walk.return_value = [
                ('/test', [], ['file1.txt', 'file2.txt'])
            ]
            
            # Test file processing
            from dataknobs_utils import file_utils
            files = list(file_utils.filepath_generator('/test'))
            
            # Verify results
            assert len(files) == 2
            assert '/test/file1.txt' in files
            assert '/test/file2.txt' in files
```

### Testing Async Code

```python
import pytest
import asyncio

# If you have async functionality
@pytest.mark.asyncio
async def test_async_processing():
    """Test asynchronous processing."""
    # Test async code here
    result = await async_function()
    assert result is not None
```

## Test Coverage

### Coverage Goals

- **Overall**: >90% code coverage
- **Critical paths**: 100% coverage
- **New code**: 100% coverage
- **Public APIs**: 100% coverage

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=packages/

# Generate HTML coverage report
pytest --cov=packages/ --cov-report=html

# View coverage report
open htmlcov/index.html

# Coverage with branch analysis
pytest --cov=packages/ --cov-branch

# Fail if coverage below threshold
pytest --cov=packages/ --cov-fail-under=90
```

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = ["packages"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/conftest.py",
    "*/__pycache__/*",
    "*/migrations/*",
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
show_missing = true
skip_covered = false
```

### Excluding Code from Coverage

```python
def complex_function():
    try:
        # Main logic
        return process_data()
    except Exception:  # pragma: no cover
        # Error handling that's hard to test
        log_error()
        raise

if TYPE_CHECKING:  # pragma: no cover
    from typing import TYPE_CHECKING
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/structures/test_tree.py

# Run specific test class
pytest tests/unit/structures/test_tree.py::TestTreeOperations

# Run specific test method
pytest tests/unit/structures/test_tree.py::TestTreeOperations::test_add_child

# Run tests matching pattern
pytest -k "test_tree and not slow"
```

### Test Selection

```bash
# Run by markers
pytest -m "unit"                    # Only unit tests
pytest -m "integration"             # Only integration tests
pytest -m "not slow"                # Skip slow tests
pytest -m "unit or integration"     # Unit OR integration tests

# Run by directory
pytest tests/unit/                  # Only unit tests
pytest tests/integration/           # Only integration tests
```

### Parallel Testing

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto          # Auto-detect CPU count
pytest -n 4             # Use 4 processes
```

### Verbose Output

```bash
# Verbose output
pytest -v

# Very verbose output
pytest -vv

# Show output from print statements
pytest -s

# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install --with dev,test
    
    - name: Run linting
      run: |
        poetry run black --check packages/
        poetry run isort --check-only packages/
        poetry run flake8 packages/
        poetry run mypy packages/
    
    - name: Run tests
      run: |
        poetry run pytest --cov=packages/ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
  
  - repo: local
    hooks:
      - id: pytest
        name: Run tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Best Practices

### Test Design

1. **Write Tests First**: Use TDD when possible
2. **Keep Tests Simple**: One assertion per test when practical
3. **Use Descriptive Names**: Test names should explain what's being tested
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Maintain Test Independence**: Tests shouldn't depend on each other

### Test Data

1. **Use Fixtures**: Create reusable test data with fixtures
2. **Minimize Test Data**: Use smallest data set that tests the behavior
3. **Avoid Hardcoded Values**: Use variables and constants
4. **Clean Up Resources**: Properly clean up files, database connections, etc.

### Mocking Guidelines

1. **Mock External Dependencies**: Don't test external services
2. **Mock at the Right Level**: Mock interfaces, not implementations
3. **Verify Mock Interactions**: Check that mocks are called correctly
4. **Keep Mocks Simple**: Don't over-complicate mock setups

## Common Patterns

### Testing File Operations

```python
import tempfile
from pathlib import Path
from dataknobs_utils import file_utils

def test_file_writing_and_reading():
    """Test writing and reading files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.txt"
        test_content = ["line1", "line2", "line3"]
        
        # Write file
        file_utils.write_lines(str(test_file), test_content)
        
        # Read and verify
        read_content = list(file_utils.fileline_generator(str(test_file)))
        assert read_content == sorted(test_content)  # write_lines sorts
```

### Testing Configuration

```python
@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        },
        "elasticsearch": {
            "host": "localhost",
            "port": 9200
        }
    }

def test_configuration_loading(test_config):
    """Test configuration loading and access."""
    from dataknobs_utils import llm_utils
    
    host = llm_utils.get_value_by_key(test_config, "database.host")
    assert host == "localhost"
    
    port = llm_utils.get_value_by_key(test_config, "database.port")
    assert port == 5432
```

### Testing Exceptions

```python
def test_exception_with_specific_message():
    """Test that specific exception message is raised."""
    with pytest.raises(ValueError, match="Invalid input: expected string"):
        process_input(123)

def test_multiple_exception_types():
    """Test handling of multiple exception types."""
    with pytest.raises((ValueError, TypeError)):
        risky_operation()
```

### Testing Data Structures

```python
def test_tree_structure_integrity():
    """Test that tree structure maintains integrity."""
    root = Tree("root")
    child1 = root.add_child("child1")
    child2 = root.add_child("child2")
    grandchild = child1.add_child("grandchild")
    
    # Test parent-child relationships
    assert child1.parent == root
    assert child2.parent == root
    assert grandchild.parent == child1
    
    # Test children collections
    assert set(root.children) == {child1, child2}
    assert grandchild in child1.children
    
    # Test tree navigation
    assert grandchild.root == root
    assert child1.depth == 1
    assert grandchild.depth == 2
```

## Troubleshooting

### Common Test Issues

1. **Flaky Tests**:
   ```python
   # Problem: Tests that sometimes pass, sometimes fail
   # Solution: Remove randomness, use fixed seeds, proper cleanup
   
   import random
   
   def test_with_randomness():
       random.seed(42)  # Fixed seed for reproducibility
       result = function_with_randomness()
       assert result in expected_range
   ```

2. **Slow Tests**:
   ```python
   # Mark slow tests
   @pytest.mark.slow
   def test_large_dataset_processing():
       # Use smaller dataset for testing
       small_dataset = create_test_dataset(size=100)
       result = process_dataset(small_dataset)
       assert validate_result(result)
   ```

3. **Test Dependencies**:
   ```python
   # Problem: Tests that depend on execution order
   # Solution: Make tests independent
   
   class TestIndependent:
       def setup_method(self):
           """Setup for each test method."""
           self.data = create_fresh_test_data()
       
       def test_operation_a(self):
           result = operation_a(self.data)
           assert result.success
       
       def test_operation_b(self):
           result = operation_b(self.data)
           assert result.success
   ```

### Debugging Test Failures

```bash
# Run specific failing test with verbose output
pytest tests/unit/test_failing.py::test_method -vv -s

# Drop into debugger on failure
pytest --pdb

# Run with warnings enabled
pytest -W error::UserWarning

# Show local variables on failure
pytest --tb=long -l
```

### Performance Testing

```python
import time
import pytest

def test_performance_benchmark():
    """Test that operation completes within time limit."""
    start_time = time.time()
    
    result = expensive_operation()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    assert execution_time < 1.0  # Should complete in less than 1 second
    assert result is not None

# Using pytest-benchmark for more sophisticated benchmarks
def test_function_benchmark(benchmark):
    """Benchmark function performance."""
    result = benchmark(function_to_benchmark, arg1, arg2)
    assert result.success
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

This testing guide provides comprehensive coverage of testing practices for the Dataknobs project. For questions or suggestions about testing, please create an issue or start a discussion on GitHub.