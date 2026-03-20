"""Pytest configuration and shared fixtures for dataknobs_fsm tests."""

import pytest
from pathlib import Path
import sys

# Add src and package root to path for testing.
# Package root is needed so ``from examples.advanced_debugging import ...``
# resolves when pytest runs from the workspace root.
_pkg_root = Path(__file__).parent.parent
src_path = _pkg_root / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "id": "test-123",
        "name": "Test Item",
        "value": 42,
        "metadata": {
            "created": "2024-01-01",
            "source": "test"
        }
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for file-based tests."""
    return tmp_path