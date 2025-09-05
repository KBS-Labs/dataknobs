"""Pytest configuration and shared fixtures for dataknobs_fsm tests."""

import pytest
from pathlib import Path
import sys

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))


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