"""Test common package basics."""

import dataknobs_common


def test_version():
    """Test that version is defined."""
    assert hasattr(dataknobs_common, "__version__")
    assert isinstance(dataknobs_common.__version__, str)
    assert dataknobs_common.__version__ == "1.0.0"
