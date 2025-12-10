"""Test that legacy package re-exports work correctly."""

import warnings


def test_legacy_package_deprecation_warning():
    """Test that importing the legacy package shows a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import dataknobs

        # Check that a deprecation warning was issued
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()


def test_legacy_utils_imports():
    """Test that utils submodules are accessible through legacy package."""
    # Import through legacy package
    from dataknobs import utils

    # Check that submodules are available
    assert hasattr(utils, "json_utils")
    assert hasattr(utils, "file_utils")
    assert hasattr(utils, "pandas_utils")
    assert hasattr(utils, "sys_utils")

    # Check that commonly used functions are available
    assert hasattr(utils, "get_value")
    assert hasattr(utils, "filepath_generator")


def test_legacy_structures_imports():
    """Test that structures submodules are accessible through legacy package."""
    from dataknobs import structures

    # Check that main classes are available
    assert hasattr(structures, "RecordStore")
    assert hasattr(structures, "Text")
    assert hasattr(structures, "TextMetaData")
    assert hasattr(structures, "Tree")
    assert hasattr(structures, "cdict")
    assert hasattr(structures, "build_tree_from_string")


def test_legacy_xization_imports():
    """Test that xization submodules are accessible through legacy package."""
    from dataknobs import xization

    # Check that main modules are available
    assert hasattr(xization, "normalize")
    assert hasattr(xization, "authorities")
    assert hasattr(xization, "masking_tokenizer")


def test_legacy_package_version():
    """Test that the legacy package has a version."""
    import dataknobs

    assert hasattr(dataknobs, "__version__")
    assert isinstance(dataknobs.__version__, str)
    # Version should match pyproject.toml - do not hardcode specific version
    assert len(dataknobs.__version__.split(".")) == 3  # semver format


def test_backward_compatibility_json_utils():
    """Test that commonly used json_utils functions work through legacy import."""
    from dataknobs.utils import get_value

    # Test get_value function
    test_dict = {"a": {"b": {"c": 42}}}
    result = get_value(test_dict, "a.b.c")
    assert result == 42

    result = get_value(test_dict, "a.b.d", default="not_found")
    assert result == "not_found"


def test_backward_compatibility_file_utils():
    """Test that commonly used file_utils functions work through legacy import."""
    import gzip
    import tempfile

    from dataknobs.utils import is_gzip_file

    # Create a temporary gzip file
    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp:
        with gzip.open(tmp.name, "wb") as gz:
            gz.write(b"test content")

        # Test is_gzip_file function
        assert is_gzip_file(tmp.name) is True

    # Test with non-gzip file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"test content")
        tmp.flush()
        assert is_gzip_file(tmp.name) is False
