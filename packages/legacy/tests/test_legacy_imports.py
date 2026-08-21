"""Test that legacy package re-exports work correctly."""

import importlib
import sys
import warnings

import pytest


def test_legacy_package_deprecation_warning():
    """Test that importing the legacy package shows a deprecation warning.

    The warning is emitted at module-import time (``dataknobs/__init__.py``).
    Since Python caches modules in ``sys.modules``, the module body only runs
    on the first import in the process. Any other test that imports
    ``dataknobs`` first would otherwise make this a no-op (an order-dependent
    flake surfaced by randomized test order). Evict the package so the import
    re-executes the module body and re-emits the warning regardless of order.
    """
    for name in [n for n in sys.modules if n == "dataknobs" or n.startswith("dataknobs.")]:
        del sys.modules[name]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import dataknobs  # noqa: F401  (import side effect under test)

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


# The submodules each shim re-exports, as ``legacy package -> submodule names``.
# Taken from the ``from <modular package> import ...`` list in each shim's
# ``__init__``: the shim already decides what it publishes, and these are the
# names a pre-split import could reach.
LEGACY_SUBMODULES = {
    "dataknobs.structures": ("conditional_dict", "document", "record_store", "tree"),
    "dataknobs.utils": (
        "elasticsearch_utils",
        "emoji_utils",
        "file_utils",
        "json_extractor",
        "json_utils",
        "llm_utils",
        "pandas_utils",
        "requests_utils",
        "resource_utils",
        "sql_utils",
        "stats_utils",
        "subprocess_utils",
        "sys_utils",
        "xml_utils",
    ),
    "dataknobs.xization": (
        "annotations",
        "authorities",
        "lexicon",
        "masking_tokenizer",
        "normalize",
    ),
}

DOTTED_PATHS = [
    f"{package}.{name}" for package, names in LEGACY_SUBMODULES.items() for name in names
]


@pytest.mark.parametrize("dotted_path", DOTTED_PATHS)
def test_legacy_submodule_resolves_as_a_dotted_module_path(dotted_path: str) -> None:
    """Every re-exported submodule is reachable as ``dataknobs.<pkg>.<name>``.

    The tests above assert ``hasattr(structures, "tree")``, which passes on an
    attribute binding alone. Pre-split code does not use attribute access -- it
    writes ``from dataknobs.structures.tree import Tree``, and Python resolves
    that through ``sys.modules``, not through the parent's attributes. Asserting
    only the attribute form is why the dotted form could be broken for the whole
    life of the package while the suite stayed green.
    """
    importlib.import_module(dotted_path)


def test_documented_legacy_import_form_works() -> None:
    """The exact ``from`` form the migration guide and READMEs publish.

    These lines are what a pre-split user's code contains verbatim, so they are
    the contract the legacy package exists to honour.
    """
    from dataknobs.structures.tree import Tree
    from dataknobs.utils.json_utils import get_value
    from dataknobs.xization.normalize import basic_normalization_fn

    assert get_value({"a": {"b": 7}}, "a.b") == 7
    assert Tree("root").data == "root"
    assert callable(basic_normalization_fn)


def test_legacy_submodule_is_the_modular_module_itself() -> None:
    """The alias is the same object, not a copy.

    Isinstance checks and module-level state have to agree across the two import
    paths, or the shim would introduce a split-brain of its own.
    """
    import dataknobs_structures.tree

    from dataknobs.structures import tree

    assert tree is dataknobs_structures.tree
    assert sys.modules["dataknobs.structures.tree"] is dataknobs_structures.tree
