"""Integration tests to verify interoperability between dataknobs packages."""

import pytest


def test_structures_package_imports():
    """Test that structures package exports are accessible."""
    from dataknobs_structures import Tree, Text, TextMetaData, cdict, RecordStore
    from dataknobs_structures.tree import build_tree_from_string
    
    # Basic smoke tests
    tree = Tree("root")
    assert tree is not None
    
    # Text requires metadata
    metadata = TextMetaData(text_id=1)
    doc = Text("test", metadata)
    assert doc.text == "test"
    
    # cdict requires a callable accept function
    cd = cdict(lambda d, k, v: True, {"key": "value"})
    assert cd["key"] == "value"


def test_utils_package_imports():
    """Test that utils package exports are accessible."""
    from dataknobs_utils import (
        json_utils,
        file_utils,
        elasticsearch_utils,
        pandas_utils,
        requests_utils,
        llm_utils,
    )
    
    # Verify modules are imported
    assert json_utils is not None
    assert file_utils is not None


def test_xization_package_imports():
    """Test that xization package exports are accessible."""
    from dataknobs_xization import (
        normalize,
        masking_tokenizer,
        annotations,
        authorities,
        lexicon,
    )
    from dataknobs_xization.masking_tokenizer import CharacterFeatures, TextFeatures
    
    # Verify modules are imported
    assert normalize is not None
    assert CharacterFeatures is not None


def test_cross_package_dependencies():
    """Test that packages can work together."""
    # Utils depends on structures
    from dataknobs_utils.json_utils import build_tree_from_string
    from dataknobs_structures.tree import Tree
    
    # Create a tree using structures
    tree = Tree("root")
    tree.add_child("child1")
    
    # Verify utils can work with structures
    assert build_tree_from_string is not None
    
    # Xization depends on both structures and utils
    from dataknobs_xization.masking_tokenizer import TextFeatures
    from dataknobs_structures.document import Text, TextMetaData
    
    # Create a document and process it
    metadata = TextMetaData(text_id=1)
    doc = Text("Hello World", metadata)
    features = TextFeatures(doc.text)
    assert features.text == "Hello World"


def test_legacy_package_compatibility():
    """Test that legacy package provides backward compatibility."""
    # Import from legacy package - modules are attributes, not subpackages
    from dataknobs.structures import tree, Tree
    from dataknobs.utils import json_utils, get_value
    from dataknobs.xization import normalize, basic_normalization_fn
    
    # Test basic functionality
    tree_obj = Tree("legacy_root")
    assert tree_obj.data == "legacy_root"
    
    # Also test via module access
    tree_obj2 = tree.Tree("legacy_root2")
    assert tree_obj2.data == "legacy_root2"
    
    test_dict = {"key": {"nested": "value"}}
    value = get_value(test_dict, "key.nested")
    assert value == "value"
    
    # Also test via module access
    value2 = json_utils.get_value(test_dict, "key.nested")
    assert value2 == "value"
    
    # Use the actual function that exists
    normalized = basic_normalization_fn("TEST")
    assert normalized == "test"
    
    # Also test via module access
    normalized2 = normalize.basic_normalization_fn("TEST2")
    assert normalized2 == "test2"


def test_legacy_package_reexports():
    """Test that legacy package properly re-exports from new packages."""
    # Import from legacy - modules are attributes
    from dataknobs.structures import tree as legacy_tree_module, Tree as LegacyTree
    from dataknobs.utils import json_utils as legacy_json, get_value as legacy_get_value
    
    # Import from new packages
    from dataknobs_structures import Tree as NewTree
    from dataknobs_structures import tree as new_tree_module
    from dataknobs_utils.json_utils import get_value as new_get_value
    from dataknobs_utils import json_utils as new_json
    
    # They should provide the same functionality
    # Create instances and verify they work the same way
    legacy_tree = LegacyTree("test")
    new_tree = NewTree("test")
    assert legacy_tree.data == new_tree.data
    
    # Also test via module access
    legacy_tree2 = legacy_tree_module.Tree("test2")
    new_tree2 = new_tree_module.Tree("test2")
    assert legacy_tree2.data == new_tree2.data
    
    # Test that functions work the same
    test_dict = {"a": {"b": "c"}}
    assert legacy_get_value(test_dict, "a.b") == new_get_value(test_dict, "a.b")
    assert legacy_json.get_value(test_dict, "a.b") == new_json.get_value(test_dict, "a.b")


def test_complex_workflow():
    """Test a complex workflow using multiple packages."""
    from dataknobs_structures import Tree, Text, TextMetaData
    from dataknobs_utils.json_utils import get_value
    from dataknobs_xization.normalize import basic_normalization_fn
    from dataknobs_xization.masking_tokenizer import TextFeatures
    
    # Create a document structure
    metadata = TextMetaData(text_id=1)
    doc = Text("Hello WORLD! 123", metadata)
    
    # Normalize the text
    normalized = basic_normalization_fn(doc.text)
    assert normalized == "hello world! 123"
    
    # Extract features - TextFeatures provides tokenization features
    features = TextFeatures(doc.text)
    assert features.text == "Hello WORLD! 123"
    assert features.cdf is not None  # Character dataframe exists
    
    # Create a tree structure
    tree = Tree({"text": doc.text, "normalized": normalized})
    child = tree.add_child({"features": {"text_length": len(doc.text)}})
    
    # Use json utils to navigate the tree data
    tree_data = tree.data
    text_value = get_value(tree_data, "text")
    assert text_value == "Hello WORLD! 123"


def test_package_versions():
    """Test that all packages have version information."""
    import dataknobs_structures
    import dataknobs_utils
    import dataknobs_xization
    import dataknobs_common
    import dataknobs
    
    # All packages should have version info
    assert hasattr(dataknobs_structures, "__version__")
    assert hasattr(dataknobs_utils, "__version__")
    assert hasattr(dataknobs_xization, "__version__")
    assert hasattr(dataknobs_common, "__version__")
    assert hasattr(dataknobs, "__version__")
    
    # Versions should be strings
    assert isinstance(dataknobs_structures.__version__, str)
    assert isinstance(dataknobs_utils.__version__, str)
    assert isinstance(dataknobs_xization.__version__, str)
    assert isinstance(dataknobs_common.__version__, str)
    assert isinstance(dataknobs.__version__, str)