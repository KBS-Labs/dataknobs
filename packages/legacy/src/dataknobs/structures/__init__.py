"""Re-export structures from dataknobs-structures package."""

from dataknobs._aliasing import alias_submodules

# Import the submodules explicitly to make them available
from dataknobs_structures import conditional_dict, document, record_store, tree

# Also import the main exports for backward compatibility
from dataknobs_structures.conditional_dict import cdict
from dataknobs_structures.document import MetaData, Text, TextMetaData
from dataknobs_structures.record_store import RecordStore
from dataknobs_structures.tree import Tree, build_tree_from_string

# Attribute binding alone leaves ``dataknobs.structures.<name>`` unresolvable as a
# dotted module path, which is the form pre-split code uses.
alias_submodules(__name__, (conditional_dict, document, record_store, tree))

__all__ = [
    "MetaData",
    "RecordStore",
    "Text",
    "TextMetaData",
    "Tree",
    "build_tree_from_string",
    "cdict",
    "conditional_dict",
    "document",
    "record_store",
    "tree",
]
