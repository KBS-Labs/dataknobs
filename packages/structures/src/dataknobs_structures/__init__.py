"""Data structures for AI knowledge bases."""

from dataknobs_structures.conditional_dict import cdict
from dataknobs_structures.document import Text, TextMetaData
from dataknobs_structures.record_store import RecordStore
from dataknobs_structures.tree import Tree, build_tree_from_string

__version__ = "1.0.0"

__all__ = [
    "RecordStore",
    "Text",
    "TextMetaData",
    "Tree",
    "build_tree_from_string",
    "cdict",
]
