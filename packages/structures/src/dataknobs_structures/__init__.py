"""Data structures for AI knowledge bases."""

from dataknobs_structures.tree import Tree, build_tree_from_string
from dataknobs_structures.document import Text, TextMetaData
from dataknobs_structures.record_store import RecordStore
from dataknobs_structures.conditional_dict import cdict

__version__ = "1.0.0"

__all__ = [
    "Tree",
    "build_tree_from_string", 
    "Text",
    "TextMetaData",
    "RecordStore",
    "cdict",
]