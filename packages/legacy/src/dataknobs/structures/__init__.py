"""Re-export structures from dataknobs-structures package."""

# Import the submodules explicitly to make them available
from dataknobs_structures import conditional_dict, document, record_store, tree

# Also import the main exports for backward compatibility
from dataknobs_structures.conditional_dict import cdict
from dataknobs_structures.document import Text, TextMetaData, MetaData
from dataknobs_structures.record_store import RecordStore
from dataknobs_structures.tree import Tree, build_tree_from_string

# Make submodules available as attributes
__all__ = [
    'conditional_dict',
    'document', 
    'record_store',
    'tree',
    'cdict',
    'Text',
    'TextMetaData',
    'MetaData',
    'RecordStore',
    'Tree',
    'build_tree_from_string',
]
