"""Re-export xization from dataknobs-xization package."""

# Import the submodules explicitly to make them available
from dataknobs_xization import (
    annotations,
    authorities,
    lexicon,
    masking_tokenizer,
    normalize,
)

# Also import commonly used classes and functions for backward compatibility
from dataknobs_xization.masking_tokenizer import CharacterFeatures, TextFeatures
from dataknobs_xization.normalize import basic_normalization_fn

# Make submodules available as attributes
__all__ = [
    "CharacterFeatures",
    "TextFeatures",
    "annotations",
    "authorities",
    "basic_normalization_fn",
    "lexicon",
    "masking_tokenizer",
    "normalize",
]
