"""Re-export xization from dataknobs-xization package."""

from dataknobs._aliasing import alias_submodules

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

# Attribute binding alone leaves ``dataknobs.xization.<name>`` unresolvable as a
# dotted module path, which is the form pre-split code uses.
alias_submodules(
    __name__,
    (
        annotations,
        authorities,
        lexicon,
        masking_tokenizer,
        normalize,
    ),
)

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
