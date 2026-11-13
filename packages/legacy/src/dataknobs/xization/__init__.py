# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Re-export xization from dataknobs-xization package.

``authorities``, ``lexicon`` and ``masking_tokenizer`` (with ``CharacterFeatures``
and ``TextFeatures``) are deprecated in ``dataknobs-xization`` and warn when
imported. They are imported here anyway, eagerly, because the dotted paths this
shim publishes resolve only for a module already in ``sys.modules``. So
importing this package raises all three modules' warnings, attributed to
library code (this module, and ``lexicon``), where Python's default filter
hides them.
"""

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
