# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Text a vocabulary is matched over: folding and locating a form, and varying one.

Two halves, each a module of its own, and every name re-exported here so
``dataknobs_common.text.<name>`` is the one import path for both:

- :mod:`~dataknobs_common.text.scan` folds a surface form for lookup and says
  where in a string a form may sit -- the fold, the token spans and the
  punctuation edge rule a scanning match reads. It imports nothing.
- :mod:`~dataknobs_common.text.variants` makes an authored form's variants --
  *Apples* for *Apple*, *Fruit and Veg* for *Fruit & Veg* -- and refuses any
  that would make two entities collide. Its rules are named in a registry, so
  a configuration can name them and a consumer can add their own. It imports
  :mod:`dataknobs_common.registry` and :mod:`dataknobs_common.exceptions`, which
  import only the standard library, so the package still opens no edge back
  into the subpackages that import it.
"""

from dataknobs_common.text.scan import (
    content_span,
    default_normalizer,
    edge_extensions,
    probe_spans,
    punctuation_edges,
    token_spans,
)
from dataknobs_common.text.variants import (
    DEFAULT_RULES,
    VARIANT_RULES,
    Declared,
    Refusal,
    Rule,
    Variant,
    Variants,
    admit,
    ampersand,
    camelcase,
    drop_parentheticals,
    generate,
    hyphen,
    inflect,
    inflect_keeping,
    parenthetical,
    rules_from_config,
    slash_or,
    split,
    strip_prefix,
)

__all__ = [
    "DEFAULT_RULES",
    "VARIANT_RULES",
    "Declared",
    "Refusal",
    "Rule",
    "Variant",
    "Variants",
    "admit",
    "ampersand",
    "camelcase",
    "content_span",
    "default_normalizer",
    "drop_parentheticals",
    "edge_extensions",
    "generate",
    "hyphen",
    "inflect",
    "inflect_keeping",
    "parenthetical",
    "probe_spans",
    "punctuation_edges",
    "rules_from_config",
    "slash_or",
    "split",
    "strip_prefix",
    "token_spans",
]
