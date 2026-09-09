"""Folding a surface form for lookup.

One function, and the module exists because of *who* needs it rather than how
much of it there is. An entity source folds the forms it indexes; a match
signal folds the query it is handed. Those two live in different subpackages,
and while this function shipped inside ``ontology/sources.py`` the second one
could only reach it by importing the first -- an edge from the resolution
family back into ``ontology``, closing a cycle through that package's
``__init__``. Hoisting it here is what makes the edge one-way.

Dependency-free on purpose: it imports nothing, so anything may import it.
"""

from __future__ import annotations

__all__ = ["default_normalizer"]


def default_normalizer(form: str) -> str:
    """Fold a surface form for lookup: strip, then case-fold.

    ``casefold`` rather than ``lower`` because it folds more than ASCII --
    a vocabulary in German should match ``STRASSE`` against ``straße``, and
    ``lower`` does not.
    """
    return form.strip().casefold()
