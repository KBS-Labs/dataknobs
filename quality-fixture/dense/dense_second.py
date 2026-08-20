"""A second file in the same cell, so that a breach can rank its offenders.

``check`` names the files behind an exceeded ceiling largest first, and a cell
holding one file cannot tell a ranking from a listing. This one holds fewer
findings than its sibling, which is what makes the order it comes back in mean
something.

See ``dense_first.py`` for why a fixture that is clean to the gate can still be
counted here.
"""

from __future__ import annotations


def labelled(values: list[str], type: str) -> list[str]:
    """Return every value prefixed with its type name."""
    labels: list[str] = []
    for value in values:
        labels.append(f"{type}:{value}")
    return labels
