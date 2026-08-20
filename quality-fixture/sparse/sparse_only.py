"""The second cell, holding fewer findings than the first.

Two cells rather than one because three of the guards need to watch a command
touch the cell it was given and leave the other exactly as declared — a property
a single cell cannot express. This is the smaller of the two, so the pair also
has a stable order.

See ``../dense/dense_first.py`` for why a fixture that is clean to the gate can
still be counted here.
"""

from __future__ import annotations


def scaled(value: int, factor: int, verbose: bool = False) -> int:
    """Return ``value * factor``, refusing a factor of zero."""
    if factor == 0:
        raise ValueError("scaling by zero discards the value entirely")
    product = value * factor
    return product


def truncated(values: list[str], format: str) -> list[str]:
    """Return every value rendered through ``format``."""
    shown: list[str] = []
    for value in values:
        shown.append(format % value)
    return shown
