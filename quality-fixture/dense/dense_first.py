"""Deliberate findings, for the guards that need a cell measuring above zero.

Clean under the repository's own ruff configuration and dirty under
``quality-fixture/ruff.toml``, which is the whole trick: the rules that report
on the code below are ones the root ``pyproject.toml`` does not select. So the
gate lints this file and finds nothing — the ``quality-fixture`` cell sits at a
ceiling of zero like every other ruff cell — while the guards over
``bin/quality-contract.py`` still have something to count.

Nothing imports this. It is read by ruff and by nothing else, so it is written
to be *measured* rather than to be run: the shapes below are chosen for the
rules they trip, not because any of them is worth writing. **Do not tidy them
up.** Clearing a finding here empties the cell those guards are driven over, and
they fail saying so rather than passing over nothing.
"""

from __future__ import annotations


def widen(value: int, flag: bool = False) -> int:
    """Return ``value``, doubled when ``flag`` is set."""
    doubled = value * 2 if flag else value
    return doubled


def render(values: list[int]) -> list[str]:
    """Return every value as text."""
    rendered: list[str] = []
    for value in values:
        rendered.append(str(value))
    return rendered


def named(type: str) -> str:
    """Return a declared type's name, refusing an empty one."""
    if not type:
        raise ValueError("a type name may not be empty")
    return type


def sized(values: list[int], id: int) -> int:
    """Return the value at ``id``, refusing an index outside the list."""
    if id >= len(values):
        raise IndexError("that index is past the end of the list")
    total = values[id]
    return total
