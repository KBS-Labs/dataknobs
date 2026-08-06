"""Reproduce-first guard for spellings the interpreter floor made redundant.

Raising a linter's target sweeps what that linter can see. Two of the rules
that matter here see less than their names suggest:

* ``UP041`` rewrites ``except`` and ``raise`` clauses. It does **not** touch a
  tuple literal assigned to a constant, or an ``isinstance`` argument — so the
  same compat spelling survives in exactly the places a reader is most likely
  to copy from.
* ``UP017`` rewrites the call site. The now-unused ``timezone`` import it
  leaves behind is a separate finding, and neither rule looks at docstrings,
  prose, or anything under ``docs/``.

Both spellings became redundant at 3.11 — ``asyncio.TimeoutError`` *is*
``TimeoutError`` and ``datetime.UTC`` *is* ``timezone.utc``, same object — so
this is about what the tree says, not what it does. A tree that still spells
them the old way reads as though it supports an interpreter it does not, and
the next contributor copies the older form from whichever neighbour they land
on.

Lint cannot close this: every entry point targets ``packages/*/src``, so tests
and docs are outside the sweep entirely.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Spelling -> what to use instead. Both are aliases of the same object on the
#: declared floor, so replacing one is a rename, not a behaviour change.
REDUNDANT_SPELLINGS = {
    "asyncio.TimeoutError": "TimeoutError (the same class since 3.11)",
    "timezone.utc": "UTC, from datetime (the same object since 3.11)",
}

def _floor() -> tuple[int, int]:
    """The workspace Python floor, from the root ``requires-python``."""
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "requires-python"
    ]
    match = re.search(r"(\d+)\.(\d+)", requires)
    assert match is not None, f"root requires-python is unparseable: {requires!r}"
    return int(match.group(1)), int(match.group(2))


def _sub_floor_claim_re() -> re.Pattern[str]:
    """``Python <major>.<minor>`` for every minor below the floor.

    Derived from the floor rather than written out, so it covers the next rise
    without an edit. A hand-written range stops where its author stopped: the
    first spelling of this listed 3.0 through 3.10 and so said nothing about
    3.11, which the very bump that prompted it had just dropped.

    Prose that records history — a changelog, a migration note — is not in
    scope; only shipped source is scanned. A version named for a reason that
    survives the floor takes the same ``# sweep-exempt:`` marker as anything
    else here.
    """
    major, minor = _floor()
    olds = "|".join(str(n) for n in range(minor - 1, -1, -1))
    return re.compile(rf"Python {major}\.(?:{olds})\b")

#: A line-level opt-out, for the case where the older spelling *is* the subject
#: under test — a test proving ``asyncio.TimeoutError`` is still caught has to
#: raise one. The reason is mandatory: a bare marker would make this an escape
#: hatch from the guard rather than a documented exception to it.
EXEMPT_RE = re.compile(r"#\s*sweep-exempt:\s*(\S.*)")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _in_history(path: Path) -> bool:
    """Archived design docs record what was written then, not what to write now."""
    return "history" in path.relative_to(ROOT).parts


def _sources() -> list[Path]:
    """Every file whose spelling a contributor might copy.

    Resolved before de-duplication: several site pages are symlinks onto a
    package doc, and reporting one finding twice under two names would send a
    reader looking for a second edit that does not exist.

    The workspace guards in this directory are deliberately outside the sweep.
    They are the files that *name* the redundant spellings — the dict above is
    two of them — so including them would make this guard fail on itself.
    """
    paths: list[Path] = []
    for pattern in ("packages/*/src/**/*.py", "packages/*/tests/**/*.py"):
        paths.extend(ROOT.glob(pattern))
    for pattern in ("packages/*/docs/**/*.md", "docs/**/*.md"):
        paths.extend(ROOT.glob(pattern))
    return sorted({p.resolve() for p in paths if not _in_history(p)})


def _hits(text: str, needle: str) -> list[int]:
    return [
        n
        for n, line in enumerate(text.splitlines(), 1)
        if needle in line and not EXEMPT_RE.search(line)
    ]


def test_no_redundant_compat_spellings():
    """The sweep is complete, including where the linters could not reach."""
    violations = [
        f"{_rel(path)}:{line}: {spelling!r} -> use {replacement}"
        for path in _sources()
        for spelling, replacement in REDUNDANT_SPELLINGS.items()
        for line in _hits(path.read_text(encoding="utf-8"), spelling)
    ]

    assert not violations, (
        "Spellings the interpreter floor made redundant:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_every_exemption_is_a_deliberate_one():
    """The opt-out is bounded, and every use of it is listed here.

    Pinned by count so adding one is an edit to this test, not a comment
    someone can drop in. The failure message names each site, so the reviewer
    sees what was exempted and why rather than a number going up.
    """
    exemptions = [
        f"{_rel(path)}:{n}: {EXEMPT_RE.search(line).group(1)}"
        for path in _sources()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if EXEMPT_RE.search(line)
    ]

    # Four raise sites in the one test that proves the asyncio spelling is
    # still caught after the duplicate tuple entry was dropped.
    assert len(exemptions) == 4, "Sweep exemptions changed:\n" + "\n".join(
        f"  - {e}" for e in exemptions
    )


def test_no_source_claims_a_sub_floor_python():
    """A comment naming a dropped interpreter outlives the code it explained.

    ``composite.py`` carried "on Python 3.10 it does not inherit from
    builtins.TimeoutError" as the stated reason for a tuple entry that had
    become a duplicate. The entry was harmless; the sentence was not, because
    it tells a reader the redundancy is load-bearing.
    """
    pattern = _sub_floor_claim_re()
    violations = [
        f"{_rel(path)}:{n}: {line.strip()}"
        for path in ROOT.glob("packages/*/src/**/*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not EXEMPT_RE.search(line)
    ]

    assert not violations, (
        "Shipped source names a Python below the declared floor:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
