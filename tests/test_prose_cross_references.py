"""Reproduce-first guard: a test named in prose must be a test that exists.

These guards explain themselves at length, and a good deal of that explanation
is cross-reference — *this* property is the floor under that one, *that* check
is where the same question is asked from the other side. The references are the
connective tissue: they are how a reader who arrives at one check finds the
three it depends on, and they are the only thing recording why a check is
allowed to be narrow, which is usually that something else is broad.

Nothing checked them. Three were wrong when this was written, from three
different eras, each found by a person happening to look. Named in plain text
below rather than in the double backticks the rest of this file uses, because
this check reads that spelling as a claim the name resolves, and it is the
whole point that these do not:

- test_ruff_config_mirror, renamed to ``test_ruff_config_single_source`` when
  the per-package ruff sections were deleted. The pointer was one line away
  from the rename and still missed.
- test_no_cell_names_a_part_of_the_tree_that_is_gone, a forward pointer to the
  replacement for a retired check, naming it as it was going to be called
  rather than as it was. The real one is
  ``test_verify_names_a_cell_that_matches_no_tracked_file``.
- test_entry_points_agree_on_the_failure_type, which kept its old spelling in a
  docstring after a test_single_reference_resolvers_ prefix was added to the
  test itself.

None of these can fail anything. That is exactly what makes them expensive: a
wrong verdict is contradicted by the code eventually, but a wrong *reference*
is read once, by someone already looking for something, and it costs them the
search. It is the same defect as advice naming a deleted script, which
``test_every_path_named_in_advice_exists`` already covers for shell scripts —
this is that check pointed at the harness's own prose.

**It also has to be able to see a file nobody has committed yet.** The first
version enumerated tracked files only, so it never scanned itself: it passed on
this very docstring while the file was untracked, and reported the three names
above only once a commit made it visible. A guard that cannot read the file
being written is green at the moment it is most needed and red afterwards with
no edit in between, which reads as flakiness rather than a finding. Hence
``tracked_and_new_files``.

Scope is ``tests/`` and ``bin/``: the workspace harness talking about itself.
The package suites carry their own cross-references and roughly five of them
are stale in the same way, but sweeping those is a wider job than this one, and
a guard that quietly covered a tenth of the tree while reading as though it
covered all of it would be the failure this whole program is about. The
boundary is named here so it can be moved deliberately.
"""

from __future__ import annotations

import re

from tests._workspace import ROOT, rel, tracked_and_new_files

#: A test name written as code in prose: ``name`` or `name`. Both spellings
#: appear, along with ``:func:`name```, whose inner backticks this also catches.
REFERENCE_RE = re.compile(r"``(test_[a-z0-9_]+)``|`(test_[a-z0-9_]+)`")

#: A test definition at any indentation. Anchoring at column zero would miss
#: every test written as a method on a class, and reading those as undefined
#: would turn a correct reference into a violation -- a guard whose false
#: positives are indistinguishable from its true ones gets suppressed wholesale.
DEFINITION_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test_[a-z0-9_]+)", re.MULTILINE)

#: Names that appear in prose, do not resolve, and should not. Each needs a
#: reason, for the same purpose the reasons on the declined ruff rules serve:
#: an entry with the rationale lost is one nobody can ever argue with again.
ALLOWED_UNRESOLVED = {
    "test_scope": (
        "a key in the quality contract and the hash manifests, not a test -- "
        "it only matches the pattern because the contract names its cells after "
        "what they scope"
    ),
    "test_every_first_party_python_file_is_linted_by_default": (
        "retired; the comment where it appears exists to say so and to explain "
        "what the quality contract replaced it with"
    ),
    "test_the_lint_deferrals_still_describe_the_repository": (
        "retired; same history comment, naming the two checks that took over its two directions"
    ),
    "test_mypy_configs_declare_the_same_search_path": (
        "retired with the second mypy config; the comment where it appears exists "
        "to say why comparing two configs to each other could not catch a search-"
        "path entry both of them omitted, and what replaces it"
    ),
}

#: Floors under the scan. Real figures when written: 34 files, 18 distinct
#: references. Set far enough below that ordinary growth does not move them and
#: far enough above zero that an enumeration resolving to nothing fails --
#: an empty scan and a clean tree are otherwise the same report, and this check
#: is the shape that passes hardest when it has stopped reading anything.
MINIMUM_FILES_SCANNED = 20
MINIMUM_REFERENCES_FOUND = 10


def _harness_files() -> list[str]:
    """Python under ``tests/`` and ``bin/``, committed or not -- the harness."""
    return [
        name
        for name in tracked_and_new_files()
        if name.endswith(".py") and name.startswith(("tests/", "bin/"))
    ]


def _defined_test_names() -> set[str]:
    """Every test function and test module in the tree, committed or not.

    Deliberately the whole tree, not just the harness: a harness docstring
    pointing into a package suite is a legitimate reference, and resolving it
    against a narrower set would report it as broken. Uncommitted files count
    on the same reasoning in reverse -- a reference to a test added in the same
    change as the prose naming it resolves, rather than failing until commit.
    """
    names: set[str] = set()
    for name in tracked_and_new_files():
        if not name.endswith(".py"):
            continue
        stem = name.rsplit("/", 1)[-1][: -len(".py")]
        if stem.startswith("test_"):
            names.add(stem)
        names |= set(DEFINITION_RE.findall((ROOT / name).read_text(encoding="utf-8")))
    return names


def test_every_test_named_in_harness_prose_exists() -> None:
    """The reference is the navigation; a broken one costs the reader the search.

    Failure names the file and line, because the fix is almost always a rename
    that happened one edit away and the reader needs to see which.
    """
    files = _harness_files()
    assert len(files) >= MINIMUM_FILES_SCANNED, (
        f"scanned only {len(files)} harness files, below the floor of "
        f"{MINIMUM_FILES_SCANNED} -- the enumeration has broken, and this check "
        "would report clean prose for files it never opened"
    )

    defined = _defined_test_names()
    assert defined, "no test definitions found anywhere in the tree"

    found = 0
    violations = []
    for name in files:
        for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            for match in REFERENCE_RE.finditer(line):
                referenced = match.group(1) or match.group(2)
                found += 1
                if referenced in defined or referenced in ALLOWED_UNRESOLVED:
                    continue
                violations.append(f"{name}:{number}: names {referenced}, which does not exist")

    assert found >= MINIMUM_REFERENCES_FOUND, (
        f"found only {found} prose references, below the floor of "
        f"{MINIMUM_REFERENCES_FOUND} -- the pattern has stopped matching the way "
        "these files are written, so this check is passing on an empty set"
    )

    assert not violations, (
        "Prose naming tests that do not exist:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\nRename the reference to match the test, or -- if the test is gone "
        "on purpose and the prose exists to record that -- add it to "
        "ALLOWED_UNRESOLVED with the reason."
    )


def test_every_allowed_unresolved_name_is_still_unresolved() -> None:
    """An allowlist entry that has stopped being needed is a hole, not a leftover.

    If a name here comes back into existence -- a retired check restored, a
    contract key renamed to something a test also uses -- the entry stops
    documenting a deliberate absence and starts excusing the next real break of
    that same name from ever being reported.
    """
    defined = _defined_test_names()
    resolved = sorted(name for name in ALLOWED_UNRESOLVED if name in defined)
    assert not resolved, (
        "ALLOWED_UNRESOLVED names tests that now exist:\n"
        + "\n".join(f"  - {name}: {ALLOWED_UNRESOLVED[name]}" for name in resolved)
        + "\n\nDrop the entry. While it stands, a future reference to that name "
        "is excused whether or not it resolves."
    )


def test_every_allowed_unresolved_name_is_actually_referenced() -> None:
    """An entry matching nothing is stale, and stale entries make the list unreadable.

    The same property the quality contract holds over its deferral cells: a
    waiver for something that is not there any more cannot be distinguished from
    a waiver that is still load-bearing, so the list stops being reviewable at
    the point it stops being exact.
    """
    referenced: set[str] = set()
    for name in _harness_files():
        text = (ROOT / name).read_text(encoding="utf-8")
        for match in REFERENCE_RE.finditer(text):
            referenced.add(match.group(1) or match.group(2))

    unused = sorted(set(ALLOWED_UNRESOLVED) - referenced)
    assert not unused, (
        "ALLOWED_UNRESOLVED entries that no prose mentions:\n"
        + "\n".join(f"  - {name}: {ALLOWED_UNRESOLVED[name]}" for name in unused)
        + f"\n\nThe prose was edited and the waiver was not. Remove it from "
        f"{rel(ROOT / 'tests' / 'test_prose_cross_references.py')}."
    )
