"""Every loader that executes a script from a path must read the file, not a cache.

Ten test modules reach their subject through ``load_bin_module``. If that helper
can hand back a previous version of a script, then every assertion in all ten is
made against code that may not be the code on disk — and the failure is silent,
because a stale module answers to the same names as a fresh one.

There are two such loaders and there cannot usefully be one. ``bin/`` scripts
have hyphenated names, so a script that needs a sibling needs a loader to get
it — and a shared loader could not itself be reached that way. That makes the
pair irreducible, not accidental, which is why the cases below are driven over
both and why a census closes the file: the danger in an irreducible twin is not
that it exists, it is that a fix lands on one half.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests._workspace import load_bin_module, load_module_from_path, tracked_python_files

#: The loaders, by the file that owns one. ``bin/package-hashes.py`` reaches
#: ``bin/changed-packages.py`` this way in the gate's own path, so a stale read
#: there is not a testing inconvenience: it is the dependency graph and the
#: release-noise stripper answering from a version that is no longer on disk,
#: and the hasher deciding which packages need re-validating on that basis.
LOADERS: dict[str, Callable[[str, Path], ModuleType]] = {
    "tests/_workspace.py": load_module_from_path,
    "bin/package-hashes.py": load_bin_module("package-hashes").load_module_from_path,
}

#: Two bodies of the same length, so the only thing distinguishing them on disk
#: is their content. CPython's ``__pycache__`` validity check compares the
#: source's size and its mtime truncated to the second, and compares nothing
#: else — so a same-length rewrite inside one second is invisible to it.
BEFORE = 'def answer():\n    return "before"\n'
AFTER = 'def answer():\n    return "after."\n'


@pytest.mark.parametrize("owner", sorted(LOADERS))
def test_a_same_second_same_length_edit_is_not_served_from_bytecode(
    owner: str, tmp_path: Path
) -> None:
    """The reproduce-first case, and it reproduces a false *green* and a false *red*.

    Written after this exact pair cost a session a false red: a guard was
    disabled to prove it could fail, restored, and re-run — and the re-run
    reported the same failure, because the restore and the disable were the same
    length and landed in the same second. The code being executed was the
    disabled version; the file being read by ``inspect.getsource`` was the
    restored one. Both were consistent with themselves.

    That cycle is not an unusual thing to do here. It is the mandated one:
    a guard is not done until it has been shown to go red.

    The lengths are asserted rather than eyeballed, because the whole scenario
    rests on them being equal and a later edit to the constants above would
    otherwise turn this into a test that passes for the wrong reason.
    """
    assert len(BEFORE) == len(AFTER), (
        "the two module bodies differ in length, so CPython would invalidate the "
        "cache on size alone and this test would pass without the loader helping"
    )

    load = LOADERS[owner]
    script = tmp_path / "subject.py"
    script.write_text(BEFORE, encoding="utf-8")
    assert load("subject", script).answer() == "before"

    # No sleep, deliberately: the collision needs both writes inside one second,
    # which is what makes this the red/green cycle rather than a contrived one.
    script.write_text(AFTER, encoding="utf-8")
    served = load("subject", script).answer()

    assert served == "after.", (
        f"the loader in {owner} served a previous version of the script. Every "
        "reader of it then measures code that is not the code on disk, and a "
        "developer proving a guard can fail gets an answer about the version "
        "they replaced."
    )


@pytest.mark.parametrize("owner", sorted(LOADERS))
def test_the_loader_leaves_no_cache_entry_behind(owner: str, tmp_path: Path) -> None:
    """Not merely a fresh read — no ``__pycache__`` entry is created either.

    Deleting a stale entry on the way in repairs the immediate read. Declining
    to write a new one is what stops the *next* process inheriting the problem,
    and it is the half that would go unnoticed: a loader that unlinks and then
    rewrites passes the test above on every run while leaving a live trap on
    disk for anything that loads the script by ordinary import.
    """
    script = tmp_path / "leaves_nothing.py"
    script.write_text(BEFORE, encoding="utf-8")
    LOADERS[owner]("leaves_nothing", script)

    cached = Path(importlib.util.cache_from_source(str(script)))
    assert not cached.exists(), (
        f"{owner} wrote {cached.name}. The next load compares against it, so "
        "the hazard is reintroduced for every process after this one."
    )


def test_the_repository_helper_still_loads_a_real_script() -> None:
    """The narrow guarantee above must not have cost the ordinary one.

    ``load_module_from_path`` builds the module through ``module_from_spec``, so
    ``__name__``, ``__file__`` and ``__spec__`` are populated as a normal import
    populates them. A script reading its own ``__file__`` — several under
    ``bin/`` do — would break if this were an ``exec`` into a bare namespace.
    """
    module = load_bin_module("quality-contract")

    assert callable(module.verify), "the loaded module has no verify()"
    assert module.__file__ is not None and module.__file__.endswith("quality-contract.py"), (
        f"__file__ is {module.__file__!r}, so a script resolving paths from it "
        "would resolve them from the wrong place"
    )


def test_no_third_loader_appears_without_the_same_treatment() -> None:
    """The census, and the reason this file ends with one rather than two cases.

    The pair above is enumerated by hand, so the cases prove exactly as much as
    the enumeration is complete — and the way this defect got here was not that
    the fix was wrong, it was that the fix landed on one of the copies while a
    docstring in the other named the one it was leaving behind. A guard driven
    over a hand-written list has the same failure available to it.

    So the list is checked against the tree instead: any file constructing a
    spec from a path is either a loader this file drives, or a loader nothing
    is asserting anything about. There is no third thing it can be.

    Read as syntax rather than as text, which is not fastidiousness: the first
    version matched the substring and its first red was a *docstring* — the one
    two files away explaining that the copy there had been removed. A census a
    comment can trip is a census that gets narrowed the third time it cries
    wolf, and narrowing this one returns it to a hand-written list.
    """
    found = {
        name
        for name in tracked_python_files()
        if any(
            isinstance(node, ast.Attribute | ast.Name)
            and getattr(node, "attr", getattr(node, "id", None)) == "spec_from_file_location"
            for node in ast.walk(ast.parse(Path(name).read_text(encoding="utf-8")))
        )
    }

    assert found == set(LOADERS), (
        f"loaders built from a path: {sorted(found)}, driven by this file: "
        f"{sorted(LOADERS)}. A new one gets the same two steps and a place in "
        "LOADERS; a removed one leaves LOADERS. What is not available is a copy "
        "that no case here reaches — that is the state this file exists to end."
    )
