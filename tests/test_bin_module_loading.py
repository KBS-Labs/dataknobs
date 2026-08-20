"""The loader every guard over ``bin/`` shares must read the file, not a cache.

Ten test modules reach their subject through ``load_bin_module``. If that helper
can hand back a previous version of a script, then every assertion in all ten is
made against code that may not be the code on disk — and the failure is silent,
because a stale module answers to the same names as a fresh one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests._workspace import load_bin_module, load_module_from_path

#: Two bodies of the same length, so the only thing distinguishing them on disk
#: is their content. CPython's ``__pycache__`` validity check compares the
#: source's size and its mtime truncated to the second, and compares nothing
#: else — so a same-length rewrite inside one second is invisible to it.
BEFORE = 'def answer():\n    return "before"\n'
AFTER = 'def answer():\n    return "after."\n'


def test_a_same_second_same_length_edit_is_not_served_from_bytecode(tmp_path: Path) -> None:
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

    script = tmp_path / "subject.py"
    script.write_text(BEFORE, encoding="utf-8")
    assert load_module_from_path("subject", script).answer() == "before"

    # No sleep, deliberately: the collision needs both writes inside one second,
    # which is what makes this the red/green cycle rather than a contrived one.
    script.write_text(AFTER, encoding="utf-8")
    served = load_module_from_path("subject", script).answer()

    assert served == "after.", (
        "the loader served a previous version of the script. Every guard over "
        "bin/ then measures code that is not the code on disk, and a developer "
        "proving a guard can fail gets an answer about the version they replaced."
    )


def test_the_loader_leaves_no_cache_entry_behind(tmp_path: Path) -> None:
    """Not merely a fresh read — no ``__pycache__`` entry is created either.

    Deleting a stale entry on the way in repairs the immediate read. Declining
    to write a new one is what stops the *next* process inheriting the problem,
    and it is the half that would go unnoticed: a loader that unlinks and then
    rewrites passes the test above on every run while leaving a live trap on
    disk for anything that loads the script by ordinary import.
    """
    script = tmp_path / "leaves_nothing.py"
    script.write_text(BEFORE, encoding="utf-8")
    load_module_from_path("leaves_nothing", script)

    cached = Path(importlib.util.cache_from_source(str(script)))
    assert not cached.exists(), (
        f"{cached.name} was written. The next load compares against it, so the "
        "hazard is reintroduced for every process after this one."
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
