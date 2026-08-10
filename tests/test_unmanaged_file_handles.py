"""Reproduce-first guard for file handles opened outside a context manager.

There is a standard detector for this, and it does not reach the spelling this
repository uses. ``SIM115`` was declined workspace-wide with the reason "not
practical for locks" -- true of one file, which holds a lock handle open across
the critical section it protects -- so for every other file the check was simply
off. Narrowing that waiver to the thirteen files that own a handle deliberately
turned the rule back on everywhere else, and it still would not have caught the
leak that prompted this: measured against ruff 0.16.1 with ``--isolated``,
``SIM115`` flags a builtin ``open(...)`` bound to a name and reports **nothing**
for ``Path.open(...)`` -- neither assigned nor method-chained.

``bin/package-hashes.py`` read a shebang with ``path.open("rb").readline()``,
which is both spellings the rule misses at once, in a file ruff lints on every
run. Refcounting closes it on CPython, so nothing observable was wrong; the
defect is that the handle's release depends on an implementation detail of the
interpreter rather than on the code.

So this is the same disposition A21(d) reached one layer down, where shellcheck
reported nothing at any severity for a live argument-parsing bug: when the
standard detector does not reach the class, the guard is hand-written. Scoped to
the code the workspace guards own -- the directories ``workspace_targets``
declares -- because that is where this class was found and where a handle owned
past its opening expression has no legitimate instance today. Package code is
out of scope: thirteen files there own handles on purpose, and telling those
apart is a judgement the linter's per-file waivers already record.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._workspace import ROOT, rel, tracked_files, workspace_targets

#: Enough scanned files that an empty finding list means "clean" rather than
#: "matched nothing". Set well under the real count (32 when this was written)
#: so ordinary growth and deletion do not move it, but far enough above zero
#: that a scope expression which resolves to nothing fails instead of passing.
MINIMUM_FILES_SCANNED = 20


def _workspace_python() -> list[Path]:
    """Every tracked ``*.py`` under a declared workspace target."""
    roots = workspace_targets()
    found = []
    for name in tracked_files():
        if not name.endswith(".py"):
            continue
        if name in roots or any(name.startswith(f"{root}/") for root in roots):
            found.append(ROOT / name)
    return found


def _unmanaged_opens(source: str, path: Path) -> list[tuple[int, str]]:
    """Calls to ``open`` / ``.open`` whose result no ``with`` statement owns.

    Matched on the attribute name rather than on a resolved type, so
    ``socket.open`` or a caller's own ``.open()`` would be reported too. That
    direction is deliberate: this scope has no such call today, and a guard that
    tries to prove the receiver is a ``Path`` has to reimplement type inference
    to answer a question a reader can settle by looking.
    """
    tree = ast.parse(source, str(path))
    managed = {
        id(item.context_expr)
        for node in ast.walk(tree)
        if isinstance(node, ast.With | ast.AsyncWith)
        for item in node.items
    }
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in managed:
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "open":
            found.append((node.lineno, f"{ast.unparse(func)}(...)"))
        elif isinstance(func, ast.Name) and func.id == "open":
            found.append((node.lineno, "open(...)"))
    return found


def test_no_workspace_file_is_opened_outside_a_context_manager():
    """The guard. A handle whose release depends on refcounting is a leak."""
    violations = [
        f"{rel(path)}:{lineno}: {call} — bind it with `with` so the handle is "
        f"released by the block rather than by the collector"
        for path in _workspace_python()
        for lineno, call in _unmanaged_opens(path.read_text(encoding="utf-8"), path)
    ]

    assert not violations, (
        "File handles opened outside a context manager:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\nSIM115 does not report these: it sees a builtin open() bound to a "
        "name and nothing else, so Path.open() in either form is invisible to "
        "it. That is why this check is hand-written."
    )


def test_the_guard_examines_the_files_it_claims_to():
    """A scope resolving to nothing passes the check above for the wrong reason.

    Two failure modes, one floor each: ``workspace_targets`` naming directories
    that no longer hold Python, and an extension filter that stops matching. Both
    leave the assertion above with an empty loop and a green result, which reads
    from the report exactly like a tree with no leaks.
    """
    scanned = _workspace_python()
    assert len(scanned) >= MINIMUM_FILES_SCANNED, (
        f"only {len(scanned)} workspace Python files were scanned, below the "
        f"floor of {MINIMUM_FILES_SCANNED} — the scope has narrowed and the "
        f"check above is passing over code it no longer reads. Targets: "
        f"{list(workspace_targets())}"
    )

    parsed = sum(
        1
        for path in scanned
        if "open(" in path.read_text(encoding="utf-8")
    )
    assert parsed, (
        "no scanned file contains an `open(` call at all, so the AST walk has "
        "nothing to reject and would pass against any implementation of it"
    )


def test_the_guard_detects_the_shape_it_exists_for():
    """Mutation test, including the two spellings SIM115 reports nothing for.

    The chained form is the one that shipped, and it is the one a reader is
    least likely to notice: with no name bound, there is nothing on the line
    that looks like a resource.
    """
    probe = Path("probe.py")
    cases = {
        "chained Path.open": 'def f(p):\n    return p.open("rb").readline()\n',
        "assigned Path.open": 'def f(p):\n    fh = p.open("rb")\n    return fh.read()\n',
        "assigned builtin open": 'def f(n):\n    fh = open(n, "rb")\n    return fh.read()\n',
    }
    for label, source in cases.items():
        assert _unmanaged_opens(source, probe), f"not detected: {label}"

    managed = 'def f(p):\n    with p.open("rb") as fh:\n        return fh.read()\n'
    assert not _unmanaged_opens(managed, probe), (
        "a `with`-managed open was reported, so the check cannot distinguish "
        "the defect from the fix and every correct call site would fail it"
    )
