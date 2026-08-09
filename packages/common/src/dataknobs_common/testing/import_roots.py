"""Declare a test tree's own directory as an import root.

A test suite that shares scaffolding between modules has to name that
scaffolding somehow. Under pytest's ``prepend`` import mode the question never
comes up: pytest inserts each collected file's directory onto ``sys.path`` as a
side effect of collecting it, so a bare ``from _stubs import Fake`` resolves
without anyone declaring anything. Under ``importlib`` pytest touches
``sys.path`` at all, and the same import fails.

That makes the import mode load-bearing for code that never mentions it — and
because a package's ``[tool.pytest.ini_options]`` applies to a single-package
run while the root config applies to a run naming two, *which* mode applies is
decided by the argument list. The same file then imports cleanly or not
depending on how the command was typed.

The fix is to stop relying on the side effect and say what the import root is,
in the tree it belongs to::

    # packages/<pkg>/tests/conftest.py
    from dataknobs_common.testing import declare_import_root

    declare_import_root(__file__)

pytest loads a ``conftest.py`` before collecting anything beside it, in every
invocation that reaches that directory, so the declaration holds for
single-package and whole-workspace runs alike.

**This does not make sibling *test* modules importable by design.** It makes a
directory an import root; importing a collected test module from another
collected test module still imports it twice, under two names, with its
module-level code run twice. Put shared scaffolding in a module pytest does not
collect — an underscore-prefixed name — and import that.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["declare_import_root"]


def declare_import_root(anchor: str | Path) -> Path:
    """Put *anchor*'s directory on ``sys.path`` so its siblings import by name.

    Args:
        anchor: A file inside the directory to declare, normally the calling
            ``conftest.py``'s ``__file__``. A directory is accepted and used
            as-is, so a caller that already holds the path need not fabricate a
            file inside it.

    Returns:
        The resolved directory, so a caller can go on to use it.

    Raises:
        ValueError: the resolved directory does not exist. Putting a
            nonexistent path on ``sys.path`` is not an error Python reports —
            it is a silent no-op — so a typo'd anchor would leave every import
            it was meant to enable failing, with the declaration sitting right
            there looking correct.

    The path is resolved before comparison and inserted at the front only when
    absent. Comparing an unresolved path — or a :class:`~pathlib.Path` against
    the list of strings ``sys.path`` actually holds — never matches, so the
    entry is re-inserted on every call and the list grows for the life of the
    process.
    """
    path = Path(anchor)
    directory = path if path.is_dir() else path.parent
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"cannot declare {str(resolved)!r} an import root: it is not a "
            f"directory. Anchor was {str(anchor)!r}."
        )
    entry = str(resolved)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return resolved
