#!/usr/bin/env python3
"""Report every Python file under the given targets that does not compile.

One process for the whole set. ``bin/validate.sh`` used to start ``uv run
python -m py_compile`` once per file, which spent about eighty seconds of every
full run starting interpreters.

``compile`` rather than ``ast.parse``: some errors, such as a ``return`` outside
a function, are raised by the compiler and not by the parser. Nothing is written
to disk.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path


def python_files(targets: Iterable[str]) -> Iterator[Path]:
    """The files named, and every ``*.py`` beneath the directories named.

    A target that is not a directory is yielded as it stands, so one that does
    not exist is reported when it is read rather than passed over.
    """
    for target in map(Path, targets):
        if not target.is_dir():
            yield target
            continue
        for directory, _, names in target.walk():
            yield from (directory / name for name in names if name.endswith(".py"))


def syntax_error(path: Path) -> str | None:
    """What stops ``path`` compiling, or ``None`` when nothing does."""
    try:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"{path}:{exc.lineno or 0}: {exc.msg}"
    except (OSError, ValueError, RecursionError, MemoryError) as exc:
        # ValueError is a null byte on some interpreters; the last two are how
        # the compiler refuses a file nested too deeply.
        return f"{path}: {type(exc).__name__}: {exc}"
    return None


def main(argv: list[str]) -> int:
    paths = sorted({path.resolve(): path for path in python_files(argv[1:])}.values())
    # A SyntaxWarning is not a syntax error, and the linter reports it.
    warnings.simplefilter("ignore")
    errors = [error for path in paths if (error := syntax_error(path)) is not None]
    for error in errors:
        print(f"    ✗ {error}")
    if errors:
        print(f"  ✗ {len(errors)} of {len(paths)} Python files do not compile")
        return 1
    print(f"  ✓ {len(paths)} Python files have valid syntax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
