"""Reproduce-first guard: an import shown in the docs must be one that resolves.

A code sample is a promise that the reader can paste it and have it work. The
first line of nearly every sample is an import, and an import is the one part of
a sample this repository can check mechanically -- so it is the part with no
excuse for being wrong.

It was wrong in 218 places when this was written, across 76 files, out of 3,211
documented targets in the ``dataknobs`` namespace. Not one of them could fail
anything: ``mkdocs build --strict`` validates links and nav, the doc-mirror
manifest validates that two copies agree, and neither reads what is *inside* a
fence. Two copies of a broken example agree with each other perfectly.

The population was not one mistake repeated. It was four, and they are worth
naming because the next hundred will be one of them again:

- **A rename the docs did not follow.** ``MemoryDatabase`` became
  ``SyncMemoryDatabase`` and the docs kept the old spelling. Mechanical, and the
  overwhelming majority.
- **A module that never existed.** ``dataknobs_llm.benchmarks`` was documented
  in two places, in detail, with a framework overview. There is no such module.
- **An API generation that was replaced wholesale.** A validation page
  documented twenty-odd constraint classes against a module that has nine.
- **A compatibility promise the code did not keep.** The legacy shims bound
  their submodules as attributes without registering them, so
  ``from dataknobs.structures.tree import Tree`` -- the form in the migration
  guide, both READMEs, and every pre-split user's code -- raised
  ``ModuleNotFoundError``. There the docs were right and the package was wrong,
  which is the case a guard scoped to "fix the docs" would have mis-diagnosed.

**The check is import resolution, not execution.** It proves the name exists at
the path shown; it cannot prove the call beneath it passes the right arguments.
That boundary is real and was paid for: repointing an import while leaving the
body calling the old name produces a sample that looks corrected and fails on
its second line, which is worse than one that fails on its first. Renaming a
symbol at its uses is therefore part of fixing an import here, and is the
reviewer's job rather than this guard's.

**A star import is the one form that satisfies the check while defeating it**,
which is why it is now refused outright below. The module it names really does
exist, so the fence reads as clean while every name used beneath it goes
unchecked -- and that is where the third class above was still living after the
first sweep: two pages had been documenting a replaced constraint and migration
API behind `import *`, reporting green throughout.

Scope is every markdown document a reader can reach: the site tree, each
package's ``docs/``, and the READMEs. Two carve-outs, both narrow and both
stated in the code below rather than left to a path convention.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from tests._workspace import HISTORICAL, ROOT, documentation_files, rel

NAMESPACE = "dataknobs"
FENCE_OPEN = re.compile(r"^```(?:python|py)\b", re.IGNORECASE)
FENCE_CLOSE = re.compile(r"^```\s*$")

#: Marks the *next* fence as one whose imports are not meant to resolve.
#:
#: A migration guide's "Before" block, or a FAQ entry whose subject is the
#: mistake itself, contains a non-resolving import on purpose -- the wrongness
#: is the content. An allowlist file would record those centrally, where the
#: doc's author never sees it and a moved block leaves a stale entry behind.
#: The marker travels with the block instead, is invisible in rendered output,
#: and carries its own reason.
ILLUSTRATIVE = re.compile(r"^<!--\s*dk-imports:\s*illustrative\b")


def import_statements(path: Path) -> list[tuple[int, str]]:
    """``(line number, source)`` for each namespace import inside a py fence.

    Prose outside a fence is not a claim that anything imports, and a fence
    carrying the illustrative marker is a claim that something does *not*.
    """
    statements: list[tuple[int, str]] = []
    inside = marked = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not inside:
            if ILLUSTRATIVE.match(raw.strip()):
                marked = True
            elif FENCE_OPEN.match(raw):
                inside = True
            elif raw.strip():
                marked = False
            continue
        if FENCE_CLOSE.match(raw):
            inside = marked = False
            continue
        line = raw.strip()
        if not marked and line.startswith(("from ", "import ")) and NAMESPACE in line:
            statements.append((number, line))
    return statements


def targets(statement: str) -> list[tuple[str, str | None]]:
    """``(module, attribute)`` pairs a statement asserts the existence of."""
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return []
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [
                (alias.name, None) for alias in node.names if alias.name.startswith(NAMESPACE)
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(NAMESPACE):
                found += [(node.module, alias.name) for alias in node.names]
    return found


def unresolved(module: str, attribute: str | None) -> str | None:
    """Why this target does not resolve, or ``None`` if it does.

    Every backend documented here imports an optional third-party driver, so a
    missing driver is an ordinary property of the environment rather than a
    defect in the document. It is told apart from a wrong path the same way
    ``packages/data/tests/conftest.py`` tells them apart, and for the same
    reason: conflating the two is what let a dead import sit unreported.
    """
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing and missing != module and not module.startswith(f"{missing}."):
            return None
        return f"no module {module!r}"
    except Exception as exc:  # a module that imports but explodes is a finding
        return f"{module!r} raised {type(exc).__name__}: {exc}"

    if attribute is None or attribute == "*":
        return None
    if hasattr(loaded, attribute):
        return None
    submodule = f"{module}.{attribute}"
    try:
        importlib.import_module(submodule)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing and missing != submodule and not submodule.startswith(f"{missing}."):
            return None
        return f"{module!r} exports no {attribute!r}"
    except Exception:  # importable at all is enough to call the name present
        return None
    return None


def findings() -> list[str]:
    """Every documented target that does not resolve, as reader-facing text."""
    broken = []
    for path in documentation_files():
        for number, statement in import_statements(path):
            for module, attribute in targets(statement):
                why = unresolved(module, attribute)
                if why:
                    broken.append(f"{rel(path)}:{number}  {statement}\n      {why}")
    return broken


def star_imports() -> list[str]:
    """Every ``from dataknobs... import *`` in a fence the guard is meant to check."""
    return [
        f"{rel(path)}:{number}  {statement}"
        for path in documentation_files()
        for number, statement in import_statements(path)
        for _, attribute in targets(statement)
        if attribute == "*"
    ]


def test_every_documented_import_resolves() -> None:
    """The guard itself: no reachable document names an import that is not there."""
    broken = findings()
    assert not broken, (
        f"{len(broken)} documented import(s) do not resolve, so a reader who "
        "pastes the sample gets an ImportError on its first line:\n  "
        + "\n  ".join(broken)
        + "\n\nRepoint the import, and rename the symbol at its uses in the same "
        "fence -- an import fixed alone leaves a sample that fails one line "
        "later. If the import is not meant to resolve, mark the fence with "
        "<!-- dk-imports: illustrative -- why --> instead."
    )


def test_no_documented_star_import() -> None:
    """A star import resolves, and takes every name under it out of reach.

    This is the one import form that satisfies the check above while defeating
    it. ``from dataknobs_data.validation.constraints import *`` names a module
    that genuinely exists, so ``unresolved`` returns nothing and the fence is
    counted as clean -- while every class the block then goes on to use is
    invisible, because no statement ever named one.

    That is not a hypothetical. Two pages sat behind such a line documenting a
    ``Pattern(regex, flags)`` overload that takes one argument, an
    ``AddField(default=...)`` keyword spelled ``default_value``, a
    ``migration.add_operation()`` that is called ``add``, and progress fields
    named ``percentage`` and ``successful`` where the object has ``percent``
    and ``succeeded``. The guard reported green over all of it.

    Naming the imports is also what the reader needs: ``import *`` does not say
    where ``Range`` came from, and a reader who cannot tell cannot look it up.
    A fence whose subject is the star form itself can carry the illustrative
    marker, which is honoured here exactly as it is above.
    """
    found = star_imports()
    assert not found, (
        f"{len(found)} documented star import(s); every name used beneath one is "
        "unverifiable, so the import check silently stops covering the rest of "
        "the block:\n  " + "\n  ".join(found) + "\n\nList the names explicitly."
    )


def test_a_star_import_is_detected(tmp_path: Path) -> None:
    """The detector fires on the form, and the marker still exempts a fence."""
    doc = tmp_path / "sample.md"
    doc.write_text("```python\nfrom dataknobs_data.validation.constraints import *\n```\n")
    statements = import_statements(doc)
    assert [attribute for _, statement in statements for _, attribute in targets(statement)] == [
        "*"
    ]

    doc.write_text(
        "<!-- dk-imports: illustrative -- the star form is the subject -->\n"
        "```python\nfrom dataknobs_data.validation.constraints import *\n```\n"
    )
    assert not import_statements(doc)


def test_star_imports_reports_one_when_the_tree_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the star check: the function itself finds a star import.

    ``test_no_documented_star_import`` calls this function over the real tree,
    which no longer has a star import in it -- so it passes whether the body
    below still matches anything or not. ``test_a_star_import_is_detected``
    exercises the two helpers rather than the function composing them. Between
    them nothing fails if the composition breaks, which is this file's own
    stated failure mode wearing the guard's clothes, so the aggregate gets the
    same treatment ``test_the_scan_actually_reads_imports`` gives the other one.

    Both module-level names ``star_imports`` reads are redirected: the file
    scan, to a tree with exactly one star import in it, and ``rel``, which
    names a path relative to the repository root and cannot name this one.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_data.validation.constraints import *\n"
        "```\n"
        "```python\n"
        "from dataknobs_data import Record\n"
        "```\n"
    )
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    found = star_imports()

    assert len(found) == 1, f"expected the one star import, got {found}"
    assert "import *" in found[0]
    assert "sample.md:2" in found[0], f"wrong line reported: {found[0]}"


def test_the_scan_actually_reads_imports() -> None:
    """Non-vacuity: a scanner matching nothing would pass the check above.

    The fence pattern, the namespace filter and the marker logic all narrow what
    is looked at, and any of them narrowing to zero produces a green run over an
    unread tree -- the failure this whole file exists to prevent, wearing the
    guard's own clothes.
    """
    files = documentation_files()
    assert len(files) > 100, f"only {len(files)} documents in scope"
    scanned = sum(len(import_statements(path)) for path in files)
    assert scanned > 1000, f"only {scanned} imports found; the scan has narrowed"


def test_a_broken_import_is_detected() -> None:
    """The detector fires on the shape of the defect this was written for."""
    assert unresolved("dataknobs_data.backends.postgres_native", "_pool_manager")
    assert unresolved("dataknobs_data.backends.memory", "MemoryDatabase")
    assert unresolved("dataknobs_data", "SyncMemoryDatabase")


def test_a_working_import_is_not_flagged() -> None:
    """And does not fire on the corrected forms, including a bare module."""
    assert unresolved("dataknobs_data.backends", "SyncMemoryDatabase") is None
    assert unresolved("dataknobs_data.backends.postgres", "_pool_manager") is None
    assert unresolved("dataknobs_common", None) is None
    # A star import really does resolve, which is exactly why resolution alone
    # is not enough -- test_no_documented_star_import covers the rest.
    assert unresolved("dataknobs_data.validation.constraints", "*") is None


def test_the_illustrative_marker_suppresses_only_the_block_it_precedes(
    tmp_path: Path,
) -> None:
    """A marker must not leak past its own fence.

    A marker that stayed in effect for the rest of the file would silence every
    later sample in it, and the silence would look exactly like a clean file.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "<!-- dk-imports: illustrative -- the old spelling is the subject -->\n"
        "```python\n"
        "from dataknobs_data import MemoryDatabase\n"
        "```\n"
        "\n"
        "Ordinary prose between the two blocks.\n"
        "\n"
        "```python\n"
        "from dataknobs_data import AlsoNotReal\n"
        "```\n"
    )
    found = import_statements(doc)
    assert [statement for _, statement in found] == ["from dataknobs_data import AlsoNotReal"]


def test_historical_documents_are_excluded_and_say_so() -> None:
    """The carve-out is load-bearing, and the reader is told about it.

    Excluding a document from the guard is only defensible if the reader who
    lands on it learns the same thing the exclusion assumes. If the banner ever
    goes missing the exclusion becomes a silent one, which is the failure mode
    this file is about.
    """
    excluded = [
        path
        for marker in HISTORICAL
        for path in ROOT.rglob("*.md")
        if marker in path.as_posix() and path.is_file()
    ]
    assert excluded, "the historical carve-out matches nothing; it is dead"
    assert not [
        rel(path)
        for path in excluded
        if "**Historical record.**" not in path.read_text(encoding="utf-8")
    ], "excluded from the import guard but carrying no notice to the reader"
