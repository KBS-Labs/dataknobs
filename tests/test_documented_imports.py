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

**The reader had the same shape of hole in it**, and held it for longer. It
took one physical line at a time, so a parenthesized import arrived at
``ast.parse`` as the fragment ``from dataknobs_data import (`` -- which starts
with ``from``, carries the namespace, and counts as a statement by every test
applied here. ``ast`` cannot read a fragment, and the empty list it produced is
the same answer a statement importing nothing from the namespace gives, so the
whole form cost one silent zero apiece: 227 statements, 958 unread targets
against the 3,055 it was reading, and 21 of the unread did not resolve. A
fence indented under a list item was never opened at all, for the same kind of
reason. Neither could fail anything, and the statement tally a non-vacuity
check watches was identical either way. What holds them now is that a statement the reader
assembles must itself parse -- an assembly that comes out wrong is a finding
rather than a zero -- and that the floor below counts *targets*, which is the
quantity a line-at-a-time reader cannot reach.

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

from tests._workspace import HISTORICAL, ROOT, code_fences, documentation_files, rel

NAMESPACE = "dataknobs"
#: The fence languages whose contents are read as Python.
PYTHON_FENCE = frozenset({"python", "py"})

#: Marks the *next* fence as one whose imports are not meant to resolve.
#:
#: A migration guide's "Before" block, or a FAQ entry whose subject is the
#: mistake itself, contains a non-resolving import on purpose -- the wrongness
#: is the content. An allowlist file would record those centrally, where the
#: doc's author never sees it and a moved block leaves a stale entry behind.
#: The marker travels with the block instead, is invisible in rendered output,
#: and carries its own reason.
ILLUSTRATIVE = re.compile(r"^dk-imports:\s*illustrative\b")


def _depth(line: str) -> int:
    """Net parenthesis depth of an import line, ignoring a trailing comment.

    An import statement cannot contain a string literal, so the only place a
    parenthesis hides from a count is after a ``#``.
    """
    return line.split("#", 1)[0].count("(") - line.split("#", 1)[0].count(")")


def import_statements(path: Path) -> list[tuple[int, str]]:
    """``(line number, source)`` for each namespace import inside a py fence.

    Prose outside a fence is not a claim that anything imports, and a fence
    carrying the illustrative marker is a claim that something does *not*.

    A statement is assembled across its lines rather than taken one line at a
    time, and is reported at the line it opens on. The line-at-a-time reader
    this replaced could not see a parenthesized import at all: it collected the
    opening ``from dataknobs_data import (`` -- which starts with ``from ``,
    carries the namespace, and is a statement by every test applied here -- and
    every name inside the parentheses went unread. 227 such statements were in
    the tree, hiding 958 targets against the 3,055 being read, 21 of which did
    not resolve while this file reported green over all of them.

    Assembled lines are rejoined with newlines rather than spaces, and the
    difference is not cosmetic: a name in one of these blocks is very often
    followed by a ``#`` comment explaining it, and one line's comment run
    together with the next swallows every name after it. That produced a
    statement that would not parse, which is how ``unreadable`` reported the
    mistake the first time this reader was run over the tree.

    A fragment whose parentheses never close is emitted anyway rather than
    dropped on the floor. Dropping it would replace one silent zero with
    another, which is the defect this file exists to refuse; ``unreadable``
    reports it instead.
    """
    statements: list[tuple[int, str]] = []
    for fence in code_fences(path):
        if fence.lang not in PYTHON_FENCE or ILLUSTRATIVE.match(fence.marker or ""):
            continue
        pending: list[str] = []
        opened = depth = 0
        for offset, raw in enumerate(fence.lines):
            line = raw.strip()
            if pending:
                pending.append(line)
                depth += _depth(line)
                if depth <= 0:
                    statements.append((opened, "\n".join(pending)))
                    pending, depth = [], 0
                continue
            if line.startswith(("from ", "import ")) and NAMESPACE in line:
                depth = _depth(line)
                if depth > 0:
                    pending, opened = [line], fence.line + offset
                else:
                    statements.append((fence.line + offset, line))
        if pending:  # a statement whose parentheses never close still owes a report
            statements.append((opened, "\n".join(pending)))
    return statements


def parsed(statement: str) -> ast.Module | None:
    """The statement's tree, or ``None`` if it cannot be read as Python.

    The single place that decides a statement is unreadable, because two
    places deciding it is how one of them stops agreeing with the other.
    """
    try:
        return ast.parse(statement)
    except SyntaxError:
        return None


def targets(statement: str) -> list[tuple[str, str | None]]:
    """``(module, attribute)`` pairs a statement asserts the existence of.

    An unreadable statement answers empty, which is also what a statement
    naming nothing in the namespace answers -- so on its own this function
    cannot tell "names nothing" from "could not be read". ``unreadable`` is
    what makes the second case reportable, and the pairing is the only reason
    the empty answer here is safe.
    """
    tree = parsed(statement)
    if tree is None:
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


def unreadable() -> list[str]:
    """Every collected statement this file cannot read as Python.

    The reader assembles a statement from the lines a document spreads it
    over, and an assembly that comes out wrong produces a string ``ast`` will
    not parse. ``targets`` answers that string with an empty list -- the same
    answer it gives a statement importing nothing from the namespace -- so
    without this the mis-read costs one silent zero and reports nothing at all.

    That is not a hypothetical failure mode; it is the one this file shipped
    with. Every parenthesized import in the tree arrived at ``ast.parse`` as
    the fragment ``from dataknobs_data import (``, and the silence was
    indistinguishable from a clean scan for as long as it lasted.
    """
    return [
        f"{rel(path)}:{number}  {statement}"
        for path in documentation_files()
        for number, statement in import_statements(path)
        if parsed(statement) is None
    ]


def test_no_documented_import_is_unreadable() -> None:
    """The reader's own output must be Python, or its silence means nothing."""
    found = unreadable()
    assert not found, (
        f"{len(found)} collected statement(s) do not parse, so every name in "
        "them is unchecked and the import guard is quietly narrower than it "
        "reports:\n  " + "\n  ".join(found) + "\n\nEither the document holds an "
        "import that is not valid Python, or the reader assembled it wrongly "
        "-- and the second is a defect in this file, not in the document."
    )


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

    named = sum(
        len(targets(statement)) for path in files for _, statement in import_statements(path)
    )
    assert named > 3500, (
        f"the scan collects {scanned} statements naming only {named} targets. A "
        "statement count cannot see the narrowing this floor is for: a reader "
        "that stops at the first physical line still collects the opening "
        "fragment of a parenthesized import, so the statement total is "
        "unchanged while every name inside the parentheses goes unchecked. "
        "This floor sits above what such a reader can reach."
    )


def test_a_parenthesized_import_is_read_across_its_lines(tmp_path: Path) -> None:
    """The form the reader stopped at, and the silence it produced.

    A reader taking one physical line at a time collects
    ``from dataknobs_data import (`` -- which starts with ``from ``, carries the
    namespace, and counts as a statement -- and then hands that fragment to
    ``ast.parse``, which cannot read it. Every name inside the parentheses is
    invisible, and the statement tally is exactly what it would be if the
    import had been on one line, so nothing anywhere reports a narrowing.
    """
    doc = tmp_path / "sample.md"
    doc.write_text("```python\nfrom dataknobs_data import (\n    Record,\n    Query,\n)\n```\n")
    found = import_statements(doc)
    assert len(found) == 1, f"expected one statement, got {found}"
    number, statement = found[0]
    assert number == 2, f"a statement is reported at its opening line, not {number}"
    assert targets(statement) == [("dataknobs_data", "Record"), ("dataknobs_data", "Query")]


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param("```\n", id="fence-closes"),
        pytest.param("", id="fence-never-closes"),
    ],
)
def test_an_unclosed_import_is_reported_rather_than_dropped(tmp_path: Path, tail: str) -> None:
    """Accumulation must not become a second way to see nothing.

    A parenthesis that never closes leaves the reader holding a fragment, at
    the end of the fence and again at the end of the file. Discarding it either
    time would be this file's own defect wearing the fix's clothes, so it is
    emitted and ``unreadable`` names it.
    """
    doc = tmp_path / "sample.md"
    doc.write_text("```python\nfrom dataknobs_data import (\n    Record,\n" + tail)
    found = import_statements(doc)
    assert len(found) == 1, f"the fragment must survive to be reported, got {found}"
    assert parsed(found[0][1]) is None, "the fragment is not Python and must say so"
    assert not targets(found[0][1]), "an unreadable fragment names nothing"


def test_unreadable_reports_one_when_the_tree_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the readability check, the way the star check gets one.

    ``test_no_documented_import_is_unreadable`` runs over a tree where nothing
    is unreadable, so it passes whether the body below still detects anything
    or not -- and a guard against silence that has itself gone silent is the
    exact shape this file is about.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_data import (\n"
        "    Record,\n"
        "```\n"
        "```python\n"
        "from dataknobs_data import Query\n"
        "```\n"
    )
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    found = unreadable()

    assert len(found) == 1, f"expected the one unreadable fragment, got {found}"
    assert "sample.md:2" in found[0], f"wrong line reported: {found[0]}"


def test_a_fence_indented_under_a_list_item_is_still_read(tmp_path: Path) -> None:
    """A fence nested in a list is indented, and was therefore never opened."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "1. Import it:\n\n    ```python\n    from dataknobs_data import Record\n    ```\n"
    )
    assert [statement for _, statement in import_statements(doc)] == [
        "from dataknobs_data import Record"
    ]


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
