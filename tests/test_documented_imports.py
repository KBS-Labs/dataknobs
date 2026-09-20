"""Reproduce-first guard: a name shown in the docs must be one that resolves.

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

**The checks are static resolution, not execution.** Three readers now ask
three questions of the same fence, and each was added because the one before it
reported green over a whole population:

- **the name exists at the path shown** -- import resolution, 218 findings;
- **the name beneath the import exists on it** -- attribute resolution, 44,
  because repointing an import while leaving the body calling the old name
  produces a sample that looks corrected and fails on its second line;
- **the call beneath the name could bind** -- signature binding, 91, because a
  class that is really there still fails on the line that constructs it if the
  sample omits an argument it requires.

That progression is the shape to expect of the next one. Each boundary was
handed to the reviewer in this docstring, in these words, and each was found by
measurement rather than by review -- so a boundary named below is a backlog
with a number on it, not a division of labour. What remains outside all three
is a receiver the fence does not import: ``response.content`` on an instance,
``registry.register(tool)`` on a local. Binding also answers only whether the
arguments could be *accepted*, never whether they are the right values.

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

**An import is not the only way a document names something to load**, and for
a long time it was the only way this file could see one. A ``class:`` value in
a YAML fence is handed to ``import_module`` exactly as an ``import`` line is.
Forty-six such paths sit in the tree -- 33 under ``class:``, ten as a ``-m`` or
``--flag`` argument in a shell fence, and the rest under ``function:`` and
``factory:``. Eight named a class that is not there: five on one page of the
development guide, and three in a single README beside two more entries
spelling the same convention correctly. None could fail anything. The
narrowing was the fence language, and it is the same shape as every other one
this file records: a reader whose scope is smaller than the corpus, reporting
green over the difference.

What decides whether a dotted path is a claim is its POSITION, not its text.
The identical token is a claim under ``class:``, a repository in
``git clone https://.../dataknobs.git``, and prose in a comment -- and a sweep
matching the token alone reports the clone URL of this repository as a broken
import nine times over. So the loading positions are enumerated instead, and
both non-claims fall out by construction rather than by an allowlist.

**The second reader has its own floor**, because the first one cannot see it.
Every path it reads sits in a fence the import reader skips, so a ``LOADABLE``
that matched nothing would leave the import floor at its full value and report
a clean sweep of an unread corpus -- the failure this file exists to refuse,
wearing the guard's own clothes for the second time.

**Both readers are scoped to fences, and a curated API reference is not
written in fences.** It is written as a heading naming a fully-qualified path,
followed by a block showing the class it names -- so the claim is in the
heading and the illustration is in the fence, which is the wrong way round for
everything above. One page had documented ``dataknobs_data.Database`` and five
backends beneath it through the rename that made them ``SyncDatabase``,
``SyncMemoryDatabase`` and the rest: six absent names, in the most
authoritative kind of document the site has, none of them able to fail
anything. That is the third of the four classes named above -- an API
generation replaced wholesale -- recurring in a position no reader reached.

**The definition was considered as the position, and measured, and it is not
one.** ``class MemoryDatabase(Database, ConfigurableBase):`` looks like the
claim, since the absent name is right there in it. But a fence is free to
define its own base: ``factory-registration.md`` imports ``ABC`` on line 307,
defines ``class AbstractDatabaseFactory(ABC)`` on 309, and subclasses it on
320 and 327 -- so a reader treating a base as a library name reports that page
twice for a class the page wrote eleven lines above its first use.
A base is a claim only when the fence imports it -- at which point it is an
ordinary import statement and the first reader already has it. So the
definition adds no reach, and the heading, which is prose, has all of it.

**A name that resolves can still be the wrong one to hand a reader.** Every
reader above asks whether a documented name is *there*; a deprecated one is,
which is what makes it invisible to all three and what makes the silence
dangerous rather than untidy. ``ConfigurableBase`` says in its own docstring
that it is superseded, and says why no runtime warning is raised: so the
transition stays quiet across a multi-cycle migration. That is defensible for
consumers who already inherit it, and it has a consequence nobody chose --
documentation becomes the only channel through which a *new* consumer could
learn, and the documentation was the channel recommending it. The bots family
is the same shape and worse, because those four names do warn at runtime: two
guides taught an API that greets the first paste with a ``DeprecationWarning``.
Eleven such silences sat in five documents when this was written.

So the fourth check asks a question about the symbol rather than about the
text, and reuses all three readers to find one. It is scoped per DOCUMENT, not
per site, because a page documenting a deprecated API names it constantly and
is right to -- what separates it from a page teaching the same class in good
faith is whether one paragraph says the word. That also reaches where the
readers cannot: a document is pulled in by any one qualified mention, and the
notice it then has to carry covers every bare prose mention beside it.

**A curated reference also names members, and a renderer that drops one says
nothing.** An ``mkdocstrings`` block lists the members to render under the
path it documents, and an entry naming a member that is not there is rendered
as *nothing at all* -- no warning, and ``mkdocs build --strict`` green over
the page with the entry still in it. So the page reads as complete while a
method the reader went looking for is silently absent from it, and the
absence looks like the method not existing rather than like a stale list.
Seven such entries sat in four pages when this was written, out of 85 across
17 pages. The position is the fifth this file has had to learn, and the same
shape as the fourth: the claim is in the directive's options block, which is
neither a fence nor prose, so no reader above could reach it.

Scope is every markdown document a reader can reach: the site tree, each
package's ``docs/``, and the READMEs. Two carve-outs, both narrow and both
stated in the code below rather than left to a path convention: a document
kept as a historical record, and a block or line declaring that its subject
is the absence.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
import textwrap
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

from tests._workspace import (
    HISTORICAL,
    ROOT,
    code_fences,
    documentation_files,
    prose_lines,
    rel,
)

NAMESPACE = "dataknobs"
#: The fence languages whose contents are read as Python.
PYTHON_FENCE = frozenset({"python", "py"})

#: A dotted path in a position that names something to LOAD.
#:
#: A ``class:`` value in YAML is the same claim as an ``import`` in Python --
#: something will be handed this exact string and asked to produce the object
#: -- but it is not Python, so the reader above never sees it. What decides is
#: POSITION, not the token: the identical text is a claim under ``class:``, a
#: repository in ``git clone https://.../dataknobs.git``, and prose in a
#: comment. Matching the token alone reads all three as claims and reports the
#: clone URL of this very repository as a broken import, nine times.
#:
#: So the value positions are enumerated instead, and the two non-claims fall
#: out by construction rather than by an allowlist: a URL matches no directive,
#: and a comment is cut from the line before this runs. Prose that names a
#: module is the cross-reference guard's business, not this one's.
#:
#: The directives are the ones a runtime entry point actually resolves --
#: ``class`` and ``factory`` (``Config.build_object``), ``chunker``
#: (``create_chunker``), ``custom_class`` (``parse_derivation_rules``),
#: ``merge_filter`` (``load_merge_filter``), and ``function``, which the
#: corpus uses for a hook. ``tests/test_dotted_path_agreement.py`` is the
#: table they come from, and is a better source than this corpus: a key with
#: no dataknobs-namespace use today still resolves one tomorrow. ``chunker``
#: is that case, and so are the last two, whose every documented value names
#: a placeholder package.
#:
#: Three were considered and cut. ``handler`` and ``target`` appear nowhere in
#: the tree, and a directive invented for a guard is surface that can only
#: ever produce a false positive. ``function_ref`` is real -- the rubric
#: registry resolves it -- but the corpus writes it as a keyword argument,
#: ``function_ref="..."``, and an ``=`` is not a position this reads. Adding
#: the word alone would match nothing while reading as coverage, which is the
#: one outcome worse than leaving it out.
#:
#: A key is read with or without the quotes a literal puts around it, and both
#: sides are needed rather than just the leading one: the opening quote is not
#: the whitespace a bare YAML key sits behind, and the closing quote stands
#: between the key and its colon. A config rendered as JSON or as a Python
#: dict makes the same claim as the YAML beside it, and the tree holds one --
#: a ``"class"`` naming an FSM resource provider, inside a python fence, which
#: no reader here could see for as long as only the leading side was allowed.
LOADABLE = re.compile(
    r"""(?:
          (?:^|["'{,\[\s])["']?
          (?:custom_class|merge_filter|class|factory|chunker|function)
          ["']?\s*:\s*["']?
        | -m\s+
        | --[\w-]+[=\s]\s*["']?
        )
        (?P<module>dataknobs[a-z_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)
        (?::(?P<attribute>[A-Za-z_][A-Za-z0-9_]*))?
    """,
    re.VERBOSE,
)

#: A fully-qualified dataknobs path written as code in prose.
#:
#: The third position, and the one the other two readers are structurally
#: unable to reach: it is not an ``import`` statement and it is not a
#: directive value, because it is not in a fence at all. A curated API
#: reference is written almost entirely in this form -- a heading naming the
#: path, then a fence showing the class -- and the heading is the only part of
#: it any reader here can check.
#:
#: Backticks are what make it a claim rather than prose. A sentence saying the
#: memory backend lives in ``dataknobs_data`` is describing; ``a`` set in code
#: font and spelled out to its last segment is naming, and a reader who cannot
#: find what it names has been sent somewhere that does not exist. Requiring
#: the backticks is also what keeps a URL and an ordinary sentence out, by the
#: same construction the directive positions use above.
#:
#: The whole path must sit inside one pair of backticks. ``from x import y`` in
#: code font is the import reader's claim in prose form and is left to it, and
#: no call form -- ``module.function()`` -- appears in the tree at all.
#:
#: **The bare name is deliberately out of reach, and it is the larger corpus.**
#: The other eight curated API pages head their sections ``DynaBot`` and
#: ``BufferMemory`` rather than spelling the module, and a check for those has
#: to ask whether the name exists *somewhere*, which is too weak to act on:
#: swept that way the tree offers 44 unresolved base names across 171 sites, of
#: which ``ConversationMiddleware``, ``AsyncLLMProvider``, ``DatabaseError``
#: and most of the rest are real and merely not top-level. Worse, ``Database``
#: does resolve -- at ``dataknobs_config.examples.Database``, which is not
#: remotely what a data API reference means by it, so the sweep would have
#: reported the page's single worst claim as fine. A qualified path says which
#: module it means and can therefore be wrong about it; a bare name cannot,
#: and a guard that cannot be wrong cannot be right either.
PROSE_PATH = re.compile(r"`(?P<path>dataknobs[a-z_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")

#: Marks a *line* whose path is not meant to resolve, as ``ILLUSTRATIVE``
#: marks a fence.
#:
#: Two kinds of document name an absent thing on purpose, and both are the
#: shape the fence marker already exists for -- the absence is the content:
#:
#: - A changelog's *Removed* entry. ``dataknobs.flask_api`` is named there
#:   precisely because it is gone, and the entry would be false if the name
#:   resolved.
#: - Advice about a mistake. "Use ``dataknobs_package`` not ``dataknobs.package``"
#:   has to spell the wrong form to warn about it.
#:
#: The marker trails the line rather than preceding it, which is the one place
#: this departs from the fence form. A claim in prose is inline, so the line
#: *is* the block; and both sites here sit inside a list, where an HTML comment
#: on its own line interrupts the list in the renderer while the trailing form
#: is invisible.
#:
#: A *Removed* section is where the next one will be, and a section-level
#: exemption is what to reach for if these stop being two. One line each, with
#: its own reason, is the cheaper answer while they are.
PROSE_ILLUSTRATIVE = re.compile(r"<!--\s*dk-imports:\s*illustrative\b.*?-->")

#: Cuts a line at the comment that ends it, in either YAML or shell.
#:
#: Required before ``LOADABLE`` rather than after: ``# dataknobs_bots.middleware.base.``
#: is prose, and a trailing ``# Environment variable with default`` sits on the
#: same line as a real claim.
COMMENT = re.compile(r"(?:^|\s)#.*$")

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


def _imported(module: str) -> tuple[ModuleType | None, BaseException | None]:
    """The module, or the exception that stopped it loading.

    Split out from ``unresolved`` when a third reader needed the same question
    answered about a *prefix* of a path rather than about a whole module, and
    the alternative was a second copy of the driver-vs-typo distinction below.
    """
    try:
        return importlib.import_module(module), None
    except Exception as exc:  # a module that imports but explodes is a finding
        return None, exc


def _absent(module: str, exc: BaseException) -> bool:
    """Whether ``exc`` says *this* module is missing, rather than one under it.

    Every backend documented here imports an optional third-party driver, so a
    missing driver is an ordinary property of the environment rather than a
    defect in the document. The two are told apart the same way
    ``packages/data/tests/conftest.py`` tells them apart, and for the same
    reason: conflating them is what let a dead import sit unreported.
    """
    return isinstance(exc, ModuleNotFoundError) and (
        not exc.name or exc.name == module or module.startswith(f"{exc.name}.")
    )


def _why(module: str, exc: BaseException) -> str | None:
    """Reader-facing text for a failed import, or ``None`` if it is the driver."""
    if isinstance(exc, ModuleNotFoundError):
        return f"no module {module!r}" if _absent(module, exc) else None
    return f"{module!r} raised {type(exc).__name__}: {exc}"


def _resolved(module: str, attribute: str | None) -> tuple[object | None, str | None]:
    """What this target names, and why it names nothing.

    The object is handed back beside the reason because a *second* question is
    asked of these same targets further down -- not whether the name resolves
    but whether what it resolves to is deprecated -- and the alternative was a
    second walk that could disagree with this one about where the module ends.
    """
    loaded, exc = _imported(module)
    if exc is not None:
        return None, _why(module, exc)
    if attribute is None or attribute == "*":
        return loaded, None
    if hasattr(loaded, attribute):
        return getattr(loaded, attribute), None
    submodule = f"{module}.{attribute}"
    found, exc = _imported(submodule)
    if exc is not None and _absent(submodule, exc):
        return None, f"{module!r} exports no {attribute!r}"
    return found, None  # importable at all is enough to call the name present


def unresolved(module: str, attribute: str | None) -> str | None:
    """Why this target does not resolve, or ``None`` if it does."""
    return _resolved(module, attribute)[1]


def _resolved_path(module_path: str) -> tuple[object | None, str | None]:
    """What a bare dotted path names, and why it names nothing.

    ``unresolved`` is handed a module and an attribute because ``ast`` knows
    which is which. A path written in prose does not say, and the last dot is
    the wrong guess: ``dataknobs_bots.memory.VectorMemory.add_message`` names a
    method on a class in a module, and read by the last dot it asks
    ``dataknobs_bots.memory.VectorMemory`` to import. Five such paths sit in
    the tree, every one of them correct, and a last-dot reader reports all five
    as broken -- a false positive indistinguishable from a true one, which is
    the shape that gets a guard suppressed wholesale.

    So the module boundary is found rather than assumed: the longest prefix
    that imports, then attribute access for whatever is left.
    """
    parts = module_path.split(".")
    for cut in range(len(parts), 0, -1):
        head = ".".join(parts[:cut])
        loaded, exc = _imported(head)
        if exc is not None:
            if _absent(head, exc):
                continue  # not a module; try a shorter prefix
            # present but broken, or an absent driver
            return None, _why(head, exc)
        found: object = loaded
        for name in parts[cut:]:
            if not hasattr(found, name):
                return None, f"{head!r} has no {name!r}"
            found = getattr(found, name)
        return found, None
    return None, f"nothing in {module_path!r} imports"


def unresolved_path(module_path: str) -> str | None:
    """Why a bare dotted path names nothing, or ``None`` if it names something."""
    return _resolved_path(module_path)[1]


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


def loadable_targets(path: Path) -> list[tuple[int, str, str, str | None]]:
    """``(line, text, module, attribute)`` for each loadable path in a fence.

    Every fence is read, including Python ones, and the reader above is not
    duplicated by that: it collects ``import`` statements, and an ``import``
    matches no directive here. What a Python fence *can* hold is a config
    sample embedded in a string, and skipping the language to avoid a
    double-count that cannot happen would have suppressed exactly that --
    a claim nothing else reads either.

    The module/attribute split is left as loose as the text allows. An explicit
    ``module:attribute`` says where the boundary is; a bare dotted path does
    not, so the last segment is offered as the attribute and ``unresolved``
    settles it, because that function already knows a submodule and an exported
    name are both acceptable answers.
    """
    found: list[tuple[int, str, str, str | None]] = []
    for fence in code_fences(path):
        if ILLUSTRATIVE.match(fence.marker or ""):
            continue
        for offset, raw in enumerate(fence.lines):
            for match in LOADABLE.finditer(COMMENT.sub("", raw)):
                module, attribute = match.group("module"), match.group("attribute")
                if attribute is None:
                    module, _, attribute = module.rpartition(".")
                found.append((fence.line + offset, match.group(0).strip(), module, attribute))
    return found


def prose_targets(path: Path) -> list[tuple[int, str]]:
    """``(line, dotted path)`` for each path named as code in this document's prose.

    A line carrying the illustrative marker is skipped whole rather than
    per-match: a line that has to spell one wrong name usually contrasts it
    with the right one, and both are the sentence's subject.
    """
    return [
        (number, match.group("path"))
        for number, line in prose_lines(path)
        if not PROSE_ILLUSTRATIVE.search(line)
        for match in PROSE_PATH.finditer(line)
    ]


def prose_findings() -> list[str]:
    """Every path named in prose that resolves to nothing."""
    return [
        f"{rel(path)}:{number}  {named}\n      {why}"
        for path in documentation_files()
        for number, named in prose_targets(path)
        if (why := unresolved_path(named))
    ]


def prose_findings_in(path: Path) -> list[str]:
    """``prose_findings`` for a single document, for the fixtures below."""
    return [
        f"{path.name}:{number}  {named}\n      {why}"
        for number, named in prose_targets(path)
        if (why := unresolved_path(named))
    ]


def path_findings() -> list[str]:
    """Every documented path naming something to load that does not resolve."""
    return [
        f"{rel(path)}:{number}  {text}\n      {why}"
        for path in documentation_files()
        for number, text, module, attribute in loadable_targets(path)
        if (why := unresolved(module, attribute))
    ]


def path_findings_in(path: Path) -> list[str]:
    """``path_findings`` for a single document, for the fixtures below."""
    return [
        f"{path.name}:{number}  {text}\n      {why}"
        for number, text, module, attribute in loadable_targets(path)
        if (why := unresolved(module, attribute))
    ]


#: The marker a symbol carries to say it is on its way out.
#:
#: The directive, not the word. A docstring that merely *contains* "deprecated"
#: is usually describing something else, and the corpus has the case ready-made:
#: ``VersionStatus`` is a live enum whose docstring documents a member called
#: ``DEPRECATED``. Swept for the word, it adds six sites across five documents
#: that are entirely correct -- 21 sites where the directive finds 15.
#: ``.. deprecated::`` is authored deliberately and read by Sphinx; it states
#: what the prose only implies.
#: The head of an ``mkdocstrings`` block: ``::: dotted.path`` on its own line.
MKDOCSTRINGS = re.compile(r"^:::\s+(?P<path>[A-Za-z_][\w.]*)\s*$")

#: The ``members:`` key inside such a block's options, and one entry under it.
MEMBERS_KEY = re.compile(r"^\s+members:\s*$")
MEMBER_ENTRY = re.compile(r"^\s+-\s+(?P<member>\w+)\s*$")


def member_targets(path: Path) -> list[tuple[int, str, str]]:
    """``(line, documented path, member)`` for every ``mkdocstrings`` entry.

    Scoped to the options block of a ``:::`` directive and to the list under
    its ``members:`` key, rather than to the text of a list item. That
    distinction is not pedantry: ``- code`` under a ``required:`` key in a YAML
    fence is the identical token, and a reader matching the item alone reports
    it as a missing member of whatever class the page documented last.
    """
    found: list[tuple[int, str, str]] = []
    documented: str | None = None
    in_members = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        head = MKDOCSTRINGS.match(line)
        if head:
            documented, in_members = head.group("path"), False
            continue
        if documented is None:
            continue
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            # The block ends at the first line that is not indented under it.
            documented, in_members = None, False
            continue
        if MEMBERS_KEY.match(line):
            in_members = True
            continue
        entry = MEMBER_ENTRY.match(line)
        if in_members and entry:
            found.append((number, documented, entry.group("member")))
        elif in_members:
            in_members = False
    return found


def member_findings() -> list[str]:
    """Every documented member that the path it is listed under does not have."""
    broken: list[str] = []
    for path in documentation_files():
        for number, documented, member in member_targets(path):
            owner, why = _resolved_path(documented)
            if why is not None:
                # The path itself is already a finding of the prose reader; a
                # second report of the same absence, once per member, would
                # bury it rather than add to it.
                continue
            if not hasattr(owner, member):
                broken.append(f"{rel(path)}:{number}: {documented} has no member {member!r}")
    return broken


DEPRECATED = re.compile(r"\.\.\s*deprecated::")

#: Says the document knows the symbol it is naming is on its way out.
#:
#: Deliberately the bare stem in rendered prose rather than a marker of our own.
#: The audience for this one is the *reader*, so a marker would be exactly
#: wrong: invisible in the built page, and therefore satisfying the guard by
#: withholding the sentence the guard exists to require. Any spelling a human
#: would reach for -- "deprecated", "soft-deprecated", "deprecation" -- counts.
NOTICE = re.compile(r"deprecat", re.IGNORECASE)

#: Splits a document into the blocks a notice has to share with its subject.
BLOCK = re.compile(r"\n\s*\n")


def deprecated(symbol: object | None) -> bool:
    """Whether this symbol's OWN docstring says it is on its way out.

    Three narrowings, and each is the difference between a finding and a
    correct document reported as one. The last two are the same narrowing
    applied at two levels, which is why neither is safe without the other.

    **A module is never the symbol.** A module docstring carrying the marker is
    almost always deprecating a *member*: ``dataknobs_data.pooling.s3`` marks
    the single alias it re-exports, while the module itself is current and is
    named four times, correctly, by the AWS session guide. Read the module as
    deprecated and those four sentences all become findings.

    **Its own docstring, not one inherited from a base.** ``inspect.getdoc``
    walks the MRO, so the first documented subclass of a deprecated base would
    be reported for inheriting a warning about its parent -- which is the shape
    a *successor* most often has.

    **Nor one inherited from its type.** Plain ``__doc__`` lookup guards a
    class against its base and an instance against nothing: an object that
    authored no docstring answers with its class's. That reaches most of what
    the readers resolve. Forty-four documented names are neither class,
    function nor module -- string constants, pytest markers,
    ``PluginRegistry`` instances, ``Annotated`` aliases, the sentinels -- and
    each of them answers with the docstring of ``str``, ``MarkDecorator``,
    ``PluginRegistry`` or ``_AnnotatedAlias``. Mark one of those types and
    every documented instance of it becomes a finding citing a warning about
    its container.

    Both inheritance narrowings return today's set exactly, because nothing in
    the tree is a documented subclass of a deprecated base and no such type
    carries the directive -- which makes now the cheapest moment there will
    ever be to choose the right spelling, and is the only reason both are
    written before either has a case to answer.

    The cost is a blind spot, and it is stated rather than hidden: a symbol
    with nowhere to put a docstring cannot carry the directive, so a
    deprecation meant to be machine-read belongs on a class or a function.
    """
    if symbol is None or isinstance(symbol, ModuleType):
        return False
    doc = getattr(symbol, "__doc__", None)
    if doc is None or doc is getattr(type(symbol), "__doc__", None):
        return False
    return bool(DEPRECATED.search(doc))


def deprecated_symbols(path: Path) -> dict[str, list[int]]:
    """Every deprecated symbol this document names, and the lines naming it.

    All three readers feed it, because which position names the symbol is not a
    property of the symbol: ``ConfigurableBase`` arrives as an ``import``, the
    module the AWS guide discusses arrives as a path in prose, and a ``class:``
    value could name either tomorrow. Asking all three costs one predicate.

    Keyed by the *name* rather than the object, because the notice the check
    below looks for is written by a human naming the symbol, and because one
    document naming a symbol six times needs one notice, not six.
    """
    found: dict[str, list[int]] = {}

    def record(name: str, symbol: object | None, number: int) -> None:
        if deprecated(symbol):
            found.setdefault(name, []).append(number)

    for number, statement in import_statements(path):
        for module, attribute in targets(statement):
            record(attribute or module, _resolved(module, attribute)[0], number)
    for number, _text, module, attribute in loadable_targets(path):
        record(attribute or module, _resolved(module, attribute)[0], number)
    for number, dotted in prose_targets(path):
        record(dotted.rsplit(".", 1)[-1], _resolved_path(dotted)[0], number)
    return found


def deprecation_findings_in(path: Path, label: str | None = None) -> list[str]:
    """Every deprecated symbol this document names without saying it is one.

    The check is per document rather than per site, and that is the whole
    design. A page documenting a deprecated API names it constantly and is
    *correct* to -- ``configurable-base.md`` names ``ConfigurableBase`` six
    times and is the page telling you not to use it. What separates that page
    from one teaching the same class in good faith is not where the name sits
    or how often, but whether one paragraph of it says the word.

    That also gives the check reach the readers themselves do not have: a
    document is pulled in by any *one* qualified mention, and then every bare
    prose mention of the same name is covered by the notice the document now
    has to carry. The bots guide names ``BotManager`` in a heading, a diagram
    and thirty sentences none of the readers can see; it is caught by the
    single ``import`` on line 39.

    The unit of proximity is the block, not the line. A notice is a paragraph
    and a paragraph wraps, so a rule wanting the name and the word on one
    physical line would reject the natural way to write one -- and would push
    an author toward the unnatural way, or toward giving up and writing four
    separate notices for four names that share a fate.

    That width has one known cost, and it is the reason the paragraph above is
    an argument rather than an assumption. A notice names two things and warns
    about one -- "X is deprecated, use Y" -- so it clears the successor on the
    same terms as its subject. ``bot-manager.md`` names ``BotRegistry`` inside
    the notice warning about ``BotManager``, which is what a good notice looks
    like and also what would hide the day ``BotRegistry`` is deprecated in
    turn. Telling the two apart needs the block's subject, which its text does
    not carry; the limit is pinned by a test below rather than guessed at.
    """
    blocks = BLOCK.split(path.read_text(encoding="utf-8"))
    found = []
    for name, numbers in sorted(deprecated_symbols(path).items()):
        if any(name in block and NOTICE.search(block) for block in blocks):
            continue
        sites = ", ".join(str(number) for number in numbers)
        found.append(
            f"{label or path.name}  names {name} at line(s) {sites}\n"
            f"      {name} is deprecated and nothing in this document says so"
        )
    return found


def deprecation_findings() -> list[str]:
    """``deprecation_findings_in`` over every document a reader can reach."""
    return [
        finding
        for path in documentation_files()
        for finding in deprecation_findings_in(path, rel(path))
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


def test_every_loadable_path_resolves() -> None:
    """A ``class:`` value in YAML is an import, and must resolve like one."""
    broken = path_findings()
    assert not broken, (
        f"{len(broken)} documented path(s) name something to load and do not "
        "resolve, so a reader who copies the config gets an error the moment "
        "it is built:\n  " + "\n  ".join(broken) + "\n\nRepoint the path, and "
        "rename the symbol at its uses in the same fence. If it is not meant "
        "to resolve, mark the fence with <!-- dk-imports: illustrative -- why "
        "--> as a python fence would be."
    )


def test_the_loadable_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, and the reason this file needed a second floor at all.

    The floor above counts import statements, and an ``import`` matches no
    directive here -- so a ``LOADABLE`` that matched nothing would leave that
    floor at its full value, the guard above green, and this one reporting a
    clean sweep of nothing, which is the shape this file's own docstring calls
    the failure it exists to prevent. That independence is a property of the
    two readers matching disjoint syntax, and not of which fences each visits:
    this one reads every fence, python included, and one path in the tree is
    found in one.

    The number is placed above what a reader with a dead arm still reaches.
    Ten of the 46 are the ``-m`` and ``--flag`` forms, which appear only in
    shell fences; both arms going dark leaves 36, and a floor of 30 accepts
    that in silence -- the narrowing this file exists to refuse, sitting in
    its own non-vacuity check.
    """
    found = sum(len(loadable_targets(path)) for path in documentation_files())
    assert found > 40, (
        f"only {found} loadable paths found; the documents naming something to "
        "load by dotted path have not gone away, so the likelier reading is "
        "that ``LOADABLE`` has stopped matching a form of them -- one arm of "
        "the pattern going dark costs about ten, which is what this number is "
        "placed to catch"
    )


def test_every_path_named_in_prose_resolves() -> None:
    """A path a document sets in code font is a claim, and must resolve like one.

    The two readers above are both scoped to fences, and a curated API
    reference is not written in fences: it is written as a heading naming a
    fully-qualified path, followed by a block showing the class's methods. The
    heading is the claim -- "this is where this lives" -- and nothing here
    could read it.

    So one page documented an API generation that had been replaced wholesale.
    ``dataknobs_data.Database`` and five backends under it kept their pre-split
    spellings through the rename that made them ``SyncDatabase``,
    ``SyncMemoryDatabase`` and the rest, in the most authoritative kind of
    document the site has. That failure class is named in this file's own
    docstring as one of the four the first sweep found, and it recurred here
    for a reason the docstring also gives: the reader's scope was smaller than
    the corpus, and it reported green over the difference.
    """
    broken = prose_findings()
    assert not broken, (
        f"{len(broken)} path(s) named in prose resolve to nothing, so a reader "
        "sent to one finds an empty place where the document says a name "
        "lives:\n  " + "\n  ".join(broken) + "\n\nRepoint the path, and rename "
        "the symbol wherever the surrounding prose and fences use it. If the "
        "absence is the point -- a changelog's Removed entry, advice about a "
        "misspelling -- end the line with <!-- dk-imports: illustrative -- why -->."
    )


def test_the_prose_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, and this reader needs its own for the same reason the last did.

    Every path it reads sits outside a fence, where neither reader above looks,
    so a ``PROSE_PATH`` that matched nothing would leave both their floors at
    full value and report a clean sweep of an unread corpus.

    The number is placed under what the tree holds and above what a reader
    losing its harder half still reaches. 322 paths sit in 96 documents, and
    the six that were broken were all in one -- so a floor set just under 322
    would be met by a reader that had stopped visiting every file but the
    largest. Two thirds is the share the ten biggest documents hold between
    them; a floor of 250 fails if any of them stops being read.
    """
    found = sum(len(prose_targets(path)) for path in documentation_files())
    assert found > 250, (
        f"only {found} paths named in prose; the documents naming a module by "
        "dotted path have not gone away, so the likelier reading is that "
        "``PROSE_PATH`` or ``prose_lines`` has stopped reaching some of them"
    )


def test_every_documented_member_exists() -> None:
    """A member an API page lists must be one the renderer can find.

    ``mkdocstrings`` drops an entry naming a member that is not there without
    a word, so the page renders clean and short and the reader concludes the
    method does not exist. That is the quietest failure in this file: the
    other four leave something a reader can paste and watch fail, and this one
    leaves nothing at all.
    """
    broken = member_findings()
    assert not broken, (
        f"{len(broken)} documented member(s) do not exist, so the API page "
        "renders without them and a reader sent there finds the method simply "
        "missing:\n  " + "\n  ".join(broken) + "\n\nRepoint the entry at the member "
        "that replaced it, or drop the entry -- and check the surrounding prose, "
        "which is usually naming the same absent member a second time."
    )


def test_the_member_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, and this reader needs its own like every reader before it.

    Every entry it reads sits in a directive's options block, which is neither
    a fence nor prose -- so a pattern that stopped matching would leave all
    four floors above at their full value and report a clean sweep of an
    unread corpus.

    The number is placed under what the tree holds and above what a reader
    with one arm dark still reaches. 85 entries sit in 17 pages; the two
    patterns are consecutive, so ``MEMBERS_KEY`` going dark takes all 85 and
    ``MKDOCSTRINGS`` going dark takes them too. A floor of 60 fails on either,
    and on a scope that has stopped visiting the largest page.
    """
    found = sum(len(member_targets(path)) for path in documentation_files())
    assert found > 60, (
        f"only {found} documented members found; the API pages listing members "
        "have not gone away, so the likelier reading is that ``MKDOCSTRINGS`` "
        "or ``MEMBERS_KEY`` has stopped matching the form they are written in"
    )


def test_a_path_in_a_fence_is_not_read_as_prose(tmp_path: Path) -> None:
    """The two scopes are complements, and a claim belongs to exactly one.

    An import inside a fence is the first reader's, and reading it here as well
    would report the same defect twice -- and worse, would report a fence
    carrying the illustrative marker, which this reader has no way to see.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "The record type is `dataknobs_data.Record`.\n\n"
        "```python\n"
        "# `dataknobs_data.NoSuchThing` is inside a fence\n"
        "from dataknobs_data import Record\n"
        "```\n"
    )
    assert [named for _, named in prose_targets(doc)] == ["dataknobs_data.Record"]
    assert not prose_findings_in(doc)


def test_a_broken_prose_path_is_detected(tmp_path: Path) -> None:
    """The detector fires on the form the API reference was actually wrong in."""
    doc = tmp_path / "sample.md"
    doc.write_text("### `dataknobs_data.backends.memory.MemoryDatabase`\n")
    assert prose_findings_in(doc)

    doc.write_text("### `dataknobs_data.backends.memory.SyncMemoryDatabase`\n")
    assert not prose_findings_in(doc)


def test_a_method_path_is_not_split_at_its_last_dot(tmp_path: Path) -> None:
    """The false positive that decided ``unresolved_path`` walks.

    ``VectorMemory`` is a class, not a module, so the last dot is not the
    module boundary -- and read as though it were, five correct paths in the
    tree report as broken. A guard whose false positives look exactly like its
    true ones is one nobody can act on.
    """
    doc = tmp_path / "sample.md"
    doc.write_text("See `dataknobs_bots.memory.VectorMemory.add_message` for the write path.\n")
    assert not prose_findings_in(doc)

    doc.write_text("See `dataknobs_bots.memory.VectorMemory.no_such_method` for the write path.\n")
    assert prose_findings_in(doc)


def test_a_line_can_declare_its_path_illustrative(tmp_path: Path) -> None:
    """A document that names an absent thing on purpose says so, and is believed.

    Both real uses are lines whose subject is the absence: a changelog entry
    recording a removal, and advice contrasting a wrong spelling with a right
    one. The marker covers the whole line for the second of those -- a sentence
    warning about ``dataknobs.package`` names ``dataknobs_package`` in the same
    breath, and splitting the line would leave the warning half-checked.
    """
    doc = tmp_path / "sample.md"
    line = "Use `dataknobs_data` not `dataknobs.data`"
    doc.write_text(line + "\n")
    assert prose_findings_in(doc)

    doc.write_text(
        line + " <!-- dk-imports: illustrative -- the wrong spelling is the subject -->\n"
    )
    assert not prose_targets(doc)


def test_a_clone_url_is_not_read_as_a_path(tmp_path: Path) -> None:
    """The nine-site false positive that decided the design.

    ``dataknobs.git`` is this repository, and it parses as a module with an
    attribute perfectly well -- ``unresolved`` says ``'dataknobs' exports no
    'git'``, which is true and completely beside the point. Nothing excludes
    it by name; it is excluded because ``git clone`` is not a position that
    loads anything, and this test fails if that stops being what decides.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```bash\ngit clone https://github.com/your-org/dataknobs.git\ncd dataknobs\n```\n"
    )
    assert not loadable_targets(doc)


def test_a_commented_path_is_not_read_as_a_path(tmp_path: Path) -> None:
    """A comment naming a module is prose, and prose is checked elsewhere.

    Both halves matter: the comment on its own line is not a claim, and the
    one trailing a real claim must not swallow it.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```yaml\n"
        "# class: dataknobs_data.backends.memory.NoSuchDatabase\n"
        "databases:\n"
        "  - class: dataknobs_data.backends.memory.SyncMemoryDatabase  # the cache\n"
        "```\n"
    )
    assert [module for _, _, module, _ in loadable_targets(doc)] == [
        "dataknobs_data.backends.memory"
    ]
    assert not path_findings_in(doc)


def test_a_broken_class_value_is_detected(tmp_path: Path) -> None:
    """The detector fires on the form the corpus was actually wrong in."""
    doc = tmp_path / "sample.md"
    doc.write_text("```yaml\ndatabases:\n  - class: dataknobs_data.backends.s3.S3Database\n```\n")
    assert path_findings_in(doc)

    doc.write_text(
        "<!-- dk-imports: illustrative -- the pre-split spelling is the subject -->\n"
        "```yaml\ndatabases:\n  - class: dataknobs_data.backends.s3.S3Database\n```\n"
    )
    assert not loadable_targets(doc)


def test_an_explicit_split_is_honoured_over_the_last_dot(tmp_path: Path) -> None:
    """``module:attribute`` says where the boundary is; a bare path does not.

    Read by the last dot instead, ``task_injection:create_review_task_hook``
    would ask ``dataknobs_bots.reasoning`` for a ``task_injection`` and get an
    answer -- the right one, for the wrong reason, which stops being harmless
    the moment the attribute after the colon is the part that is wrong.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```yaml\nhooks:\n"
        '  - function: "dataknobs_bots.reasoning.task_injection:no_such_hook"\n```\n'
    )
    assert [(m, a) for _, _, m, a in loadable_targets(doc)] == [
        ("dataknobs_bots.reasoning.task_injection", "no_such_hook")
    ]
    assert path_findings_in(doc)


def test_a_config_embedded_in_python_is_still_read(tmp_path: Path) -> None:
    """The case that decided against skipping Python fences here.

    A document that shows its YAML as a Python string is showing the same
    claim, and neither reader would have seen it: the one above collects
    ``import`` statements, and this one would have skipped the fence for its
    language. An ``import`` matches no directive below, so reading every
    fence double-counts nothing.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        'CONFIG = """\n'
        "databases:\n"
        "  - class: dataknobs_data.backends.s3.NoSuchDatabase\n"
        '"""\n```\n'
    )
    assert path_findings_in(doc)

    doc.write_text("```python\nfrom dataknobs_data.backends.s3 import SyncS3Database\n```\n")
    assert not loadable_targets(doc), "an import is the other reader's claim"


def test_a_quoted_key_is_read_as_a_directive(tmp_path: Path) -> None:
    """A config shown as a literal quotes its keys, and names the same thing.

    The directive is the same word in the same position; only the punctuation
    around it differs, because a sample rendered as a Python or JSON literal
    puts the key in quotes. Reading one spelling and not the other makes the
    reader's scope a property of how a document happens to render its config
    -- the narrowing this file's docstring names, arriving through the pattern
    this time instead of through the fence language.

    Both boundaries are load-bearing and both were wrong. The opening quote is
    not the whitespace the pattern demanded ahead of the directive, and the
    closing quote sits between the directive and its colon -- so widening only
    the first leaves the form matching exactly as little as before.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "CONFIG = {\n"
        '    "resources": [\n'
        '        {"class": "dataknobs_data.backends.s3:NoSuchDatabase"},\n'
        "    ]\n"
        "}\n```\n"
    )
    assert [(m, a) for _, _, m, a in loadable_targets(doc)] == [
        ("dataknobs_data.backends.s3", "NoSuchDatabase")
    ]
    assert path_findings_in(doc)


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


def test_no_document_teaches_a_deprecated_symbol_in_silence() -> None:
    """A name that resolves can still be the wrong one to hand a reader.

    Everything above asks whether a documented name is *there*. A deprecated
    one is -- that is what makes it invisible to all three readers, and what
    makes the silence dangerous rather than merely untidy. ``ConfigurableBase``
    is soft-deprecated in as many words in its own docstring, and the docstring
    says why the transition raises no runtime warning: so it stays quiet across
    a multi-cycle migration. That is a defensible choice for the consumers who
    already inherit it, and it has one consequence nobody chose -- documentation
    becomes the ONLY channel through which a new consumer could learn the base
    is going away, and the documentation was the channel recommending it. This
    check found eleven such silences in five documents on its first run.

    The bots family is the same shape and worse. ``BotManager`` and the three
    singleton helpers around it *do* warn at runtime, so two guides were
    teaching an API that greets the reader with a ``DeprecationWarning`` the
    moment their first paste runs.
    """
    silent = deprecation_findings()
    assert not silent, (
        f"{len(silent)} document(s) teach a deprecated symbol without saying "
        "it is deprecated, so a reader who follows the sample adopts a name "
        "that is scheduled for removal:\n  " + "\n  ".join(silent) + "\n\nEither "
        "rewrite the sample against the successor the symbol's docstring "
        "names, or -- if the document is about the deprecated symbol -- say so "
        "in a line that names it, which is what the reader needed anyway."
    )


def test_the_deprecation_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, and here it guards the one thing a green result depends on.

    This check reports green in two situations that look identical from the
    outside: every document that names a deprecated symbol carries its notice,
    and no document names one at all. The second is what a broken ``DEPRECATED``
    pattern produces, or a ``deprecated`` predicate narrowed by one clause too
    many, and it is indistinguishable from success without this.

    The floor counts documents *reached*, not findings, so it holds steady
    across the repair this check asks for -- a document that gains its notice
    still names the symbol and is still counted.

    Six documents are reached, and the number is placed to fail if either
    reader feeding this stops working. The two are not interchangeable and
    neither dominates: three documents name their symbol only in prose (the
    notices, which spell ``dataknobs_config.ConfigurableBase`` in running
    text), two name one only inside a fence, and one does both. So losing the
    import reader leaves four and losing the prose reader leaves three -- and
    five is the floor that fails on either, where four would have sat quietly
    through the first.

    **One repair does move it, and it is the other one this check offers.**
    The failure message beside it names two remedies -- carry a notice, or
    rewrite the sample against the successor -- and the second removes the
    document from this count, correctly. Only ``configurable-base.md`` is
    unmovable, being the page *about* its symbol; the two bots guides are
    reached through a surface already scheduled for a rewrite against the
    registry, and taking that pass drops this to four. That is a floor to
    re-derive, not a guard to doubt, which is why the message below leads with
    the reading it cannot distinguish from a broken reader.
    """
    reached = [rel(path) for path in documentation_files() if deprecated_symbols(path)]
    assert len(reached) >= 5, (
        f"only {len(reached)} document(s) name a deprecated symbol at all "
        f"({', '.join(reached)}). Two readings, and they need telling apart "
        "before either is acted on. If a document was rewritten against a "
        "successor -- the second remedy this check offers -- the corpus has "
        "legitimately shrunk and this number wants re-deriving against the "
        "documents that remain. If none was, the deprecated symbols have not "
        "gone away, so ``deprecated`` or one of the three readers feeding it "
        "has stopped recognising them, and this guard is reporting green over "
        "a corpus it never read"
    )


def test_a_notice_naming_the_symbol_is_what_clears_a_document(tmp_path: Path) -> None:
    """The pass condition, and it is the sentence the reader needed anyway.

    Both halves are load-bearing. A document that says "deprecated" about
    something else has not warned anyone about this symbol, and a document that
    names the symbol without the word has not warned anyone at all.
    """
    doc = tmp_path / "sample.md"
    sample = "```python\nfrom dataknobs_config import ConfigurableBase\n```\n"

    doc.write_text(sample)
    assert deprecation_findings_in(doc)

    doc.write_text("Some other API is deprecated.\n\n" + sample)
    assert deprecation_findings_in(doc), "a notice must name the symbol it is about"

    doc.write_text("`ConfigurableBase` is the old base.\n\n" + sample)
    assert deprecation_findings_in(doc), "naming the symbol is not warning about it"

    doc.write_text("`ConfigurableBase` is deprecated; use the successor.\n\n" + sample)
    assert not deprecation_findings_in(doc)

    doc.write_text(
        "> `ConfigurableBase` and the rest of that generation\n"
        "> are deprecated; use the successor.\n\n" + sample
    )
    assert not deprecation_findings_in(doc), "a notice is a paragraph, and it wraps"


def test_a_notice_also_clears_the_successor_it_names(tmp_path: Path) -> None:
    """The known limit of block proximity, pinned so it stays deliberate.

    A notice names two things and warns about one: "X is deprecated, use Y".
    The block holds both names and the word, so it clears Y on the same terms
    as X -- and Y is not some contrived case, it is the successor, which is
    precisely the name most likely to be deprecated next.

    The tree already has the shape, written by the repairs this check
    prompted. The notice in ``bot-manager.md`` warns about ``BotManager`` and
    the three singleton helpers, and names ``BotRegistry`` and
    ``InMemoryBotRegistry`` as what to use instead. Deprecate either successor
    and that document goes on passing in silence.

    Separating the two needs the block's *subject*, which the text does not
    carry -- and a heuristic that guesses it from "use" or "instead" would be
    the contents-sniffing this file rejects everywhere else. The alternative
    is to read the successor out of the deprecating symbol's own directive,
    which is a second prose parser and wants its own justification.

    So it is recorded instead of guessed at, and this test is the record: it
    asserts the loose behaviour, so a later tightening announces itself here
    rather than somewhere a reader has to go looking.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "> **Deprecated.** `LegacyThing` is deprecated.\n"
        "> Use `ConfigurableBase` instead.\n\n"
        "```python\nfrom dataknobs_config import ConfigurableBase\n```\n"
    )
    assert not deprecation_findings_in(doc), (
        "the limit has been tightened -- update this test and the docstring "
        "of deprecation_findings_in, which records it as standing"
    )


def test_a_module_marker_is_not_read_as_a_marker_on_the_module(tmp_path: Path) -> None:
    """The false positive that decided ``deprecated`` skips modules.

    ``dataknobs_data.pooling.s3`` carries ``.. deprecated::`` in its module
    docstring, about one alias it re-exports. The module is current, holds the
    genuinely S3-specific surface, and is named four times by the AWS session
    guide in sentences that are all correct -- and read as deprecated it turns
    that guide into four findings requiring a notice that would be false.
    """
    doc = tmp_path / "sample.md"
    doc.write_text("The pool config lives in `dataknobs_data.pooling.s3`.\n")
    assert not deprecation_findings_in(doc)


def test_an_inherited_marker_is_not_read_as_the_subclass_own() -> None:
    """``inspect.getdoc`` walks the MRO; ``__doc__`` is the symbol's own word.

    A subclass of a deprecated base is not thereby deprecated -- it is the most
    likely shape of a *successor* -- so resolving the docstring through the MRO
    would report the replacement for carrying its predecessor's warning.
    """
    # Imported here rather than at module scope: every other name this file
    # touches is resolved dynamically with its ImportError caught, and one
    # package promoted to a collection-time dependency would take the whole
    # guard down with it.
    from dataknobs_config import ConfigurableBase

    class Successor(ConfigurableBase):
        pass

    assert deprecated(ConfigurableBase)
    assert not deprecated(Successor)
    assert inspect.getdoc(Successor) == inspect.getdoc(ConfigurableBase)


def test_a_type_marker_is_not_read_as_a_marker_on_its_instances() -> None:
    """The same leak as the one above, one level further down.

    ``inspect.getdoc`` is rejected there for walking the MRO. Plain attribute
    lookup does the identical thing between an object and its class: an
    instance that authored no docstring answers with its type's, and
    ``__doc__`` offers no protection against it -- the protection it does
    offer is between a class and its base.

    The corpus is full of the shape and none of it is a class. Forty-four
    documented names resolve to something that is neither class, function nor
    module -- sixteen string constants, eight pytest markers, six
    ``PluginRegistry`` instances, two ``Annotated`` aliases, the sentinels --
    and every one answers with the docstring of ``str``, ``MarkDecorator``,
    ``PluginRegistry`` or ``_AnnotatedAlias``. Mark any one of those types
    tomorrow and six correct registry mentions become findings citing a
    warning about their container.

    No such type carries the directive today, so both spellings return the
    same set on the tree and choosing costs nothing -- which is the same
    reason, and the same moment, as the narrowing above.
    """

    class Marked:
        """A container.

        .. deprecated:: 1.0
           Use the successor.
        """

    assert deprecated(Marked)
    assert not deprecated(Marked()), "an instance did not author its type's warning"


def test_a_symbol_that_authors_no_docstring_cannot_be_read_as_deprecated() -> None:
    """The limit the narrowing above leaves behind, stated rather than found.

    A constant, a sentinel or a ``typing`` alias has nowhere to put a
    docstring, so it cannot carry the directive and this check cannot see it.
    ``BotManagerDep`` is the live instance: it is the deprecated half of a
    pair, says so in a ``#`` comment no runtime reads, and is indistinguishable
    here from ``BotRegistryDep``, the successor it is paired against.

    That is a property of Python rather than of this predicate -- reading the
    comment means reading source, which is a different guard. What follows
    from it is a rule for authors: a deprecation that has to be machine-read
    belongs on a class or a function, which is where every one in the tree
    is today.
    """
    from dataknobs_bots.api import BotManagerDep, BotRegistryDep

    assert not deprecated(BotManagerDep)
    assert not deprecated(BotRegistryDep)
    assert BotManagerDep.__doc__ == BotRegistryDep.__doc__


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


# --- The name beneath the import -------------------------------------------
#
# Every reader above asks whether a documented name exists at the path shown.
# None of them reads the line *under* the import. This file's docstring used to
# name that boundary and hand it to the reviewer: "repointing an import while
# leaving the body calling the old name produces a sample that looks corrected
# and fails on its second line, which is worse than one that fails on its
# first."
#
# The reviewer did not catch it. When this was written, 44 attribute accesses
# in the corpus named something their own module does not have -- 15 on one
# page of the utils API reference, ten spelling ``dk_doc.Document`` for a class
# called ``Text``, six calling a ``normalize_whitespace_fn`` that is a flag on
# another function rather than a function. Every one of them sat under an
# import that resolves, so every one reported green through all four readers.
#
# **The scope is an attribute on a name the fence itself imported**, which is
# the only receiver whose type a reader can know without executing anything.
# ``response.content`` on an instance is out of reach and stays out: the
# fence's own ``LLMResponse`` is a dataclass whose fields are invisible to
# ``hasattr`` on the class, so a reader that guessed at instances would report
# a correct sample as broken -- the false positive this file's prose calls
# "indistinguishable from a true one, which is the shape that gets a guard
# suppressed wholesale".
#
# For the same reason a CLASS receiver is asked more than ``hasattr``: a
# dataclass field, a bare annotation and a ``__slots__`` entry are all members
# that ``hasattr`` on the class denies, and a class defining ``__getattr__``
# answers for names nobody declared. Each is checked before a finding is
# raised. A name the fence REBINDS is dropped, because after ``json_utils = ...``
# the import no longer says what the name holds.


def _bound(tree: ast.Module) -> tuple[dict[str, object], set[str]]:
    """``(name -> live object, modules that would not load)`` for this fence.

    The second half is the point of the tuple. A module that fails to import
    binds no names, so every attribute access and every call beneath it leaves
    both readers below with nothing to say -- silently, and in exactly the case
    the *import* reader is designed to stay green on: a missing optional
    driver is a property of the environment, not a defect in the document, so
    ``_why`` returns nothing for it and no finding is raised.

    That tolerance is right for the question the import reader asks and wrong
    as a coverage story, because the fence then goes unread by the two readers
    that would have checked the lines under the import. Returning the failures
    rather than swallowing them lets ``test_the_readers_are_not_quietly_losing_fences``
    put a number and a name on whatever was skipped, which is the same bargain
    ``call_sites`` already strikes with its unanalysable count.
    """
    env: dict[str, object] = {}
    unloadable: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not (node.module and node.module.startswith(NAMESPACE)):
                continue
            loaded, exc = _imported(node.module)
            if exc is not None:
                unloadable.add(node.module)
                continue
            for alias in node.names:
                if alias.name != "*" and hasattr(loaded, alias.name):
                    env[alias.asname or alias.name] = getattr(loaded, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(NAMESPACE):
                    continue
                # `import a.b.c` binds `a`, but the submodule has to be
                # imported for `a.b` to resolve as an attribute of it. Import
                # the dotted name either way and bind whichever name the
                # statement actually introduces -- otherwise `a.b` resolves
                # only when some earlier test in the same process happened to
                # import it, which makes a finding depend on collection order.
                loaded, exc = _imported(alias.name)
                if exc is not None:
                    unloadable.add(alias.name)
                    continue
                if alias.asname:
                    env[alias.asname] = loaded
                else:
                    top = alias.name.split(".")[0]
                    package, package_exc = _imported(top)
                    if package_exc is None:
                        env[top] = package
                    else:
                        unloadable.add(top)
    return env, unloadable


def _rebound(tree: ast.Module) -> set[str]:
    """Every name the fence binds itself, whatever an import said about it.

    Deliberately over-broad, and flat: a name bound anywhere in the fence is
    dropped everywhere in it, including by a function parameter, a subscript
    target (``d[Config] = 2``) or an attribute-assignment receiver. This is not
    scoping and should not be mistaken for it -- it trades false negatives for
    the certainty of no false positives, which is the trade a guard people can
    switch off has to make. Measured across the corpus, the cost is currently
    zero accesses and zero calls lost.
    """
    shadowed: set[str] = set()

    def bind(target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                shadowed.add(node.id)

    def arguments(args: ast.arguments) -> None:
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            shadowed.add(argument.arg)
        for optional in (args.vararg, args.kwarg):
            if optional is not None:
                shadowed.add(optional.arg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target)
        elif isinstance(
            node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.comprehension)
        ):
            bind(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shadowed.add(node.name)
            arguments(node.args)
        elif isinstance(node, ast.ClassDef):
            shadowed.add(node.name)
        elif isinstance(node, ast.Lambda):
            arguments(node.args)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            shadowed.add(node.name)
    return shadowed


def _slots(base: type) -> tuple[str, ...]:
    """``__slots__`` as a tuple of names, however the class spelled it.

    A single slot may be declared as a bare string, and ``in`` against a string
    is a substring test: ``"alp" in "alpha"`` is true, so a class with
    ``__slots__ = "alpha"`` would answer for an attribute it does not have.
    """
    declared = getattr(base, "__slots__", ())
    return (declared,) if isinstance(declared, str) else tuple(declared)


def declares(obj: object, attribute: str) -> bool:
    """Whether ``obj`` has ``attribute``, including the ways ``hasattr`` misses."""
    if hasattr(obj, attribute):
        return True
    if not isinstance(obj, type):
        return False
    for base in obj.__mro__:
        if attribute in getattr(base, "__annotations__", {}):
            return True
        if attribute in _slots(base):
            return True
        if dataclasses.is_dataclass(base) and any(
            field.name == attribute for field in dataclasses.fields(base)
        ):
            return True
    # A class answering for undeclared names cannot be asked this question --
    # and for a CLASS receiver, which is all this reader ever holds, the name
    # that answers is the metaclass's. A ``__getattr__`` defined on the class
    # itself serves its *instances*; consulting it here was asking the object
    # below the receiver about the receiver, which suppressed real findings on
    # every class that defines one.
    metaclass: type = type(obj)
    return any("__getattr__" in vars(base) for base in metaclass.__mro__[:-1])


@cache
def attribute_sites(path: Path) -> list[tuple[int, str, object, str]]:
    """``(line, receiver, object, attribute)`` for each resolvable access in ``path``.

    Cached: a pure reading of one file, asked for by both the findings list and
    the corpus floor, and the corpus is read four times over across this
    file's whole-tree tests.
    """
    found: list[tuple[int, str, object, str]] = []
    for fence in code_fences(path):
        if fence.lang not in PYTHON_FENCE or ILLUSTRATIVE.match(fence.marker or ""):
            continue
        tree = parsed(fence.body)
        if tree is None:
            continue
        env, _ = _bound(tree)
        if not env:
            continue
        for name in _rebound(tree):
            env.pop(name, None)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
                continue
            obj = env.get(node.value.id)
            if obj is not None:
                found.append((fence.line + node.lineno - 1, node.value.id, obj, node.attr))
    return found


def attribute_findings() -> list[str]:
    """Every documented attribute access whose receiver does not have it."""
    return [
        f"{rel(path)}:{line}: {receiver}.{attribute}"
        for path in documentation_files()
        for line, receiver, obj, attribute in attribute_sites(path)
        if not declares(obj, attribute)
    ]


def test_every_documented_attribute_resolves() -> None:
    """The line under the import is a claim too, and must hold."""
    broken = attribute_findings()
    assert not broken, (
        f"{len(broken)} documented attribute access(es) name something their "
        "own module or class does not have, so the sample fails below its "
        "import rather than on it:\n  "
        + "\n  ".join(broken)
        + "\n\nRepoint the call at the name that exists. If the access is not "
        "meant to resolve, mark the fence with "
        "<!-- dk-imports: illustrative -- why --> as an import would be."
    )


def test_the_attribute_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, and this reader needs its own floor more than most.

    It is the only one whose corpus is filtered twice -- a fence must import
    from the namespace *and* then use what it imported by attribute -- so it is
    the one most able to go quiet without any of the counts above moving. Both
    the import floor and the loadable floor would sit at their full values
    while this returned an empty list, which is a clean sweep of nothing.

    The number is placed below what the tree holds (983 across 383 documents
    when this was last measured) and well above what a single page contributes,
    so losing one document is survivable and losing an arm of ``_bound`` is
    not. Re-measure rather than trusting the figure: it is a reading of the
    tree on a given day, and the floor below it is what is load-bearing.
    """
    found = sum(len(attribute_sites(path)) for path in documentation_files())
    assert found > 700, (
        f"only {found} resolvable attribute accesses found; the documents using "
        "an imported module by attribute have not gone away, so the likelier "
        "reading is that ``_bound`` has stopped binding one of the import forms"
    )


def unreadable_fences() -> list[str]:
    """``file:line -- module`` for each fence a reader could not bind names in."""
    found: list[str] = []
    for path in documentation_files():
        for fence in code_fences(path):
            if fence.lang not in PYTHON_FENCE or ILLUSTRATIVE.match(fence.marker or ""):
                continue
            tree = parsed(fence.body)
            if tree is None:
                continue
            _, unloadable = _bound(tree)
            for module in sorted(unloadable):
                exc = _imported(module)[1]
                reason = _why(module, exc) if exc is not None else None
                found.append(
                    f"{rel(path)}:{fence.line}: {module} -- {reason or 'optional driver absent'}"
                )
    return found


def test_the_readers_are_not_quietly_losing_fences() -> None:
    """A fence whose module will not load is read by neither reader below.

    ``_bound`` binds no names for such a fence, so every attribute access and
    every call in it is skipped -- and skipped in the one case the import
    reader deliberately stays green on, a missing optional third-party driver.
    Nothing else in this file would notice: the floors are absolute numbers and
    a handful of fences does not move them, so the loss is invisible at exactly
    the size it is most likely to occur.

    This does not forbid the skip -- the drivers really are optional, and a
    lean environment is a legitimate place to run the suite. It forbids the
    skip being silent, and bounds it: the message names every module that would
    not load, so a reader can see which pages went unchecked rather than
    inferring it from a count that did not move.
    """
    lost = unreadable_fences()
    modules = {finding.split(": ", 1)[1].split(" -- ")[0] for finding in lost}
    assert len(modules) < 10, (
        f"{len(lost)} fence(s) across {len(modules)} module(s) bind no names, so "
        "neither the attribute reader nor the binding reader read them:\n  "
        + "\n  ".join(lost)
        + "\n\nInstall the drivers these modules need, or accept the gap "
        "knowing which documents it covers."
    )


def test_a_fence_whose_module_will_not_load_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity: the skip must be visible when there is one to see."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\nfrom dataknobs_utils import json_utils\njson_utils.get_value({}, 'a')\n```\n"
    )
    real = _imported

    def refuse(module: str) -> tuple[ModuleType | None, BaseException | None]:
        if module == "dataknobs_utils":
            return None, ModuleNotFoundError("No module named 'yaml'", name="yaml")
        return real(module)

    monkeypatch.setitem(globals(), "_imported", refuse)
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    # The import reader tolerates this by design -- the missing name is a
    # driver, not the module -- which is precisely why the readers below it
    # must not also fall silent without saying so.
    refused = refuse("dataknobs_utils")[1]
    assert refused is not None, "the stub must report a failure to be a stub"
    assert _why("dataknobs_utils", refused) is None
    assert attribute_sites(doc) == [], "no names bind, so nothing is read"

    lost = unreadable_fences()
    assert len(lost) == 1, f"the skipped fence must be reported, got {lost}"
    assert "dataknobs_utils" in lost[0]


def test_a_slotted_class_does_not_answer_for_a_substring_of_a_slot() -> None:
    """``__slots__`` may be a bare string, and ``in`` on a string is a substring."""

    class Slotted:
        # A bare string is the subject: `in` against one is a substring test,
        # so this is the shape `_slots` exists to normalize. PLC0205 is right
        # about production code and wrong about the fixture that proves it.
        __slots__ = "alpha"  # noqa: PLC0205

    assert declares(Slotted, "alpha"), "the declared slot is a member"
    assert not declares(Slotted, "alp"), "a substring of it is not"


def test_a_class_defining_getattr_for_its_instances_is_still_asked() -> None:
    """``__getattr__`` on the class answers for instances, not for the class.

    The reader only ever holds a class or a module as receiver, so the name
    that could answer for an undeclared attribute is the *metaclass*'s. Reading
    the class's own MRO instead let every class defining ``__getattr__`` -- a
    common shape -- suppress real findings about the class object itself.
    """

    class ForInstances:
        def __getattr__(self, name: str) -> object:  # pragma: no cover - never called
            raise AttributeError(name)

    with pytest.raises(AttributeError):
        ForInstances.absent  # type: ignore[attr-defined]  # noqa: B018 - the access IS the assertion
    assert not declares(ForInstances, "absent")

    class Answering(type):
        def __getattr__(cls, name: str) -> object:
            return object()

    class ViaMetaclass(metaclass=Answering):
        pass

    assert declares(ViaMetaclass, "anything_at_all")


def test_a_broken_attribute_is_detected(tmp_path: Path) -> None:
    """The detector fires on the shape of the defect this was written for."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_utils import json_utils\n"
        "data = json_utils.load_json_file('x.json')\n"
        "```\n"
    )
    found = attribute_sites(doc)
    assert [(name, attr) for _, name, _, attr in found] == [("json_utils", "load_json_file")]
    assert not declares(found[0][2], "load_json_file")


def test_a_working_attribute_is_not_flagged(tmp_path: Path) -> None:
    """And does not fire on the corrected form."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_utils import json_utils\n"
        "value = json_utils.get_value(data, 'a.b')\n"
        "```\n"
    )
    assert [
        (name, attr) for _, name, obj, attr in attribute_sites(doc) if not declares(obj, attr)
    ] == []


def test_a_rebound_name_is_not_read_as_the_import(tmp_path: Path) -> None:
    """After the fence assigns the name, the import no longer says what it holds.

    Without this the reader reports every attribute of a local object that
    happens to share a name with an imported module -- a false positive
    indistinguishable from a true one, which is what gets a guard switched off.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_utils import json_utils\n"
        "json_utils = MyOwnWrapper()\n"
        "json_utils.anything_at_all()\n"
        "```\n"
    )
    assert attribute_sites(doc) == [], "a rebound name must not be read as the import"


def test_a_dataclass_field_is_not_read_as_absent() -> None:
    """``hasattr`` on the class denies a dataclass field, and the class is right.

    ``LLMResponse.content`` is the live case: a field every consumer uses, which
    a reader asking ``hasattr`` alone reports as fiction.
    """
    from dataknobs_llm.llm.base import LLMResponse

    assert not hasattr(LLMResponse, "content"), "the premise of this test has moved"
    assert declares(LLMResponse, "content"), "a dataclass field is a member"
    assert not declares(LLMResponse, "not_a_field_at_all")


def test_an_illustrative_fence_is_not_read_for_attributes(tmp_path: Path) -> None:
    """The marker covers this reader too, or it covers half a fence."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "<!-- dk-imports: illustrative -- the old spelling is the subject -->\n"
        "```python\n"
        "from dataknobs_utils import json_utils\n"
        "json_utils.load_json_file('x.json')\n"
        "```\n"
    )
    assert attribute_sites(doc) == []


def test_attribute_findings_report_one_when_the_tree_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the finding path, the way the other checks get one.

    ``test_every_documented_attribute_resolves`` runs over a tree where nothing
    is broken, so it passes whether the body still detects anything or not.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_utils import json_utils\n"
        "json_utils.load_json_file('x.json')\n"
        "json_utils.get_value({}, 'a')\n"
        "```\n"
    )
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    found = attribute_findings()

    assert len(found) == 1, f"expected the one broken access, got {found}"
    assert "sample.md:3" in found[0], f"wrong line reported: {found[0]}"


# --- The arguments beneath the name ----------------------------------------
#
# The reader above asks whether a documented attribute exists. It cannot ask
# whether a documented CALL could bind: ``TextMetaData()`` names a class that
# is really there, and fails on the line that constructs it because the class
# requires an argument the sample never passes. That miss is not theoretical.
# One of those sat on a page this file's own attribute scan had just cleared,
# in a commit that reported the page fixed.
#
# When this was written, 91 documented calls could not bind, against the 3,290
# then readable. As with the class above, the population was a handful of shapes
# rather than ninety-one mistakes:
#
# - **A required argument the docs treat as optional.** ``LLMConfig`` needs a
#   ``model``; twelve samples passed only a provider. Every provider takes one
#   config object and four pages passed it the config's fields instead.
# - **A keyword that was never there.** ``Config.from_file(apply_env_overrides=True)``
#   across six pages, for behaviour that is unconditional -- a whole page was
#   built on a method that does not exist.
# - **An abstract class instantiated.** ``Tool(name=..., func=...)`` in six
#   places; ``Tool`` is an ABC and the function-wrapping constructor is
#   imagined. Here the finding is the *keyword*, and the abstractness is what
#   the reader finds when they follow it.
# - **A parameter renamed and not followed.** ``file_extension`` for
#   ``file_extensions``, ``polling_interval`` for ``poll_interval``,
#   ``metadata_fields`` for ``metadata_field``.
#
# **The scope is the same receiver the attribute reader uses** -- a name the
# fence imported, or an attribute of one -- for the same reason: it is the only
# callable whose identity is knowable without executing the fence. A call on an
# instance (``registry.register(tool)``) is out of reach and stays out.
#
# **Binding is not calling.** ``signature().bind()`` answers whether the
# arguments could be *accepted*, never whether they are the right values, so a
# sample passing ``model=3`` binds cleanly. That is deliberate: the check has
# no false positives to trade away, which is what lets it run over the whole
# corpus. Calls carrying ``*args``/``**kwargs``, and callables whose signature
# cannot be read at all, are counted separately and skipped rather than
# guessed at.


def _call_target(node: ast.Call, env: dict[str, object]) -> Callable[..., object] | None:
    """The live callable a ``Call`` node names, or ``None`` if unknowable."""
    func = node.func
    found: object = None
    if isinstance(func, ast.Name):
        found = env.get(func.id)
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        owner = env.get(func.value.id)
        found = None if owner is None else getattr(owner, func.attr, None)
    return found if callable(found) else None


def _unpacks(node: ast.Call) -> bool:
    """Whether the call spreads a sequence or mapping, hiding its real arity."""
    return any(isinstance(arg, ast.Starred) for arg in node.args) or any(
        keyword.arg is None for keyword in node.keywords
    )


def _arity(node: ast.Call) -> tuple[list[object], dict[str, object]]:
    """Placeholder arguments matching the call's shape, standing in for values.

    ``_unpacks`` has already rejected the unnamed-keyword form, so every
    ``keyword.arg`` reaching here is a real name; the filter restates that for
    the type checker rather than asserting it.
    """
    return (
        [object()] * len(node.args),
        {kw.arg: object() for kw in node.keywords if kw.arg is not None},
    )


@cache
def call_sites(
    path: Path,
) -> tuple[list[tuple[int, str, Callable[..., object], ast.Call]], int]:
    """``(sites, unanalysable)`` for the calls in ``path`` on imported names.

    Cached for the same reason as ``attribute_sites``: three whole-corpus tests
    ask for it, and one of them asks twice in a single assertion.
    """
    sites: list[tuple[int, str, Callable[..., object], ast.Call]] = []
    unanalysable = 0
    for fence in code_fences(path):
        if fence.lang not in PYTHON_FENCE or ILLUSTRATIVE.match(fence.marker or ""):
            continue
        tree = parsed(fence.body)
        if tree is None:
            continue
        env, _ = _bound(tree)
        if not env:
            continue
        for name in _rebound(tree):
            env.pop(name, None)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _call_target(node, env)
            if target is None:
                continue
            if _unpacks(node):
                unanalysable += 1
                continue
            try:
                inspect.signature(target)
            except (ValueError, TypeError):
                # A C-level or otherwise unreadable signature. Counted, not guessed.
                unanalysable += 1
                continue
            sites.append((fence.line + node.lineno - 1, ast.unparse(node.func), target, node))
    return sites, unanalysable


def binding_findings() -> list[str]:
    """Every documented call whose arguments cannot bind to the real signature."""
    broken: list[str] = []
    for path in documentation_files():
        sites, _ = call_sites(path)
        for line, shown, target, node in sites:
            positional, keywords = _arity(node)
            try:
                inspect.signature(target).bind(*positional, **keywords)
            except TypeError as exc:
                broken.append(f"{rel(path)}:{line}: {shown}(...) -- {exc}")
    return broken


def test_every_documented_call_can_bind() -> None:
    """A documented call must be one the reader could actually make."""
    broken = binding_findings()
    assert not broken, (
        f"{len(broken)} documented call(s) cannot bind to the signature of the "
        "name they invoke, so the sample fails on the line that calls rather "
        "than the line that imports:\n  "
        + "\n  ".join(broken)
        + "\n\nRepoint the call at the real signature. If the call is not meant "
        "to resolve, mark the fence with "
        "<!-- dk-imports: illustrative -- why --> as an import would be."
    )


def test_the_binding_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity, with the same hazard as the attribute floor above.

    This reader filters the corpus three times -- a fence must import from the
    namespace, call what it imported, and expose a readable signature -- so it
    can fall silent while every count above holds. The floor is placed below
    what the tree holds (3,421 when this was last measured) and far above any
    one page, so losing a document is survivable and losing an arm of
    ``_bound`` or ``_call_target`` is not.
    """
    bindable = sum(len(call_sites(path)[0]) for path in documentation_files())
    assert bindable > 2500, (
        f"only {bindable} analysable calls found; the documents calling an "
        "imported name have not gone away, so the likelier reading is that "
        "``_call_target`` has stopped resolving one of the receiver forms"
    )


def test_the_unanalysable_calls_stay_a_small_minority() -> None:
    """The skip path is an escape hatch, and an escape hatch can swallow a corpus.

    ``*args``/``**kwargs`` and unreadable signatures are skipped rather than
    guessed at, which is correct and also the one way this check could report
    green over everything. Fourteen were skipped when this was last measured,
    against 3,421 read. The bound is loose because the number is small; what it
    forbids is the skip path quietly becoming the common path.
    """
    unanalysable = sum(call_sites(path)[1] for path in documentation_files())
    bindable = sum(len(call_sites(path)[0]) for path in documentation_files())
    assert unanalysable < bindable // 10, (
        f"{unanalysable} calls were skipped as unanalysable against {bindable} "
        "read, which is too large a share for an escape hatch -- check whether "
        "``inspect.signature`` has started failing on a whole family"
    )


def test_a_call_missing_a_required_argument_is_detected(tmp_path: Path) -> None:
    """The shape that shipped: a real class, constructed without what it needs."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\nfrom dataknobs_structures import TextMetaData\nmeta = TextMetaData()\n```\n"
    )
    sites, _ = call_sites(doc)
    assert len(sites) == 1, f"expected the one call, got {sites}"
    _, _, target, node = sites[0]
    positional, keywords = _arity(node)
    with pytest.raises(TypeError):
        inspect.signature(target).bind(*positional, **keywords)


def test_a_call_passing_an_unknown_keyword_is_detected(tmp_path: Path) -> None:
    """The other half of the population: a keyword that was never there."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_config import Config\n"
        "config = Config.from_file('c.yaml', apply_env_overrides=True)\n"
        "```\n"
    )
    sites, _ = call_sites(doc)
    assert len(sites) == 1, f"expected the one call, got {sites}"
    _, _, target, node = sites[0]
    positional, keywords = _arity(node)
    with pytest.raises(TypeError):
        inspect.signature(target).bind(*positional, **keywords)


def test_a_correct_call_is_not_flagged(tmp_path: Path) -> None:
    """The floor under the two above: a good call must stay quiet."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_structures import TextMetaData\n"
        "meta = TextMetaData('doc-1', source='input.txt')\n"
        "```\n"
    )
    sites, _ = call_sites(doc)
    assert len(sites) == 1, f"expected the one call, got {sites}"
    _, _, target, node = sites[0]
    positional, keywords = _arity(node)
    inspect.signature(target).bind(*positional, **keywords)


def test_an_unpacked_call_is_skipped_rather_than_guessed(tmp_path: Path) -> None:
    """``**kwargs`` at a call site hides its arity; the reader must not invent one."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_structures import TextMetaData\n"
        "meta = TextMetaData(**settings)\n"
        "```\n"
    )
    sites, unanalysable = call_sites(doc)
    assert sites == [], f"an unpacked call must not be analysed, got {sites}"
    assert unanalysable == 1, f"it must still be counted, got {unanalysable}"


def test_an_illustrative_fence_is_not_read_for_calls(tmp_path: Path) -> None:
    """The marker that exempts an import exempts the calls beneath it too."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "<!-- dk-imports: illustrative -- the API before the migration -->\n"
        "```python\n"
        "from dataknobs_structures import TextMetaData\n"
        "meta = TextMetaData()\n"
        "```\n"
    )
    sites, _ = call_sites(doc)
    assert sites == [], f"an illustrative fence must not be read, got {sites}"


def test_binding_findings_report_one_when_the_tree_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the finding path, as the checks above each have one."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_structures import TextMetaData\n"
        "good = TextMetaData('doc-1')\n"
        "bad = TextMetaData()\n"
        "```\n"
    )
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    found = binding_findings()

    assert len(found) == 1, f"expected the one unbindable call, got {found}"
    assert "sample.md:4" in found[0], f"wrong line reported: {found[0]}"


# --- The receiver the other three readers cannot hold ----------------------
#
# Every reader above stops at the same boundary, and says so: the receiver must
# be a name the fence imported. ``hot_reload.shutdown()`` on a local is out of
# reach, so a fence can pass all three while being unrunnable on its third
# line -- which is the failure this file exists to refuse, one step further out.
#
# The step that is cheap to take is the one where the local's type is written
# in the fence itself: ``manager = HotReloadManager(...)`` names its own class,
# and from there ``manager.shutdown()`` is the same question ``declares``
# already answers. Twenty-five such calls named a method that does not exist
# when this was written, across eleven documents -- two of them inside fences a
# commit had just corrected by keyword while leaving the method below fictional.
#
# It stays conservative in three ways, because a false positive here is what
# would get the whole family switched off:
#
# - **Only a name assigned exactly once**, and assigned directly from a call on
#   an imported class. A name the fence binds twice says nothing reliable.
# - **Only method calls.** ``obj.attribute`` may be set in ``__init__`` and is
#   invisible from the class; ``obj.method()`` is a class attribute or it does
#   not exist. The narrower question is the answerable one.
# - **Never against an attribute the class assigns to ``self``.** A callable
#   stored on the instance is a real method call whose name the class object
#   does not carry, so the class's own source is read before a finding stands.


@cache
def _assigned_on_self(cls: type) -> frozenset[str]:
    """Attribute names ``cls`` or its bases assign to ``self``.

    An instance attribute is invisible on the class, so a call through one
    (``self._client = build(); obj._client.get()``) would read as a method that
    does not exist. Reading the class's own source is what keeps that out of
    the findings, and it is paid for only on the handful that would otherwise
    be reported.
    """
    names: set[str] = set()
    for base in cls.__mro__[:-1]:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
        except (OSError, TypeError, SyntaxError, IndentationError):
            continue  # a class with no readable source cannot be asked
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                names.add(node.attr)
    return frozenset(names)


def _bind_counts(tree: ast.Module) -> dict[str, int]:
    """How many times the fence binds each name.

    ``_rebound`` answers *whether* a name is bound, which cannot distinguish
    the single assignment that gives a local a knowable type from the second
    one that takes it away again.
    """
    counts: dict[str, int] = {}

    def bind(target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                counts[node.id] = counts.get(node.id, 0) + 1

    def arguments(args: ast.arguments) -> None:
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            counts[argument.arg] = counts.get(argument.arg, 0) + 1
        for optional in (args.vararg, args.kwarg):
            if optional is not None:
                counts[optional.arg] = counts.get(optional.arg, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target)
        elif isinstance(
            node, (ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.comprehension)
        ):
            bind(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts[node.name] = counts.get(node.name, 0) + 1
            arguments(node.args)
        elif isinstance(node, ast.ClassDef):
            counts[node.name] = counts.get(node.name, 0) + 1
        elif isinstance(node, ast.Lambda):
            arguments(node.args)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            counts[node.name] = counts.get(node.name, 0) + 1
    return counts


def _constructed(tree: ast.Module, env: dict[str, object]) -> dict[str, type]:
    """Local name -> the imported class the fence constructs it from."""
    counts = _bind_counts(tree)
    made: dict[str, type] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and isinstance(node.value, ast.Call)):
            continue
        if counts.get(target.id, 0) != 1:
            continue  # bound more than once: the fence has taken the type back
        built = _call_target(node.value, env)
        if isinstance(built, type):
            made[target.id] = built
    return made


@cache
def receiver_sites(path: Path) -> list[tuple[int, str, type, str, ast.Call]]:
    """``(line, receiver, class, method, call)`` for each call on a constructed local.

    Cached as the two readers above are.
    """
    found: list[tuple[int, str, type, str, ast.Call]] = []
    for fence in code_fences(path):
        if fence.lang not in PYTHON_FENCE or ILLUSTRATIVE.match(fence.marker or ""):
            continue
        tree = parsed(fence.body)
        if tree is None:
            continue
        env, _ = _bound(tree)
        if not env:
            continue
        for name in _rebound(tree):
            env.pop(name, None)
        made = _constructed(tree, env)
        if not made:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            receiver = node.func.value
            if not isinstance(receiver, ast.Name) or receiver.id not in made:
                continue
            found.append(
                (
                    fence.line + node.lineno - 1,
                    receiver.id,
                    made[receiver.id],
                    node.func.attr,
                    node,
                )
            )
    return found


def _receiver_fault(cls: type, method: str, node: ast.Call) -> str | None:
    """Why this call on an instance of ``cls`` would fail, or ``None``.

    Two questions, asked in order, because the second is only meaningful once
    the first is answered: a method that is not there has no signature to bind
    against, and a method that is there can still be called wrongly. Splitting
    them into two readers would have walked the corpus twice to ask about the
    same resolved receiver.
    """
    if not declares(cls, method) and method not in _assigned_on_self(cls):
        return f"no such method on {cls.__module__}.{cls.__qualname__}"
    target = getattr(cls, method, None)
    if not callable(target) or _unpacks(node):
        return None
    try:
        signature = inspect.signature(target)
    except (ValueError, TypeError):
        return None  # unreadable signature: counted nowhere, guessed at never
    positional, keywords = _arity(node)
    try:
        # `self` is bound at the instance, so the recorded call supplies one
        # fewer positional argument than the unbound function expects.
        signature.bind(object(), *positional, **keywords)
    except TypeError as exc:
        return str(exc)
    return None


def receiver_findings() -> list[str]:
    """Every documented call on a constructed local that could not be made."""
    found: list[str] = []
    for path in documentation_files():
        for line, receiver, cls, method, node in receiver_sites(path):
            fault = _receiver_fault(cls, method, node)
            if fault is not None:
                found.append(f"{rel(path)}:{line}: {receiver}.{method}() -- {fault}")
    return found


def test_every_documented_method_call_reaches_a_real_method() -> None:
    """A local whose class the fence names is a receiver we can still check."""
    broken = receiver_findings()
    assert not broken, (
        f"{len(broken)} documented call(s) invoke a method the constructed "
        "object's class does not have, or pass it arguments that cannot bind, "
        "so the sample fails partway down a fence whose imports and top-level "
        "signatures are all correct:\n  "
        + "\n  ".join(broken)
        + "\n\nRepoint the call at the method that exists. If the call is not "
        "meant to resolve, mark the fence with "
        "<!-- dk-imports: illustrative -- why --> as an import would be."
    )


def test_the_receiver_scan_reads_a_meaningful_corpus() -> None:
    """Non-vacuity: this reader filters the corpus harder than any above it.

    A fence must import a class from the namespace, construct it into a local
    bound exactly once, and then call a method on that local. Every count above
    can sit at its full value while this returns nothing, so it needs a floor
    of its own. The number is placed below what the tree holds (1,082 when this
    was last measured) and well above any one document's contribution.
    """
    found = sum(len(receiver_sites(path)) for path in documentation_files())
    assert found > 700, (
        f"only {found} calls on constructed locals found; the documents that "
        "build an object and then use it have not gone away, so the likelier "
        "reading is that ``_constructed`` has stopped binding one of the forms"
    )


def test_a_call_on_a_constructed_local_is_checked(tmp_path: Path) -> None:
    """The shape this reader was written for, and the corrected form beside it."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_common import Registry\n"
        "registry = Registry(name='tools')\n"
        "registry.register('a', 1)\n"
        "registry.deregister('a')\n"
        "```\n"
    )
    sites = receiver_sites(doc)
    assert [method for _, _, _, method, _ in sites] == ["register", "deregister"]
    broken = [method for _, _, cls, method, node in sites if _receiver_fault(cls, method, node)]
    assert broken == ["deregister"], f"expected the one phantom method, got {broken}"


def test_a_local_the_fence_rebinds_is_not_read(tmp_path: Path) -> None:
    """Assigned twice, the name no longer says what it holds."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_common import Registry\n"
        "registry = Registry(name='tools')\n"
        "registry = something_else()\n"
        "registry.anything_at_all()\n"
        "```\n"
    )
    assert receiver_sites(doc) == [], "a twice-bound local must not be read"


def test_an_attribute_the_class_sets_on_self_is_not_read_as_absent() -> None:
    """A callable stored on the instance is invisible on the class, and real."""

    class Holder:
        def __init__(self) -> None:
            self.handler = print

    assert not declares(Holder, "handler"), "the premise of this test has moved"
    assert "handler" in _assigned_on_self(Holder)


def test_an_illustrative_fence_is_not_read_for_receivers(tmp_path: Path) -> None:
    """The marker covers this reader too, or it covers three quarters of a fence."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "<!-- dk-imports: illustrative -- the API before the rename -->\n"
        "```python\n"
        "from dataknobs_common import Registry\n"
        "registry = Registry(name='tools')\n"
        "registry.deregister('a')\n"
        "```\n"
    )
    assert receiver_sites(doc) == []


def test_receiver_findings_report_one_when_the_tree_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity for the finding path, as every reader here has one."""
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_common import Registry\n"
        "registry = Registry(name='tools')\n"
        "registry.register('a', 1)\n"
        "registry.deregister('a')\n"
        "```\n"
    )
    monkeypatch.setitem(globals(), "documentation_files", lambda: [doc])
    monkeypatch.setitem(globals(), "rel", str)

    found = receiver_findings()

    assert len(found) == 1, f"expected the one phantom method, got {found}"
    assert "sample.md:5" in found[0], f"wrong line reported: {found[0]}"


def test_a_method_call_with_a_bad_argument_is_detected(tmp_path: Path) -> None:
    """The method is real and the call still cannot be made.

    This is the half the existing binding reader structurally cannot reach: it
    resolves a callable only through an imported name, and ``batch_ops`` is a
    local. Checking arity here costs one more question of a receiver already
    resolved.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "```python\n"
        "from dataknobs_common import Registry\n"
        "registry = Registry(name='tools')\n"
        "registry.register('a', 1)\n"
        "registry.register('b', 2, nonexistent_keyword=True)\n"
        "```\n"
    )
    faults = [
        (method, _receiver_fault(cls, method, node))
        for _, _, cls, method, node in receiver_sites(doc)
    ]
    assert [f for _, f in faults].count(None) == 1, f"the good call must stay quiet: {faults}"
    bad = [f for _, f in faults if f is not None]
    assert len(bad) == 1 and "nonexistent_keyword" in bad[0], bad
