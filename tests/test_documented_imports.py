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

Scope is every markdown document a reader can reach: the site tree, each
package's ``docs/``, and the READMEs. Two carve-outs, both narrow and both
stated in the code below rather than left to a path convention: a document
kept as a historical record, and a block or line declaring that its subject
is the absence.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
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

    Two narrowings, and each is the difference between a finding and a correct
    document reported as one.

    **A module is never the symbol.** A module docstring carrying the marker is
    almost always deprecating a *member*: ``dataknobs_data.pooling.s3`` marks
    the single alias it re-exports, while the module itself is current and is
    named four times, correctly, by the AWS session guide. Read the module as
    deprecated and those four sentences all become findings.

    **Its own docstring, not an inherited one.** ``inspect.getdoc`` walks the
    MRO, so the first documented subclass of a deprecated base would be
    reported for inheriting a warning about its parent -- which is the shape a
    *successor* most often has. No documented name is such a subclass today, so
    the two spellings return the same set and this costs nothing, which makes
    it the cheapest moment there will ever be to choose the right one.
    """
    if symbol is None or isinstance(symbol, ModuleType):
        return False
    doc = getattr(symbol, "__doc__", None)
    return bool(doc and DEPRECATED.search(doc))


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

    The floor counts documents *reached*, not findings, so it holds steady as
    the findings are repaired -- a repaired document still names the symbol and
    still carries the notice, which is the whole point of the repair.

    Six documents are reached, and the number is placed to fail if either
    reader feeding this stops working. The two are not interchangeable and
    neither dominates: three documents name their symbol only in prose (the
    notices, which spell ``dataknobs_config.ConfigurableBase`` in running
    text), two name one only inside a fence, and one does both. So losing the
    import reader leaves four and losing the prose reader leaves three -- and
    five is the floor that fails on either, where four would have sat quietly
    through the first.
    """
    reached = [rel(path) for path in documentation_files() if deprecated_symbols(path)]
    assert len(reached) >= 5, (
        f"only {len(reached)} document(s) name a deprecated symbol at all "
        f"({', '.join(reached)}); the deprecated symbols have not gone away, "
        "so the likelier reading is that ``deprecated`` or one of the three "
        "readers feeding it has stopped recognising them -- in which case this "
        "guard is reporting green over a corpus it never read"
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
