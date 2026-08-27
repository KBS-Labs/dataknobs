#!/usr/bin/env python3
"""Doc-sync guard: enforce the package<->site doc-mirror invariant.

The dual-docs rule keeps two copies of most package docs:

  * package-local  ``packages/<pkg>/docs/*``   (GitHub)
  * mkdocs site    ``docs/packages/<pkg>/*``    (site)

Both trees spell a doc the same way -- lower-hyphen -- so a bare link to a
sibling doc reads identically from either.

Historically nothing enforced that the two agree, so pages drifted silently
until the rendered site taught a fictional API. This guard closes that gap.

Every pair is classified in ``.dataknobs/docs-mirror-manifest.json``. An entry
key this guard does not know is refused rather than skipped -- see
:func:`check_known_classes`, and the ``mirror`` note at the end of this
docstring for why that refusal was worth adding the day the class went away:


  ``symlink``     Site page is a symlink to the package source; drift is
                  structurally impossible. The guard verifies the site path is
                  a symlink resolving to the package source.
  ``transclude``  Site page is a pymdownx ``--8<--`` include of the package
                  source; drift is structurally impossible. The guard only
                  verifies the include still points at the source.
  ``diverge``     Intentional content divergence (structural landing page,
                  faithful condensation, independent elaboration). Recorded,
                  not content-checked; both files must exist. May additionally
                  declare ``shared_sections``: named blocks that, despite the
                  surrounding divergence, are transcluded from the package
                  source rather than copied. Each is verified at both ends --
                  the source carries the ``--8<-- [start:name]`` / ``[end:name]``
                  markers, the site page carries the matching
                  ``--8<-- "<source>:<name>"`` include. This is the shape for a
                  pair that is two genuinely different documents sharing one
                  block; without it such a block has no expressible
                  classification and ends up hand-copied and unverified.
  ``package_only``  Package doc with no site mirror.
  ``site_only``     Site-native page with no package source.

Neither unpaired class has a per-class check -- they appear only in the
completeness pass, which any classification satisfies. So naming a real pair
in one of them opts it out of every invariant here, silently. ``check_unpaired``
closes that: an entry whose counterpart exists in the other tree (matched on
the canonicalized basename, at any depth) fails. Two same-named documents that
are genuinely unrelated are recorded with ``diverge`` and a reason.

Beyond the per-class invariant, three properties hold across every doc:

  **One document, one name.** A paired doc is spelled the same in both trees,
  so a bare link to a sibling reads identically from either. ``diverge`` is
  exempt -- that class records two genuinely different documents, and
  requiring them to share a name would contradict the classification.

  **Lower-hyphen spelling.** Every package doc is ``lower-hyphen.md``
  (``README.md`` excepted -- GitHub renders it as a directory index). This is
  what stops the old ``UPPER_SNAKE.md`` convention returning through a doc
  that is new, or unpaired, and so invisible to the rule above.

  **Links resolve.** Every relative ``.md`` link resolves -- case-sensitively --
  in every tree its document is served from, and one that does not fails. Two
  shapes, one verdict, different remedies: a link broken by *spelling* names
  the rename that fixes it, while one whose target is absent under any spelling
  names the three that do -- an absolute site URL, publishing the target into
  the package tree, or a prose mention. The second shape was counted and
  printed rather than failed while the question it posed was open, because the
  two trees nest some documents differently and no relative path reaches such a
  target from both. None of the three remedies is declared anywhere: an
  absolute URL is not a relative ``.md`` link and a prose mention is not a link
  at all, so all three are invisible here by construction.

Completeness: every ``*.md`` **in scope** MUST be classified. An unclassified
file (or a manifest entry with no file on disk) fails the check -- that is
what makes silent drift impossible to introduce: a doc in scope forces a
classification decision at PR time.

Scope is per package, set by the manifest's ``recursive`` flag:

  ``false`` (default)  Only *top-level* ``*.md``. A paired entry may still
                       point at a subdirectory on either side -- a package
                       source under ``guides/`` (a transclusion of
                       ``guides/events.md``), or a site page under
                       ``guides/`` (where every bots guide lives). Such a
                       file is exempt from the completeness set, though its
                       existence and the pair's invariant are still enforced
                       by the per-class check. The gap: a *new* subdirectory
                       doc is not required to be classified at all, so it
                       gets no verification.
  ``true``             Every ``*.md`` at any depth, keyed by its path
                       relative to the tree root. Closes that gap.

Opting a package in requires classifying everything already nested in it,
which is per-package work -- hence the flag, so packages are reconciled one
at a time rather than holding the guarantee hostage to one sweeping change.

There was a sixth class, ``mirror``: a hand-authored site copy held
byte-identical to its package source by a content comparison, with a
canonicaliser for link filenames and a ``line_exceptions`` list for the lines
that genuinely had to read differently in each tree. It existed for exactly one
reason -- it was the only class holding two real files, so it was the only one
able to carry a per-tree link text. Once every such link became an absolute
site URL there was nothing left for it to express, and a class that guarantees
by *comparison* what two other classes guarantee by *construction* is strictly
the weaker way to say the same thing. Its twelve pairs are ``transclude`` now,
and with it went the canonicaliser, the exception machinery and ``--fix``,
whose only writer regenerated a ``mirror`` page.

Modes:

  ``--check`` (default)  Exit 1 on any drift / unclassified / missing file.
                         The only mode; it is accepted for compatibility with
                         callers that pass it and does nothing on its own.
  ``--package <name>``   Restrict to one package in the manifest (default: all).

Standard library only -- runs under the CI runner's system ``python3`` with no
``uv sync``, exactly like ``bin/docs-update-versions.sh``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / ".dataknobs" / "docs-mirror-manifest.json"

# ANSI colors (disabled when stdout is not a TTY, matching the other bin/ scripts
# gracefully in CI logs).
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def red(t: str) -> str:
    return _c("0;31", t)


def green(t: str) -> str:
    return _c("0;32", t)


def cyan(t: str) -> str:
    return _c("0;36", t)


# A markdown inline-link target, in the ``](target)`` position, with an optional
# ``"title"`` suffix left untouched. This used to serve two readers -- a
# canonicaliser that rewrote a target and a resolver that checks one -- and the
# canonicaliser retired with the `mirror` class, so `link_targets` is now its
# only caller.
_LINK_RE = re.compile(r"\]\((?P<target>[^)\s]+)(?P<rest>[^)]*)\)")


# A fenced-code-block delimiter: first non-space content is a run of 3+ backticks
# or tildes (optionally followed by an info string). ``](target)`` text inside a
# fence is a literal code example, not a real link, and must not be rewritten.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# An inline code span: a run of N backticks closed by a run of N backticks. Link
# syntax shown literally in a `code span` must not be rewritten either.
_CODE_SPAN_RE = re.compile(r"(`+).*?\1")


def _protected_spans(line: str) -> list[tuple[int, int]]:
    """Character ranges of ``line`` holding literal example text.

    Link syntax inside an inline ``code span`` is a sample, not a link. The
    rule is defined once here because it once had two callers that had to agree
    about it; only the resolver is left, and the definition stays separate
    because what counts as literal text is a property of markdown rather than
    of resolution.
    """
    return [(m.start(), m.end()) for m in _CODE_SPAN_RE.finditer(line)]


def _iter_lines(text: str) -> Iterator[tuple[str, bool]]:
    """Yield ``(line, literal)`` for each line, tracking fenced-code state.

    ``literal`` is True for a fence delimiter and for everything between an
    opening and closing run: that content is a code sample, so a ``](target)``
    in it is not a link to resolve. Same rationale as :func:`_protected_spans`,
    one line-scope up.
    """
    fence: str | None = None
    for ln in text.splitlines():
        m = _FENCE_RE.match(ln)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            yield ln, True
            continue
        yield ln, fence is not None


def link_targets(text: str) -> list[str]:
    """Every relative ``.md`` link target in the prose of ``text``, in order.

    Fenced blocks and inline code spans contribute nothing, via ``_iter_lines``
    and ``_protected_spans``. Skipped as well: anything with a URL
    scheme or a leading ``/`` (not resolved against a doc tree at all) and any
    target with no ``.md`` file part (a same-page ``#anchor``, an image, a
    ``.py``). What is left is exactly the population that has to resolve inside
    one of the two trees.
    """
    out: list[str] = []
    for line, literal in _iter_lines(text):
        if literal:
            continue
        protected = _protected_spans(line)
        for m in _LINK_RE.finditer(line):
            if any(start <= m.start() < end for start, end in protected):
                continue
            file_part = m.group("target").partition("#")[0]
            if not file_part or ":" in file_part or file_part.startswith("/"):
                continue
            if file_part.endswith(".md"):
                out.append(file_part)
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
    """``path`` shown relative to the repo root, or absolute if it is outside it.

    A link target can normalise its way above ``ROOT``, and a symlink can
    resolve anywhere; both are worth naming in a message rather than crashing
    the check that found them.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


class Result:
    """Accumulates errors and warnings for a run."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_symlink(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        res.fail(f"symlink: package source missing: {pkg_path.relative_to(ROOT)}")
        return
    if not site_path.is_symlink():
        res.fail(
            f"symlink: {site_path.relative_to(ROOT)} is classified as a symlink to "
            f"the package source but is not a symlink. Restore the symlink "
            f"(`ln -s`) or reclassify the pair in {MANIFEST.relative_to(ROOT)}."
        )
        return
    if site_path.resolve() != pkg_path.resolve():
        shown = _rel(site_path.resolve())
        res.fail(
            f"symlink: {site_path.relative_to(ROOT)} resolves to '{shown}' but "
            f"should point at the package source '{pkg_path.relative_to(ROOT)}'."
        )


def check_transclude(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        res.fail(f"transclude: package source missing: {pkg_path.relative_to(ROOT)}")
        return
    if not site_path.exists():
        res.fail(f"transclude: site page missing: {site_path.relative_to(ROOT)}")
        return

    want = f"{(pkg_dir / pair['package']).relative_to(ROOT).as_posix()}"
    include_re = re.compile(r'^\s*(?:-{2,}|;{2,})8<-{2,}\s+"(?P<path>[^"]+)"')
    found = None
    for ln in _read(site_path).splitlines():
        m = include_re.match(ln)
        if m:
            found = m.group("path")
            if found == want:
                return
    if found is None:
        res.fail(
            f"transclude: {site_path.relative_to(ROOT)} is classified as a "
            f'transclusion but contains no `--8<-- "..."` include line. If it '
            f"is now a hand-authored copy, restore the include -- a hand copy "
            f"has no classification here, and `diverge` is for two genuinely "
            f"different documents, not a drifted copy of one. See "
            f"{MANIFEST.relative_to(ROOT)}."
        )
    else:
        res.fail(
            f"transclude: {site_path.relative_to(ROOT)} includes '{found}' but "
            f"should include the package source '{want}'."
        )


def _has_section_marker(text: str, name: str, edge: str) -> bool:
    """Whether ``text`` carries the pymdownx ``[start:]``/``[end:]`` marker.

    The marker is normally written inside an HTML comment so it stays
    invisible when the package doc is read directly on GitHub, so this
    matches it anywhere on the line rather than anchoring to the margin.
    """
    pattern = rf"(?:-{{2,}}|;{{2,}})8<-{{2,}}\s+\[{edge}:{re.escape(name)}\]"
    return re.search(pattern, text) is not None


def check_shared_sections(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    """Verify the named sections a divergent pair shares structurally.

    Two docs may be different documents overall yet still need one block to
    stay identical. Copying that block by hand puts it back outside every
    guarantee this guard exists to provide -- and because the pair as a whole
    is genuinely divergent, a whole-file ``transclude`` cannot express it,
    which is how such a block ends up unclassified and unverified.

    ``shared_sections`` closes that: the block lives once in the package
    source between pymdownx section markers, and the site page pulls it in
    with ``--8<-- "<source>:<section>"``. Drift is then impossible by
    construction. What remains possible is someone replacing the include with
    a fresh hand-copy, or deleting the markers -- so this checks both ends of
    the arrangement still exist.
    """
    sections = pair.get("shared_sections") or []
    if not sections:
        return

    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists() or not site_path.exists():
        # The missing file is already reported by the caller; reporting every
        # section against it too would bury that one real error.
        return

    pkg_text = _read(pkg_path)
    want_source = pkg_path.relative_to(ROOT).as_posix()
    include_re = re.compile(r'^\s*(?:-{2,}|;{2,})8<-{2,}\s+"(?P<path>[^"]+)"')
    included = set()
    for ln in _read(site_path).splitlines():
        m = include_re.match(ln)
        if m:
            included.add(m.group("path"))

    rel_pkg = pkg_path.relative_to(ROOT)
    rel_site = site_path.relative_to(ROOT)
    for name in sections:
        absent = [
            edge for edge in ("start", "end") if not _has_section_marker(pkg_text, name, edge)
        ]
        if absent:
            res.fail(
                f"shared_sections: {rel_pkg} is missing the "
                f"{' and '.join(absent)} marker for section '{name}'. The "
                f"site page includes it, so removing the marker breaks the "
                f"docs build. Restore `<!-- --8<-- [start:{name}] -->` / "
                f"`<!-- --8<-- [end:{name}] -->` around the shared block, or "
                f"drop '{name}' from shared_sections in "
                f"{MANIFEST.relative_to(ROOT)}."
            )
        want = f"{want_source}:{name}"
        if want not in included:
            res.fail(
                f"shared_sections: {rel_site} does not include "
                f'`--8<-- "{want}"`. A section declared as shared must be '
                f"transcluded, not copied -- a hand-authored copy drifts "
                f"silently, which is the failure this classification exists "
                f"to prevent."
            )


def check_diverge(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        res.fail(f"diverge: package source missing: {pkg_path.relative_to(ROOT)}")
    if not site_path.exists():
        res.fail(f"diverge: site page missing: {site_path.relative_to(ROOT)}")
    check_shared_sections(pair, pkg_dir, site_dir, res)


def _canon_name(name: str) -> str:
    """The comparison key for one doc: its basename, depth discarded.

    ``user-guide.md`` and ``guides/user-guide.md`` share it; so do
    ``html/html-conversion.md`` and ``html-conversion.md``. That is what
    lets an entry be matched against a counterpart the other tree nests
    differently.

    The case/underscore fold is a residual. Both trees now spell every
    doc in lower-hyphen, so it folds nothing that currently exists; it
    stays because nothing yet *enforces* that spelling, and a doc added
    in the old convention should still match its counterpart rather than
    silently read as unpaired.
    """
    return Path(name).name.lower().replace("_", "-")


def _exists_cs(path: Path) -> bool:
    """Case-SENSITIVE existence, answered from the parent's directory listing.

    ``Path.exists()`` answers *yes* on a case-insensitive checkout when a link
    says ``configuration.md`` and only ``CONFIGURATION.md`` is on disk. That is
    not a hypothetical risk: it is why 89 broken package-tree links passed
    every local run until they were measured this way. Comparing the name
    against the listing gives macOS the same verdict as the CI runner, which is
    the only way this check means anything where it is actually written.
    """
    try:
        return path.name in {p.name for p in path.parent.iterdir()}
    except OSError:
        return False


def _fold_sibling(path: Path) -> str | None:
    """A name in ``path``'s directory differing from it only in spelling.

    :func:`_canon_name` folds case and underscores, so a hit means "the
    document is right there, spelled the other way" -- the one kind of
    unresolved link a rename can fix, and the kind the one-name convention
    exists to make impossible.
    """
    try:
        siblings = {p.name for p in path.parent.iterdir()}
    except OSError:
        return None
    want = _canon_name(path.name)
    for name in sorted(siblings):
        if name != path.name and _canon_name(name) == want:
            return name
    return None


def check_unpaired(entry: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    """``package_only`` / ``site_only`` must mean unpaired, not unclassified.

    Neither class has a per-class check -- they appear only in the
    completeness pass, which is satisfied by *any* classification. So
    naming a doc in one of them opts it out of every invariant this
    guard exists to enforce, and the package still reports clean while
    the counterpart drifts, moves or disappears.

    That made them the path of least resistance whenever a pair was
    awkward to express, and the awkwardness was real: until
    subdirectory site pages could be paired at all, every bots guide
    *had* to be recorded this way. The guard changed; the entries left
    behind did not, and nothing noticed, because a wrong answer here
    looks exactly like a right one.

    This is what notices. A counterpart is anything in the other tree
    whose basename matches under :func:`_canon_name`, at any depth --
    the completeness pass is scoped by ``recursive`` but this is not,
    since the question is whether a partner exists, not whether it is
    in the top-level set.

    Two same-named documents that are genuinely unrelated are not a
    false positive to suppress: say so with ``diverge`` and a reason.
    Recording the pair is the point -- both files then get an existence
    check, and the reason survives for whoever reads it next.
    """

    def _counterparts(root: Path) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for path in root.rglob("*.md"):
            found.setdefault(_canon_name(path.name), []).append(path.relative_to(root).as_posix())
        return found

    paired: set[str] = set()
    for kind in _PAIRED_CLASSES:
        for pair in entry.get(kind, []):
            paired.add(pair["site"])
            paired.add(pair["package"])

    for cls, names, own_dir, other_dir, other_side in zip(
        _UNPAIRED_CLASSES,
        (entry.get("package_only", []), entry.get("site_only", [])),
        (pkg_dir, site_dir),
        (site_dir, pkg_dir),
        ("site", "package"),
        strict=True,
    ):
        counterparts = _counterparts(other_dir)
        for name in names:
            # Existence is this check's job for these two classes, and only
            # for these two. A pair's file is opened by its own per-class
            # check, and every classified top-level name is compared against
            # the on-disk set by ``check_completeness`` -- but a nested name
            # is exempt from that set precisely so it can be classified at
            # all, and nothing below asks whether it is there. The absence
            # of a counterpart is what this function is looking for, so a
            # name that points at nothing looks like a clean answer.
            if not (own_dir / name).exists():
                res.fail(
                    f"{cls}: '{name}' is classified but no such file exists "
                    f"({(own_dir / name).relative_to(ROOT)}). A rename leaves the "
                    f"entry behind and a typo never had a file; either way the "
                    f"document this was meant to cover is verified by nothing."
                )
                continue
            hits = [h for h in sorted(counterparts.get(_canon_name(name), [])) if h not in paired]
            if not hits:
                continue
            res.fail(
                f"{cls}: '{name}' is classified as unpaired, but the {other_side} "
                f"tree has {', '.join(hits)}. {cls} carries no per-class check, so "
                f"this pair is currently verified by nothing. Classify it "
                f"(symlink / transclude), or record the divergence with "
                f"`diverge` and a reason if the two are genuinely different documents."
            )


#: Classes whose two sides are the *same* document. Both survivors are one
#: file served at two paths, so name parity looks briefly redundant -- it is
#: not. A ``transclude`` site page is a real file whose *content* names the
#: source, and nothing about writing ``--8<-- "packages/x/docs/a.md"`` into
#: ``docs/packages/x/b.md`` prevents it; the include would resolve and the two
#: paths would disagree. This is the check that says so.
#:
#: ``diverge`` is deliberately absent: it records two genuinely different
#: documents that happen to be counterparts, so requiring them to share a name
#: would contradict the classification. Two pairs differ today and both are
#: correct.
_SAME_DOCUMENT_CLASSES = ("symlink", "transclude")

#: Every class that pairs two paths. ``_SAME_DOCUMENT_CLASSES`` is the subset
#: whose two paths carry one text; ``diverge`` is the rest, and the distinction
#: is physics rather than taste -- see :func:`_served_docs`.
_PAIRED_CLASSES = (*_SAME_DOCUMENT_CLASSES, "diverge")

#: Classes naming one path, with nothing in the other tree.
_UNPAIRED_CLASSES = ("package_only", "site_only")

#: Keys in a package entry that are not classes.
_ENTRY_META = ("package_dir", "site_dir", "recursive")

#: Retired classes, named so the refusal can say what happened rather than
#: only that the key is unknown. ``mirror`` held a hand-authored site copy
#: byte-identical to its source; every one of its pairs is a ``transclude``
#: now, which guarantees by construction what it guaranteed by comparison.
_RETIRED_CLASSES = {
    "mirror": "retired -- use `transclude` (or `symlink`); one text, two paths, "
    "so there is nothing left to compare",
}

#: A conforming doc filename: lower case, digits, hyphens and dots only.
_LOWER_HYPHEN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.md$")

#: GitHub renders ``README.md`` as a directory index; ``readme.md`` on a
#: case-sensitive host does not get that treatment. The only exemption.
_SPELLING_EXEMPT = frozenset({"README.md"})


def check_name_parity(entry: dict, res: Result) -> None:
    """One document, one name: a paired doc is spelled the same in both trees.

    This is what lets a bare link to a sibling read identically from either
    tree -- and so what lets ``symlink`` and ``transclude``, one file served at
    two paths, carry cross-doc links at all. Without it the same link text can
    resolve in at most one of the trees the document is served from, which is
    how 89 package-tree links came to be broken while the rendered site stayed
    clean and every guard reported green.

    Only the basename is compared. The two trees nest some documents
    differently and no filename reconciles that; :func:`check_link_resolution`
    is where the consequences of the nesting show up.
    """
    for kind in _SAME_DOCUMENT_CLASSES:
        for pair in entry.get(kind, []):
            pkg_name = Path(pair["package"]).name
            site_name = Path(pair["site"]).name
            if pkg_name == site_name:
                continue
            res.fail(
                f"name parity: {kind} pair '{pair['package']}' <-> '{pair['site']}' "
                f"spells one document two ways ('{pkg_name}' / '{site_name}'). A "
                f"bare link to it then resolves in at most one tree. Give both "
                f"sides the same filename, or -- if these are really two "
                f"different documents -- record them as `diverge` with a reason "
                f"in {MANIFEST.relative_to(ROOT)}."
            )


def check_doc_spelling(pkg_dir: Path, res: Result) -> None:
    """Every package doc filename is lower-hyphen.

    The package tree is where the old ``UPPER_SNAKE.md`` convention lived, where
    every unresolved link was found, and the tree an author populates by
    following ``docs/development/new-package-checklist.md``. A doc added in the
    old spelling breaks a sibling's bare link the moment someone writes one, and
    :func:`check_name_parity` would not see it: a new doc need not be paired.

    The site tree is covered transitively rather than directly. A paired site
    page must match its package counterpart, which this requires to be
    lower-hyphen -- so the only site pages outside the rule are the genuinely
    site-only ones, whose names are published URLs and are sometimes taken from
    the module they document (``fsm/api/async_simple.md``). Renaming those would
    move a URL to fix nothing: a document served from one tree has no second
    tree for its link text to disagree with.

    Read from disk, and recursively whatever the package's ``recursive`` flag
    says. That flag scopes which docs must be *classified*; a doc can be spelled
    wrong without being classified at all, and that is the case worth catching.
    """
    if not pkg_dir.is_dir():
        return
    for path in sorted(pkg_dir.rglob("*.md")):
        if path.name in _SPELLING_EXEMPT or _LOWER_HYPHEN_RE.match(path.name):
            continue
        res.fail(
            f"doc spelling: {_rel(path)} is not lower-hyphen. Both trees spell a "
            f"document the same way, so a bare link to it reads identically from "
            f"either; uppercase or underscores break that for every doc that "
            f"links to it. Rename it to '{_canon_name(path.name)}' and update the "
            f"references."
        )


def _served_docs(entry: dict, pkg_dir: Path, site_dir: Path) -> Iterator[tuple[Path, list[Path]]]:
    """Yield ``(file, [directory, ...])`` for every classified doc.

    A ``symlink`` or ``transclude`` pair is **one** document served at two
    paths, so its text is read once and must resolve from both directories: the
    symlink is the same file under the site path, and a transclusion is inlined
    into the site page, so the source's relative links are resolved from the
    site page's location. The remaining classes hold two independent files, each
    served only from its own tree.
    """
    for kind in _SAME_DOCUMENT_CLASSES:
        for pair in entry.get(kind, []):
            yield (
                pkg_dir / pair["package"],
                [(pkg_dir / pair["package"]).parent, (site_dir / pair["site"]).parent],
            )
    for pair in entry.get("diverge", []):
        yield pkg_dir / pair["package"], [(pkg_dir / pair["package"]).parent]
        yield site_dir / pair["site"], [(site_dir / pair["site"]).parent]
    for name in entry.get("package_only", []):
        yield pkg_dir / name, [(pkg_dir / name).parent]
    for name in entry.get("site_only", []):
        yield site_dir / name, [(site_dir / name).parent]


def check_link_resolution(entry: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    """A relative ``.md`` link resolves in every tree its document is served from.

    One rule, and it took two passes to get here. Both populations below fail;
    they differ only in the remedy their message names, because the remedies are
    genuinely different and a single message could name neither.

    **Spelling.** The target is absent from that directory but a file differing
    from it only in case or underscores is present. The document is right there
    under the other name, so either the link or the file is misspelled and a
    rename fixes it.

    **Everything else.** The target is absent under any spelling, so no rename
    reaches it: the two trees nest the same document differently
    (``packages/<pkg>/docs/`` against ``docs/packages/<pkg>/guides/``), or the
    target is site-native, or it is gone. This half was counted and printed
    rather than failed while it was still a question -- what may a package doc
    link to, when a relative path cannot answer? -- and it is failed now that
    the answer exists: link the published page by its absolute site URL, publish
    the target into the package tree, or name it in prose without a link.

    Nothing has to declare which of those was chosen. An absolute URL is not a
    relative ``.md`` link and a prose mention is not a link at all, so all three
    remedies are invisible to this check by construction rather than by
    allowlist -- which is what keeps the standing maintenance cost at zero.
    """
    seen: set[tuple[str, str, str]] = set()
    for src, dirs in _served_docs(entry, pkg_dir, site_dir):
        if not src.is_file():
            # A classified file that is not there is already reported by its own
            # class check (or by the completeness pass); saying so again here,
            # once per link it does not have, would bury that one real error.
            continue
        try:
            text = _read(src)
        except (OSError, UnicodeDecodeError):
            continue
        targets = link_targets(text)
        if not targets:
            continue
        rel_src = _rel(src)
        for base in dirs:
            for target in targets:
                path = (base / target).resolve() if "/" in target else base / target
                if _exists_cs(path):
                    continue
                key = (rel_src, target, _rel(path.parent))
                if key in seen:
                    continue
                seen.add(key)
                fold = _fold_sibling(path)
                if fold is None:
                    res.fail(
                        f"link resolution: {rel_src} links to '{target}', which does "
                        f"not exist in {key[2]} under any spelling. The document is "
                        f"served from that directory, so the link has to resolve "
                        f"there: link the published page by its absolute site URL "
                        f"(https://kbs-labs.github.io/dataknobs/...), publish the "
                        f"target into the package tree, or name it in prose without "
                        f"a link."
                    )
                    continue
                res.fail(
                    f"link spelling: {rel_src} links to '{target}', which does not "
                    f"exist in {key[2]} -- but '{fold}' does. Both trees spell a "
                    f"document the same way, so the link and the file have to "
                    f"agree: rename the file, or correct the link."
                )


def check_known_classes(entry: dict, res: Result) -> None:
    """Every key in a package entry is one this guard acts on.

    Nothing checked this, and the omission was structural rather than an
    oversight: the class names were enumerated at five call sites, so an
    unrecognised key simply never matched any of them. A pair recorded under
    ``symlnk``, or under ``mirror`` after that class was retired, was opted out
    of every invariant here.

    How loudly that failed depended on where the files sat, which is the worst
    property a silence can have. A *top-level* doc nobody classified fails
    :func:`check_completeness` -- the right verdict, arrived at for the wrong
    reason, and reported as "unclassified" to someone looking at a manifest
    entry that plainly classifies it. A *nested* doc in a non-recursive package
    is not in the completeness set at all, so the same mistake is silent and
    the package reports clean.

    Retired names get their own message. "unknown key" is true of ``mirror``
    and unhelpful: whoever wrote it was following a convention that used to be
    correct, and the thing they need to know is what replaced it.

    A leading underscore means commentary and is exempt, which is not a
    concession -- it is the convention the manifest already runs on, and this
    check found it rather than being written around it. ``_note`` and
    ``_schema`` sit at the top level, and ``structures`` and ``utils`` each
    carry a per-package ``_note`` explaining why the package is all
    ``site_only``. JSON has no comments; a reserved prefix is how a document
    like this one carries its own reasoning, and a guard that refused it would
    be telling authors to delete the explanation.
    """
    known = {*_PAIRED_CLASSES, *_UNPAIRED_CLASSES, *_ENTRY_META}
    for key in entry:
        if key in known or key.startswith("_"):
            continue
        if key in _RETIRED_CLASSES:
            res.fail(
                f"manifest: '{key}' is {_RETIRED_CLASSES[key]}. Its entries are "
                f"classified by nothing, so both files are verified by nothing."
            )
        else:
            res.fail(
                f"manifest: unknown key '{key}'. Entries under it are classified "
                f"by nothing and silently unverified. Known: "
                f"{', '.join(sorted(known))}."
            )


def check_completeness(entry: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    """Every *.md in both trees must be classified exactly once.

    Scope depends on the package's ``recursive`` flag:

    * ``false`` (default) -- only *top-level* ``*.md`` must be classified.
      Subdirectory files may be classified (as one side of a pair) but are
      not required to be, so a new ``guides/whatever.md`` passes unnoticed.
    * ``true`` -- every ``*.md`` at any depth must be classified, keyed by
      its path relative to the tree root.

    The flag exists because turning recursion on for a package demands
    classifying everything already nested there, which is per-package work.
    Making it opt-in lets packages be reconciled one at a time instead of
    holding the guarantee hostage to a single sweeping change; a package
    that has not opted in keeps exactly its previous behaviour.
    """
    recursive = bool(entry.get("recursive", False))
    pkg_classified: dict[str, str] = {}
    site_classified: dict[str, str] = {}

    def _add(store: dict[str, str], name: str, bucket: str, side: str) -> None:
        if name in store:
            res.fail(
                f"manifest: {side} doc '{name}' classified twice ({store[name]} and {bucket})."
            )
        store[name] = bucket

    def _add_in_scope(store: dict[str, str], name: str, bucket: str, side: str) -> None:
        # An entry may point at a *subdirectory* on either side: a package
        # source under ``guides/`` (a transclusion of ``guides/events.md``),
        # or a site page under ``guides/`` (where every bots guide lives).
        # Under the default non-recursive scope such a file is not part of
        # the completeness set -- its existence, and its class's invariant,
        # are enforced by the per-class check instead. For the paired
        # classes that is free: each one opens both files. For the unpaired
        # ones it is not, because ``check_unpaired`` asks only whether a
        # counterpart exists, so it carries an existence assertion of its
        # own -- added when this exemption was extended to them, since the
        # exemption is what removes the completeness pass's version of it.
        #
        # Both sides must apply that rule. Exempting only the package side
        # made a subdirectory site page inexpressible as a pair -- it would
        # fail the "manifest references missing site doc" check below, since
        # ``site_on_disk`` was a non-recursive glob of basenames -- which in
        # turn forced genuine pairs to be recorded as ``package_only`` and
        # silently opted them out of per-class verification.
        #
        # So must both *populations*. The rule was first written for the
        # paired classes alone, back when the unpaired ones had no per-class
        # check to fall back on; :func:`check_unpaired` is that check, and it
        # matches on the canonicalized basename at any depth, so it reads a
        # nested path exactly as it reads a top-level one. Leaving the
        # exemption off them made a genuinely unpaired subdirectory page
        # inexpressible in the other direction -- not misclassifiable, but
        # unclassifiable, which under a non-recursive scope means silently
        # unverified.
        #
        # Under ``recursive`` the exemption is exactly what we are removing,
        # so subdirectory paths participate like any other.
        if not recursive and "/" in name:
            return
        _add(store, name, bucket, side)

    for kind in _PAIRED_CLASSES:
        for pair in entry.get(kind, []):
            _add_in_scope(pkg_classified, pair["package"], kind, "package")
            _add_in_scope(site_classified, pair["site"], kind, "site")
    for name in entry.get("package_only", []):
        _add_in_scope(pkg_classified, name, "package_only", "package")
    for name in entry.get("site_only", []):
        _add_in_scope(site_classified, name, "site_only", "site")

    def _on_disk(root: Path) -> set[str]:
        if recursive:
            return {p.relative_to(root).as_posix() for p in root.rglob("*.md")}
        return {p.name for p in root.glob("*.md")}

    pkg_on_disk = _on_disk(pkg_dir)
    site_on_disk = _on_disk(site_dir)

    for name in sorted(pkg_on_disk - set(pkg_classified)):
        res.fail(
            f"unclassified package doc: {(pkg_dir / name).relative_to(ROOT)} — "
            f"add it to {MANIFEST.relative_to(ROOT)} (symlink / transclude / "
            f"diverge / package_only)."
        )
    for name in sorted(site_on_disk - set(site_classified)):
        res.fail(
            f"unclassified site doc: {(site_dir / name).relative_to(ROOT)} — "
            f"add it to {MANIFEST.relative_to(ROOT)} (symlink / transclude / "
            f"diverge / site_only)."
        )
    for name in sorted(set(pkg_classified) - pkg_on_disk):
        res.fail(
            f"manifest references missing package doc: "
            f"{(pkg_dir / name).relative_to(ROOT)} (classified {pkg_classified[name]})."
        )
    for name in sorted(set(site_classified) - site_on_disk):
        res.fail(
            f"manifest references missing site doc: "
            f"{(site_dir / name).relative_to(ROOT)} (classified {site_classified[name]})."
        )


def run(manifest: dict, only: str | None) -> int:
    packages = manifest["packages"]
    names = [only] if only else sorted(packages)
    overall = Result()

    for name in names:
        if name not in packages:
            print(red(f"✗ package '{name}' not in manifest"), file=sys.stderr)
            return 2
        entry = packages[name]
        pkg_dir = ROOT / entry["package_dir"]
        site_dir = ROOT / entry["site_dir"]
        print(cyan(f"Doc-mirror check: {name}  ({entry['package_dir']} <-> {entry['site_dir']})"))

        res = Result()
        check_known_classes(entry, res)
        check_completeness(entry, pkg_dir, site_dir, res)
        check_unpaired(entry, pkg_dir, site_dir, res)
        check_name_parity(entry, res)
        check_doc_spelling(pkg_dir, res)
        check_link_resolution(entry, pkg_dir, site_dir, res)
        for pair in entry.get("symlink", []):
            check_symlink(pair, pkg_dir, site_dir, res)
        for pair in entry.get("transclude", []):
            check_transclude(pair, pkg_dir, site_dir, res)
        for pair in entry.get("diverge", []):
            check_diverge(pair, pkg_dir, site_dir, res)

        # `shared_sections` is only meaningful where the two files legitimately
        # differ. On the other classes it would be silently ignored -- and a
        # declaration that is quietly ignored reads as a guarantee while
        # providing none, which is worse than not offering the key at all.
        for kind in ("symlink", "transclude"):
            for pair in entry.get(kind, []):
                if pair.get("shared_sections"):
                    res.fail(
                        f"manifest: {kind} pair '{pair.get('package')}' declares "
                        f"shared_sections, which only applies to `diverge` pairs. "
                        f"A {kind} pair already shares its whole content."
                    )

        if res.ok:
            n = (
                len(entry.get("symlink", []))
                + len(entry.get("transclude", []))
                + len(entry.get("diverge", []))
            )
            print(green(f"  ✓ {n} classified pair(s) in sync; all docs classified"))
        else:
            for err in res.errors:
                print(red("  ✗ " + err.replace("\n", "\n    ")))
            overall.errors.extend(res.errors)

    print()
    if overall.ok:
        print(green("✓ Documentation mirrors are in sync"))
        return 0
    print(red(f"✗ Documentation mirror check failed ({len(overall.errors)} issue(s))"))
    print(cyan("  Each error above names the remedy for its own kind. There is no"))
    print(cyan("  blanket one, and there is no `--fix`: the flag regenerated a `mirror`"))
    print(cyan("  page from its source, and that class is gone. What is left cannot be"))
    print(cyan("  repaired by a rewrite -- a link resolves or it does not, a symlink"))
    print(cyan("  points at its source or it does not, and a doc is classified in"))
    print(cyan(f"  {MANIFEST.relative_to(ROOT)} or it is not."))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce the package<->site doc-mirror invariant.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift. The only mode; accepted so existing callers keep working.",
    )
    parser.add_argument("--package", metavar="NAME", help="Restrict to one manifest package.")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        print(red(f"✗ manifest not found: {MANIFEST}"), file=sys.stderr)
        return 2
    manifest = json.loads(_read(MANIFEST))
    return run(manifest, only=args.package)


if __name__ == "__main__":
    raise SystemExit(main())
