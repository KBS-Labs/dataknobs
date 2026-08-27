#!/usr/bin/env python3
"""Doc-sync guard: enforce the package<->site doc-mirror invariant.

The dual-docs rule keeps two copies of most package docs:

  * package-local  ``packages/<pkg>/docs/*``   (GitHub)
  * mkdocs site    ``docs/packages/<pkg>/*``    (site)

Both trees spell a doc the same way -- lower-hyphen -- so a bare link to a
sibling doc reads identically from either.

Historically nothing enforced that the two agree, so pages drifted silently
until the rendered site taught a fictional API. This guard closes that gap.

Every pair is classified in ``.dataknobs/docs-mirror-manifest.json``:

  ``symlink``     Site page is a symlink to the package source; drift is
                  structurally impossible. The guard verifies the site path is
                  a symlink resolving to the package source.
  ``transclude``  Site page is a pymdownx ``--8<--`` include of the package
                  source; drift is structurally impossible. The guard only
                  verifies the include still points at the source.
  ``mirror``      Hand-authored copy: byte-identical except intra-doc ``.md``
                  link filenames (canonicalized here) plus any declared
                  per-pair line exceptions. The content-guarded invariant --
                  drift fails the check.
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

Modes:

  ``--check`` (default)  Exit 1 on any drift / unclassified / missing file.
  ``--fix``              Regenerate ``mirror`` site files from their package
                         source (canonicalize link filenames + apply declared
                         line exceptions) so ``--check`` passes by construction.
  ``--package <name>``   Restrict to one package in the manifest (default: all).

Standard library only -- runs under the CI runner's system ``python3`` with no
``uv sync``, exactly like ``bin/docs-update-versions.sh``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
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


def yellow(t: str) -> str:
    return _c("1;33", t)


def cyan(t: str) -> str:
    return _c("0;36", t)


# A markdown inline-link target we should canonicalize: a bare local ``.md``
# filename (no path separator, no URL scheme, optional ``#anchor``), inside the
# ``](target)`` position, with an optional ``"title"`` suffix left untouched.
_LINK_RE = re.compile(r"\]\((?P<target>[^)\s]+)(?P<rest>[^)]*)\)")


def _canon_target(target: str) -> str:
    """Canonicalize one link target to the site filename convention.

    A bare local ``.md`` file (``FOO_BAR.md`` / ``FOO_BAR.md#anchor``) becomes
    ``foo-bar.md`` / ``foo-bar.md#anchor``. Anything with a ``/``, a URL scheme,
    or no ``.md`` file part is returned unchanged (relative paths, external URLs
    and same-page anchors are identical across both trees already).
    """
    file_part, sep, anchor = target.partition("#")
    if "/" in file_part or ":" in file_part or not file_part.endswith(".md"):
        return target
    canon = file_part.lower().replace("_", "-")
    return canon + sep + anchor


# A fenced-code-block delimiter: first non-space content is a run of 3+ backticks
# or tildes (optionally followed by an info string). ``](target)`` text inside a
# fence is a literal code example, not a real link, and must not be rewritten.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# An inline code span: a run of N backticks closed by a run of N backticks. Link
# syntax shown literally in a `code span` must not be rewritten either.
_CODE_SPAN_RE = re.compile(r"(`+).*?\1")


def canonicalize_line(line: str) -> str:
    """Rewrite intra-doc ``.md`` link targets to site form, outside code spans.

    Link syntax that appears inside an inline ``code span`` is literal example
    text: its character range is protected and left untouched, so only real
    links in the prose portion of the line are canonicalized.
    """
    protected = [(m.start(), m.end()) for m in _CODE_SPAN_RE.finditer(line)]

    def repl(m: re.Match[str]) -> str:
        if any(start <= m.start() < end for start, end in protected):
            return m.group(0)
        return f"]({_canon_target(m.group('target'))}{m.group('rest')})"

    return _LINK_RE.sub(repl, line)


def canonicalize_text(text: str) -> list[str]:
    """Canonicalize link targets line-by-line, skipping fenced code blocks.

    A fenced code block (opened and closed by a ``` or ~~~ run) holds literal
    example text; rewriting ``](target)`` inside it would corrupt code samples.
    Fence state is tracked across lines so fenced content passes through verbatim.
    """
    out: list[str] = []
    fence: str | None = None
    for ln in text.splitlines():
        m = _FENCE_RE.match(ln)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            out.append(ln)
            continue
        out.append(ln if fence is not None else canonicalize_line(ln))
    return out


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Result:
    """Accumulates errors and warnings for a run."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _exception_map(pair: dict) -> dict[str, str]:
    """Map canonicalized package line -> canonicalized site line for a pair."""
    out: dict[str, str] = {}
    for ex in pair.get("line_exceptions", []):
        out[canonicalize_line(ex["package"])] = canonicalize_line(ex["site"])
    return out


def _apply_line_exceptions(lines: list[str], exmap: dict[str, str]) -> tuple[list[str], list[str]]:
    """Apply canonicalized package->site line substitutions to ``lines``.

    ``line_exceptions`` match by exact (canonicalized) line *content*, not by
    position, so a substitution is only well-defined when its package line occurs
    exactly once in the source. If the same line text recurs, substituting every
    occurrence would silently rewrite unintended lines, so such a key is reported
    as ambiguous and left un-substituted rather than applied. Returns the
    substituted lines and the sorted list of ambiguous package keys.
    """
    if not exmap:
        return lines, []
    counts: dict[str, int] = {}
    for ln in lines:
        if ln in exmap:
            counts[ln] = counts.get(ln, 0) + 1
    ambiguous = sorted(key for key, count in counts.items() if count > 1)
    ambiguous_set = set(ambiguous)
    out = [exmap[ln] if ln in exmap and ln not in ambiguous_set else ln for ln in lines]
    return out, ambiguous


def check_mirror(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        res.fail(f"mirror: package source missing: {pkg_path.relative_to(ROOT)}")
        return
    if not site_path.exists():
        res.fail(f"mirror: site mirror missing: {site_path.relative_to(ROOT)}")
        return

    if site_path.is_symlink():
        res.fail(
            f"mirror: {site_path.relative_to(ROOT)} is classified as a hand-authored "
            f"mirror but is a symlink. Reclassify it as `symlink` in "
            f"{MANIFEST.relative_to(ROOT)}."
        )
        return

    exmap = _exception_map(pair)
    pkg_lines, ambiguous = _apply_line_exceptions(canonicalize_text(_read(pkg_path)), exmap)
    if ambiguous:
        rel_pkg = pkg_path.relative_to(ROOT)
        for key in ambiguous:
            res.fail(
                f"mirror: ambiguous line_exception for {rel_pkg}: the package line "
                f"{key!r} occurs more than once, so a content-matched exception would "
                f"rewrite every occurrence. Make the surrounding line unique, or drop "
                f"the exception, in {MANIFEST.relative_to(ROOT)}."
            )
        return
    site_lines = canonicalize_text(_read(site_path))

    if pkg_lines == site_lines:
        return

    rel_pkg = pkg_path.relative_to(ROOT)
    rel_site = site_path.relative_to(ROOT)
    diff = "\n".join(
        difflib.unified_diff(
            pkg_lines,
            site_lines,
            fromfile=f"{rel_pkg} (canonicalized)",
            tofile=f"{rel_site}",
            lineterm="",
        )
    )
    res.fail(
        f"mirror drift: {rel_pkg} <-> {rel_site}\n"
        f"  The site mirror must equal the package source modulo intra-doc\n"
        f"  link filenames and declared line_exceptions. Reconcile the two\n"
        f"  (or run `bin/docs-mirror-check.py --fix`), or, if the divergence\n"
        f"  is intentional, reclassify the pair in {MANIFEST.relative_to(ROOT)}.\n"
        + "\n".join("  " + ln for ln in diff.splitlines())
    )


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
        target = site_path.resolve()
        try:
            shown = target.relative_to(ROOT)
        except ValueError:
            shown = target
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
            f"is now a hand-authored copy, reclassify it as `mirror` or "
            f"`diverge` in {MANIFEST.relative_to(ROOT)}."
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
    is genuinely divergent, neither ``mirror`` nor whole-file ``transclude``
    can express it, which is how such a block ends up unclassified and
    unverified.

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
    for kind in ("symlink", "mirror", "transclude", "diverge"):
        for pair in entry.get(kind, []):
            paired.add(pair["site"])
            paired.add(pair["package"])

    for cls, names, own_dir, other_dir, other_side in (
        ("package_only", entry.get("package_only", []), pkg_dir, site_dir, "site"),
        ("site_only", entry.get("site_only", []), site_dir, pkg_dir, "package"),
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
                f"(symlink / transclude / mirror), or record the divergence with "
                f"`diverge` and a reason if the two are genuinely different documents."
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

    for kind in ("symlink", "mirror", "transclude", "diverge"):
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
            f"mirror / diverge / package_only)."
        )
    for name in sorted(site_on_disk - set(site_classified)):
        res.fail(
            f"unclassified site doc: {(site_dir / name).relative_to(ROOT)} — "
            f"add it to {MANIFEST.relative_to(ROOT)} (symlink / transclude / "
            f"mirror / diverge / site_only)."
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


def fix_mirror(pair: dict, pkg_dir: Path, site_dir: Path) -> bool:
    """Regenerate a mirror site file from its package source. Returns True if changed."""
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        return False
    exmap = _exception_map(pair)
    text = _read(pkg_path)
    lines, ambiguous = _apply_line_exceptions(canonicalize_text(text), exmap)
    if ambiguous:
        # An ambiguous exception cannot be applied safely — regenerating would
        # rewrite every occurrence of the recurring line. Leave the file as-is
        # for `--check` to report rather than silently corrupt it.
        return False
    regenerated = "\n".join(lines)
    if text.endswith("\n"):
        regenerated += "\n"
    if site_path.exists() and _read(site_path) == regenerated:
        return False
    site_path.write_text(regenerated, encoding="utf-8")
    return True


def run(manifest: dict, only: str | None, fix: bool) -> int:
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

        if fix:
            changed = []
            for pair in entry.get("mirror", []):
                if fix_mirror(pair, pkg_dir, site_dir):
                    changed.append(pair["site"])
            if changed:
                print(yellow(f"  regenerated {len(changed)} mirror page(s): " + ", ".join(changed)))
            else:
                print(green("  mirror pages already in sync"))
            continue

        res = Result()
        check_completeness(entry, pkg_dir, site_dir, res)
        check_unpaired(entry, pkg_dir, site_dir, res)
        for pair in entry.get("symlink", []):
            check_symlink(pair, pkg_dir, site_dir, res)
        for pair in entry.get("mirror", []):
            check_mirror(pair, pkg_dir, site_dir, res)
        for pair in entry.get("transclude", []):
            check_transclude(pair, pkg_dir, site_dir, res)
        for pair in entry.get("diverge", []):
            check_diverge(pair, pkg_dir, site_dir, res)

        # `shared_sections` is only meaningful where the two files legitimately
        # differ. On the other classes it would be silently ignored -- and a
        # declaration that is quietly ignored reads as a guarantee while
        # providing none, which is worse than not offering the key at all.
        for kind in ("symlink", "mirror", "transclude"):
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
                + len(entry.get("mirror", []))
                + len(entry.get("transclude", []))
                + len(entry.get("diverge", []))
            )
            print(green(f"  ✓ {n} classified pair(s) in sync; all docs classified"))
        else:
            for err in res.errors:
                print(red("  ✗ " + err.replace("\n", "\n    ")))
            overall.errors.extend(res.errors)

    if fix:
        return 0

    print()
    if overall.ok:
        print(green("✓ Documentation mirrors are in sync"))
        return 0
    print(red(f"✗ Documentation mirror check failed ({len(overall.errors)} issue(s))"))
    print(cyan("  Reconcile the mirror, run `bin/docs-mirror-check.py --fix`, or"))
    print(cyan(f"  reclassify the pair in {MANIFEST.relative_to(ROOT)}."))
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce the package<->site doc-mirror invariant.")
    parser.add_argument("--check", action="store_true", help="Check for drift (default).")
    parser.add_argument(
        "--fix", action="store_true", help="Regenerate mirror site files from source."
    )
    parser.add_argument("--package", metavar="NAME", help="Restrict to one manifest package.")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        print(red(f"✗ manifest not found: {MANIFEST}"), file=sys.stderr)
        return 2
    manifest = json.loads(_read(MANIFEST))
    return run(manifest, only=args.package, fix=args.fix)


if __name__ == "__main__":
    raise SystemExit(main())
