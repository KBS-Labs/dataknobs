#!/usr/bin/env python3
"""Doc-sync guard: enforce the package<->site doc-mirror invariant.

The dual-docs rule keeps two copies of most package docs:

  * package-local  ``packages/<pkg>/docs/*``   (GitHub, UPPER_SNAKE.md)
  * mkdocs site    ``docs/packages/<pkg>/*``    (site, lower-hyphen.md)

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
                  not content-checked; both files must exist.
  ``package_only``  Package doc with no site mirror.
  ``site_only``     Site-native page with no package source.

Completeness: every top-level ``*.md`` in both trees MUST be classified. An
unclassified file (or a manifest entry with no file on disk) fails the check
-- that is what makes silent drift impossible to introduce: a new doc forces
a classification decision at PR time. A paired entry may point at a package
*subdirectory* source (e.g. a transclusion of ``guides/events.md``); such a
source is not a top-level package doc, so it is exempt from the top-level
completeness set (its existence is still enforced by the per-class check).

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


def _apply_line_exceptions(
    lines: list[str], exmap: dict[str, str]
) -> tuple[list[str], list[str]]:
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
    out = [
        exmap[ln] if ln in exmap and ln not in ambiguous_set else ln
        for ln in lines
    ]
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
    pkg_lines, ambiguous = _apply_line_exceptions(
        canonicalize_text(_read(pkg_path)), exmap
    )
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


def check_diverge(pair: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    pkg_path = pkg_dir / pair["package"]
    site_path = site_dir / pair["site"]
    if not pkg_path.exists():
        res.fail(f"diverge: package source missing: {pkg_path.relative_to(ROOT)}")
    if not site_path.exists():
        res.fail(f"diverge: site page missing: {site_path.relative_to(ROOT)}")


def check_completeness(entry: dict, pkg_dir: Path, site_dir: Path, res: Result) -> None:
    """Every top-level *.md in both trees must be classified exactly once."""
    pkg_classified: dict[str, str] = {}
    site_classified: dict[str, str] = {}

    def _add(store: dict[str, str], name: str, bucket: str, side: str) -> None:
        if name in store:
            res.fail(
                f"manifest: {side} doc '{name}' classified twice "
                f"({store[name]} and {bucket})."
            )
        store[name] = bucket

    def _add_pkg(name: str, bucket: str) -> None:
        # A paired entry may source from a package subdirectory (e.g. a
        # transclusion of ``guides/events.md``). Such a source is not a
        # top-level package doc, so it is not part of the top-level
        # completeness set -- its existence is enforced by the per-class check
        # instead. Only top-level sources participate here.
        if "/" in name:
            return
        _add(pkg_classified, name, bucket, "package")

    for pair in entry.get("symlink", []):
        _add_pkg(pair["package"], "symlink")
        _add(site_classified, pair["site"], "symlink", "site")
    for pair in entry.get("mirror", []):
        _add_pkg(pair["package"], "mirror")
        _add(site_classified, pair["site"], "mirror", "site")
    for pair in entry.get("transclude", []):
        _add_pkg(pair["package"], "transclude")
        _add(site_classified, pair["site"], "transclude", "site")
    for pair in entry.get("diverge", []):
        _add_pkg(pair["package"], "diverge")
        _add(site_classified, pair["site"], "diverge", "site")
    for name in entry.get("package_only", []):
        _add(pkg_classified, name, "package_only", "package")
    for name in entry.get("site_only", []):
        _add(site_classified, name, "site_only", "site")

    pkg_on_disk = {p.name for p in pkg_dir.glob("*.md")}
    site_on_disk = {p.name for p in site_dir.glob("*.md")}

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
        for pair in entry.get("symlink", []):
            check_symlink(pair, pkg_dir, site_dir, res)
        for pair in entry.get("mirror", []):
            check_mirror(pair, pkg_dir, site_dir, res)
        for pair in entry.get("transclude", []):
            check_transclude(pair, pkg_dir, site_dir, res)
        for pair in entry.get("diverge", []):
            check_diverge(pair, pkg_dir, site_dir, res)

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
    parser.add_argument("--fix", action="store_true", help="Regenerate mirror site files from source.")
    parser.add_argument("--package", metavar="NAME", help="Restrict to one manifest package.")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        print(red(f"✗ manifest not found: {MANIFEST}"), file=sys.stderr)
        return 2
    manifest = json.loads(_read(MANIFEST))
    return run(manifest, only=args.package, fix=args.fix)


if __name__ == "__main__":
    raise SystemExit(main())
