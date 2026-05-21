#!/usr/bin/env python3
"""Guard against internal-tracking-label leakage into shipped source/tests.

Planning identifiers (``Item NNN``, ``RCN``, ``Change C``, ``Bug BNN``,
``PR #NNN``, ``review #XN``, ``consumer-gaps``, plan ``Phase 0``,
plan-document section refs like ``02b §5.2`` / ``02b P5a``, bare-number
tracker references like ``pre-141`` / ``the 141 failure`` / ``as 141``,
etc.) must never appear in committed package source or tests: they render
into published API docs and IDE hovers and mean nothing to consumers.
A prior one-time cleanup scrubbed the pre-existing leakage; this script
is the recurring guard that prevents reintroduction.

Scope: ``packages/*/src/**/*.py`` and ``packages/*/tests/**/*.py``.

``Phase N`` for N >= 1 is intentionally NOT enforced: it has a
high-frequency *legitimate* meaning (runtime pipeline stage, e.g.
``# Phase 2: Deterministic retrieval``) that cannot be mechanically
separated from the plan-reference leak meaning.  ``Phase 0`` is enforced
because it only ever appears as the tracker label.

False positives (fixture record values, markdown list-item test content)
are suppressed via ``bin/internal-label-allowlist.txt``, keyed by
(repo-relative path, exact substring) -- never by line number, because
line numbers drift while content is stable.  Extending the allowlist
requires a reviewer-visible diff to that file with a stated reason.

Exit status: 0 when clean (modulo allowlist), 1 when any non-allowlisted
label is found.  No autofix -- rewording requires human judgement.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_FILE = Path(__file__).resolve().parent / "internal-label-allowlist.txt"

# Default scan scope (repo-relative glob roots) when no paths are passed.
DEFAULT_GLOBS = ("packages/*/src", "packages/*/tests")

# Unambiguous tracker-label classes only.  ``Phase \d`` (>=1) is
# deliberately absent (see module docstring); ``Phase 0`` is kept.
LABEL_PATTERN = re.compile(
    r"Item [0-9]{1,3}"
    r"|Items [0-9]+\+[0-9]+"
    r"|consumer-gaps"
    r"|\bRC[0-9]+\b"
    r"|pre-Item"
    r"|post-Item [0-9]"
    r"|Phase 0"
    r"|\bChange C\b"
    r"|\bBug B[0-9]+\b"
    r"|PR #[0-9]{2,4}"
    r"|review #X?[0-9]"
    # Plan-document section refs (e.g. ``02b §5.2``, ``02b P5a``).
    # Unambiguous: the ``§``/``P<digit>`` suffix never collides with
    # format specs like ``{i:03d}`` (no space + section marker).
    r"|\b0[0-9][a-z] (?:§|P[0-9])"
    # Bare-number tracker references that slipped past the ``Item NN``
    # form: ``pre-141`` / ``post-141`` (hyphenated qualifier),
    # ``the 141 failure`` / ``the 141 drift`` (definite article + tracker
    # noun), and the trailing form ``... drift mode as 141`` (tracker
    # noun + ``as``/``like`` + number).  The tracker-noun set is closed
    # (failure|drift|ctor|call|case|fix|gap|item|mode|issue|bug) to avoid
    # false-positives like ``reports a missing bucket as 404`` (HTTP
    # status), ``the 200 response``, or ``the 30-second timeout``.
    r"|\b(?:pre|post)-[0-9]{2,3}\b"
    r"|\b[Tt]he [0-9]{2,3} (?:failure|drift|ctor|call|case|fix|gap|item|mode|issue|bug)\b"
    r"|\b(?:failure|drift|ctor|call|case|fix|gap|item|mode|issue|bug)s? (?:as|like) [0-9]{2,3}\b"
    r"|\bPre-[0-9]{2,3} (?:call|ctor)\b"
)


def load_allowlist() -> list[tuple[str, str]]:
    """Return (relative_path, exact_substring) suppression pairs.

    File format: ``path<TAB>exact-substring<TAB>reason``.  Blank lines and
    lines beginning with ``#`` are ignored.  The reason column is for
    human reviewers only.
    """
    entries: list[tuple[str, str]] = []
    if not ALLOWLIST_FILE.exists():
        return entries
    for raw in ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            print(
                f"WARNING: malformed allowlist line (need path<TAB>substring"
                f"<TAB>reason): {raw!r}",
                file=sys.stderr,
            )
            continue
        entries.append((parts[0].strip(), parts[1]))
    return entries


def iter_target_files(args: list[str]) -> list[Path]:
    """Resolve CLI args (or the default scope) to a sorted list of .py files."""
    files: set[Path] = set()
    if args:
        for arg in args:
            p = Path(arg)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_file() and p.suffix == ".py":
                files.add(p.resolve())
            elif p.is_dir():
                files.update(f.resolve() for f in p.rglob("*.py"))
    else:
        for glob in DEFAULT_GLOBS:
            for root_dir in ROOT.glob(glob):
                if root_dir.is_dir():
                    files.update(f.resolve() for f in root_dir.rglob("*.py"))
    return sorted(files)


def is_allowlisted(
    rel_path: str, line: str, allowlist: list[tuple[str, str]]
) -> bool:
    """A hit is suppressed iff its file matches an allowlist path AND the
    offending line contains that entry's exact substring."""
    for allow_path, substring in allowlist:
        if rel_path == allow_path and substring in line:
            return True
    return False


def package_of(rel_path: str) -> str:
    """Return ``packages/<pkg>`` for grouping, else the parent dir."""
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[0] == "packages":
        return f"packages/{parts[1]}"
    return str(Path(rel_path).parent)


def main() -> int:
    allowlist = load_allowlist()
    findings: list[tuple[str, int, str, str]] = []  # pkg, lineno, label, rel

    for path in iter_target_files(sys.argv[1:]):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        try:
            rel_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = LABEL_PATTERN.search(line)
            if not match:
                continue
            if is_allowlisted(rel_path, line, allowlist):
                continue
            findings.append(
                (package_of(rel_path), lineno, match.group(0), rel_path)
            )

    if not findings:
        print("    ✓ No internal-tracking-label leakage found")
        return 0

    findings.sort(key=lambda f: (f[0], f[3], f[1]))
    print(
        "    ✗ Found internal-tracking-label leakage "
        "(reword to drop the planning reference; preserve technical intent):"
    )
    current_pkg = ""
    for pkg, lineno, label, rel_path in findings:
        if pkg != current_pkg:
            current_pkg = pkg
            print(f"      - {pkg}:")
        print(f"        {rel_path}:{lineno}: {label!r}")
    print(
        f"\n    {len(findings)} occurrence(s). If a hit is a genuine "
        f"fixture/data value (not a tracker label), add a reviewed entry "
        f"to bin/internal-label-allowlist.txt."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
