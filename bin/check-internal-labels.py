#!/usr/bin/env python3
"""Guard against internal-tracking-label leakage into shipped source/tests.

Planning identifiers (``Item NNN`` / ``item NNa``, ``RCN``, ``Change C``,
``Bug BNN``, ``PR #NNN``, ``review #XN``, ``consumer-gaps``, plan-phase
tags (``Phase N`` without a trailing ``:``), plan sub-item ids (``77a`` /
``92b`` / ``146b``), ``decision N``, plan-document section refs like
``02b §5.2`` / ``02b P5a``, bare-number tracker references like
``pre-141`` / ``pre-146b`` / ``the 141 failure`` / ``as 141``, etc.) must
never appear in committed package source or tests: they render into
published API docs and IDE hovers and mean nothing to consumers.
A prior one-time cleanup scrubbed the pre-existing leakage; this script
is the recurring guard that prevents reintroduction.

Scope: ``packages/*/src`` and ``packages/*/tests``, plus the first-party code
belonging to no package -- ``bin/``, root ``tests/``, the workspace shim and
the root conftest -- and every tracked shell script.

That second half was outside the scan until it was measured.  The docstring
justified the narrow scope by where labels do damage: rendered API docs and IDE
hovers, which is shipped code.  True, and not the only reason -- a tracker
label is a reference to a document the reader cannot open, and in ``bin/`` the
reader is a maintainer rather than a consumer.  ``bin/`` is also where the
guards that check this repository live, and where the longest explanatory prose
gets written, so it was simultaneously the likeliest place for one to land and
the one place nothing looked.  Widening it cost nothing: measured across all of
it, the only hits were in this file.

Both halves are asked for rather than listed -- ``package-discovery.sh
workspace-targets`` for the Python, ``lint-shell.sh --print-targets`` for the
shell -- because a fourth hand-kept copy of "which code is ours" is how the
first three came to disagree.  Neither is allowed to fail quietly: a scope that
silently narrows reports a clean scan over code it never read.

Data files are deliberately out.  Measured, they contribute no true positives
and eleven false ones, because the digit-suffix branches below are tuned for
English prose about code: ``70b`` is a model's parameter count, ``447f`` a hash
fragment, ``18a`` a Unicode codepoint.  The line is authored prose, not file
count.

``Phase N`` is enforced only in its *leak* form -- a planning-phase tag
NOT immediately followed by ``:``.  The legitimate runtime-pipeline-stage
usage always takes the colon form (``# Phase 2: Deterministic retrieval``)
and is left alone, so authors documenting a real pipeline stage must keep
the colon.  Plan sub-item ids are matched only as ``[0-9]{2,3}[a-g]`` so
unit-bearing tokens (``200k``, ``30s``) and printf format specs
(``{i:03d}``) do not trip the guard.

False positives (fixture record values, markdown list-item test content)
are suppressed via ``bin/internal-label-allowlist.txt``, keyed by
(repo-relative path, exact substring) rather than by line number, because
line numbers drift under any edit above them.  Content is *more* stable
than a line number but is not stable: adopting the formatter broke two
entries at once, one on quote style and one that had only ever worked
because two tokens shared a line.  See that file's header for what a
durable substring looks like.

An entry matching nothing is therefore an error, not a shrug -- on a
full-scope run, which is the only run where every entry is reachable.
A suppression whose target moved or was reworded goes on suppressing
nothing, and nothing about the report distinguishes that from a clean
scan; the two sibling guards in this repository
(``assert_no_ad_hoc_dotted_import``, ``assert_no_broad_except_in_error_text``)
both take the same position.  Extending the allowlist requires a
reviewer-visible diff to that file with a stated reason.

Exit status: 0 when clean (modulo allowlist), 1 when any non-allowlisted
label is found.  No autofix -- rewording requires human judgement.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_FILE = Path(__file__).resolve().parent / "internal-label-allowlist.txt"

# Default scan scope for package code (repo-relative glob roots).  The code
# belonging to no package is asked for rather than listed -- see _extra_roots.
DEFAULT_GLOBS = ("packages/*/src", "packages/*/tests")

#: Files exempt from their own check, because describing a label requires
#: writing one.  This module quotes fourteen across its docstring and its
#: pattern comments, and the allowlist file is a table of them; keyed by exact
#: substring, allowlisting them individually would mean editing that table
#: every time the pattern gains a branch, which is the shape of upkeep nobody
#: does.  Narrow on purpose: a genuine label written into this file is not
#: caught, and there is no way to have both without a marker syntax that would
#: itself need explaining.  The allowlist entry is dead under the current
#: suffix rule -- a ``.txt`` is not scanned -- and is kept so that widening the
#: scope to data files does not silently make this file fail its own check.
SELF_DESCRIBING = frozenset(
    {
        "bin/check-internal-labels.py",
        "bin/internal-label-allowlist.txt",
    }
)


def _declared(command: list[str], what: str) -> list[str]:
    """Run a declaration-printing helper and split its output into names.

    ``check=True``, and a failure is not caught.  A scope that comes back empty
    because the helper broke is indistinguishable in the report from a scope
    with nothing in it, and this check announces success by printing a tick --
    so degrading to a partial scan would print that tick over unread code.
    """
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    names = result.stdout.split()
    if not names:
        msg = f"{what} named nothing: {' '.join(command)}"
        raise RuntimeError(msg)
    return names


def _extra_roots() -> list[Path]:
    """The first-party code belonging to no package, plus every shell script."""
    workspace = _declared(
        [str(ROOT / "bin" / "package-discovery.sh"), "workspace-targets"],
        "workspace targets",
    )
    shell = _declared(
        [str(ROOT / "bin" / "lint-shell.sh"), "--print-targets"],
        "shell lint targets",
    )
    return [ROOT / name for name in (*workspace, *shell)]


# Unambiguous tracker-label classes only.
LABEL_PATTERN = re.compile(
    # ``[ -]`` rather than a space: the hyphenated spelling is what an author
    # reaches for mid-sentence (``the post-Item-116 contract``), and matching
    # only the space form left seven of these in the scope this guard already
    # covered -- one of them in shipped package source, which is the exact
    # thing it exists to keep out of rendered API docs. The separator is the
    # one degree of freedom an author has here, so both spellings are the
    # class rather than two classes.
    r"Item[ -][0-9]{1,3}"
    r"|Items [0-9]+\+[0-9]+"
    r"|consumer-gaps"
    r"|\bRC[0-9]+\b"
    r"|pre-Item"
    r"|post-Item[ -][0-9]"
    # ``Phase N`` is enforced only in its *leak* form: a planning-phase
    # tag NOT immediately followed by ``:``.  Legitimate runtime-pipeline
    # -stage usage always takes the colon form (``# Phase 2: Deterministic
    # retrieval``) and is left alone -- the colon is the convention that
    # separates a stage label from a plan reference.  The ``\b`` after the
    # digits stops ``[0-9]+`` backtracking to a partial match inside a
    # legitimate ``Phase 10:``.
    r"|Phase [0-9]+\b(?!\s*:)"
    r"|\bChange C\b"
    r"|\bBug B[0-9]+\b"
    r"|PR #[0-9]{2,4}"
    r"|review #X?[0-9]"
    # Lowercase ``item NN`` plan references (the capital ``Item N`` branch
    # above handles the other casing).  Restricted to 2-3 digits so the
    # common-English ``item 1`` / ``item 2`` (list-position prose in
    # markdown/error fixtures) does not trip; the optional ``[a-h]``
    # suffix catches sub-item ids like ``item 77a``.
    r"|\bitem [0-9]{2,3}[a-h]?\b"
    # Plan-document section refs (e.g. ``02b §5.2``, ``02b P5a``).
    # Unambiguous: the ``§``/``P<digit>`` suffix never collides with
    # format specs like ``{i:03d}`` (no space + section marker).
    r"|\b0[0-9][a-z] (?:§|P[0-9])"
    # Plan sub-item ids: 2-3 digits + a single ``a``-``g`` suffix
    # (``77a``, ``92b``, ``146b``, ``18a``).  The ``[a-g]`` ceiling plus
    # the ``(?<![:>%])`` lookbehind keep this off unit-bearing tokens
    # (``200k``, ``30s``, ``100x``), printf/format specs (``{i:03d}``,
    # ``{i:>08b}``) and percent-escapes (``sv%40c`` is a URL-encoded
    # ``sv@c``, not sub-item ``40c``); UUID/hex segments (``e29b``,
    # ``cafef00d``) have no leading word boundary so never match.
    #
    # ``%`` earns its place in that set the same way ``:`` and ``>`` did:
    # immediately before digits it means they belong to an encoding
    # rather than to an identifier.  ``%40`` is the escape for ``@``, the
    # one character a userinfo field nearly always has to encode, so the
    # collision fires on any encoded username whose next character is
    # ``a``-``g`` -- a standing class wherever DSNs are written encoded,
    # not one unlucky fixture.  Narrowing here rather than allowlisting
    # each hit: an allowlist entry is keyed to one (path, substring) pair
    # and would be spent again on the next one.
    r"|(?<![:>%])\b[0-9]{2,3}[a-g]\b"
    # Plan ``decision N`` references.
    r"|\bdecision [0-9]{1,3}\b"
    # Bare-number tracker references that slipped past the ``Item NN``
    # form: ``pre-141`` / ``post-141`` (hyphenated qualifier, optional
    # sub-item letter as in ``pre-146b``), ``the 141 failure`` / ``the
    # 141 drift`` (definite article + tracker noun), and the trailing
    # form ``... drift mode as 141`` (tracker noun + ``as``/``like`` +
    # number).  The tracker-noun set is closed
    # (failure|drift|ctor|call|case|fix|gap|item|mode|issue|bug) to avoid
    # false-positives like ``reports a missing bucket as 404`` (HTTP
    # status), ``the 200 response``, or ``the 30-second timeout``.
    r"|\b(?:pre|post)-[0-9]{2,3}[a-z]?\b"
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
                f"WARNING: malformed allowlist line (need path<TAB>substring<TAB>reason): {raw!r}",
                file=sys.stderr,
            )
            continue
        entries.append((parts[0].strip(), parts[1]))
    return entries


def iter_target_files(args: list[str]) -> list[Path]:
    """Resolve CLI args (or the default scope) to a sorted list of files.

    A named file is scanned whatever its suffix -- naming it is the statement
    that it should be -- while a directory contributes its ``*.py`` only.  The
    shell half of the default scope arrives as individual paths from
    ``lint-shell.sh``, so it needs no suffix rule here; extending a *directory*
    walk to shell would mean a fourth copy of the suffix-or-shebang question
    that three files in this repository already answer differently on purpose.
    """
    files: set[Path] = set()
    if args:
        for arg in args:
            p = Path(arg)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_file():
                files.add(p.resolve())
            elif p.is_dir():
                files.update(f.resolve() for f in p.rglob("*.py"))
    else:
        roots = [
            root_dir for glob in DEFAULT_GLOBS for root_dir in ROOT.glob(glob) if root_dir.is_dir()
        ]
        for root_dir in roots:
            files.update(f.resolve() for f in root_dir.rglob("*.py"))
        for extra in _extra_roots():
            if extra.is_dir():
                files.update(f.resolve() for f in extra.rglob("*.py"))
            elif extra.is_file():
                files.add(extra.resolve())
    return sorted(files)


def matching_entry(
    rel_path: str, line: str, allowlist: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Return the entry suppressing this hit, or ``None``.

    A hit is suppressed iff its file matches an allowlist path AND the
    offending line contains that entry's exact substring.  The matched entry
    is returned rather than a bool so the caller can tell which suppressions
    are load-bearing and which have gone dead.
    """
    for allow_path, substring in allowlist:
        if rel_path == allow_path and substring in line:
            return (allow_path, substring)
    return None


def package_of(rel_path: str) -> str:
    """Return ``packages/<pkg>`` for grouping, else the parent dir."""
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[0] == "packages":
        return f"packages/{parts[1]}"
    return str(Path(rel_path).parent)


def report_dead_entries(allowlist: list[tuple[str, str]], used: set[tuple[str, str]]) -> int:
    """Print any allowlist entry that suppressed nothing, and return 1 if any did.

    Only meaningful after a full-scope run -- on a targeted invocation every
    entry outside the named paths is trivially unused.
    """
    dead = [entry for entry in allowlist if entry not in used]
    if not dead:
        return 0
    print(
        "    ✗ Allowlist entries that suppressed nothing. Each one is a "
        "suppression over code that no longer says what it did, which reads "
        "exactly like a clean scan:"
    )
    for allow_path, substring in dead:
        print(f"        {allow_path}\t{substring!r}")
    print(
        f"\n    {len(dead)} dead entr(ies). Re-read the target line: the text "
        f"was reworded or reformatted (quote style counts), the hit moved to "
        f"another line, or the suppression is no longer needed and the entry "
        f"should be deleted."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """Scan, report, and return the exit status.

    ``argv`` is the target list, defaulting to the real one.  Taken as a
    parameter because the dead-entry check below runs only on a full-scope
    run, and a test that cannot ask for one cannot reach it.
    """
    args = sys.argv[1:] if argv is None else argv
    allowlist = load_allowlist()
    findings: list[tuple[str, int, str, str]] = []  # pkg, lineno, label, rel
    full_scope = not args
    used: set[tuple[str, str]] = set()

    for path in iter_target_files(args):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        try:
            rel_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        if rel_path in SELF_DESCRIBING:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = LABEL_PATTERN.search(line)
            if not match:
                continue
            entry = matching_entry(rel_path, line, allowlist)
            if entry is not None:
                used.add(entry)
                continue
            findings.append((package_of(rel_path), lineno, match.group(0), rel_path))

    if findings:
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

    # Reported even when there are findings: the two are independent faults,
    # and a run that stopped at the first would hide dead suppressions behind
    # every unrelated leak until the last one was reworded.
    dead = report_dead_entries(allowlist, used) if full_scope else 0

    if not findings and not dead:
        print("    ✓ No internal-tracking-label leakage found")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
