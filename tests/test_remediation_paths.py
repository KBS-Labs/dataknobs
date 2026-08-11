"""Reproduce-first guard: the command the harness tells you to run must work.

Every check here failed when it was written, and one of them failed in the worst
available way. ``bin/validate.sh`` prints ``Run: ./bin/fix.sh`` on every failed
validation. ``bin/fix.sh`` invoked bare ``ruff`` — which is not on ``PATH`` in
this workspace, because the toolchain is pinned behind ``uv run``. So the single
command the quality gate recommends printed *"⚠ Some issues remain that need
manual fixing"* — indistinguishable from ruff having run and reached its limit —
and then exited 1 without having fixed, formatted, or examined anything.

A wrong verdict is caught eventually by the code disagreeing with it. Wrong
*advice* is not: it is read by a person, once, at the moment they are already
dealing with a failure, and its cost is their next half hour. Nothing in the
gate reads these strings, so nothing else can notice.

Four properties, each an instance of the same sentence:

1. **Pinned toolchain.** A bare ``ruff``/``mypy`` resolves against ``PATH``, so
   it is either absent (the failure above) or a different version than the one
   that produced the verdict being remediated.
2. **Advice names an entry point, not a hand-written invocation.** A printed
   ``uv run ruff check packages/*/src`` is a fourth answer to *which code do we
   check* that no one will remember to update. It also dropped the ``--config``
   the gate passes, which used to resolve the per-package ``[tool.ruff]``
   sections and report findings the gate did not have; those sections are gone,
   so that half is closed, but the target set it names is still a fourth answer
   and still the reason this property holds.
3. **A printed path exists.** Advice pointing at a deleted script is a dead end
   discovered by the reader.
4. **The formatter runs where it is checked.** ``ruff format`` used to be
   enforced nowhere, so the property here was containment: one opt-in command
   could run it and ``fix`` could not, because a user asking to fix findings
   would have received a repo-wide diff instead. Now that ``validate.sh``
   checks formatting, the fault inverts — a command that formats *less* than
   the check covers reports success and leaves the gate red, which is this
   file's first paragraph with a different tool in it. Three scripts may run
   it; a fourth is a target set nobody keeps in step.
"""

from __future__ import annotations

import re

from tests._workspace import ROOT, tracked_shell_files
from tests.test_linter_invocation_resolution import (
    BARE_RE,
    PINNED_RE,
    TOOLS,
    logical_lines,
)

#: The scripts allowed to run the formatter: the check, the command that
#: repairs what the check reports, and the opt-in named after it. Anything else
#: is a fourth answer to which code gets formatted.
FORMATTER_OWNERS = frozenset({"bin/validate.sh", "bin/fix.sh", "bin/dk"})

#: A command being printed rather than run.
ADVICE_RE = re.compile(r"^\s*(?:echo|printf)\b")

#: A repo path named inside printed text: ``./bin/x.sh``, ``bin/x.sh``. A bare
#: ``dk`` is not matched — it is a command name, not a path.
#:
#: The lookbehind is load-bearing: without it ``venv/bin/activate`` matches from
#: ``bin/`` onward and reports a path the repo never claimed to have.
ADVISED_PATH_RE = re.compile(r"(?<![\w/.])(?:\./)?((?:bin|packages|docs|tests)/[\w./*-]+)")

#: Sentence punctuation swept up by the path pattern's trailing character class.
#: These scripts write prose, so a path can end a sentence.
_TRAILING_PUNCTUATION = ".,;:)"

#: ``ruff format`` in either direction — the rewrite and the ``--check``.
#: Both are the same claim about whether formatting is enforced here.
RUFF_FORMAT_RE = re.compile(r"\bruff format\b")


def _statements() -> list[tuple[str, int, str]]:
    """``(path, line, statement)`` for every tracked shell script."""
    out = []
    for name in tracked_shell_files():
        text = (ROOT / name).read_text(encoding="utf-8")
        out += [(name, line, stmt) for line, stmt in logical_lines(text)]
    assert out, "no tracked shell statements found — has the enumeration broken?"
    return out


def _is_comment(statement: str) -> bool:
    return statement.lstrip().startswith("#")


def test_every_linter_runs_through_the_pinned_toolchain():
    """A bare ``ruff`` is whatever is on ``PATH``, which here is nothing at all.

    The failure mode is not "an older ruff": it is ``command not found`` inside
    an ``if``, which takes the else branch and prints the message written for
    "ruff ran and left some findings".
    """
    violations = []
    for path, line, statement in _statements():
        if _is_comment(statement) or ADVICE_RE.match(statement):
            continue
        pinned = {m.start(1) for m in PINNED_RE.finditer(statement)}
        violations += [
            f"{path}:{line}: bare {m.group(1)} — resolves against PATH, not the workspace"
            for m in BARE_RE.finditer(statement)
            if m.start(1) not in pinned
        ]

    assert not violations, (
        "Linters invoked outside the pinned environment:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\nPrefix with 'uv run'. The workspace pins the version that produced "
        "the verdict being remediated; PATH does not."
    )


def test_printed_advice_names_an_entry_point_not_a_raw_invocation():
    """A hand-written command in a string is a target set nothing keeps current.

    ``bin/diagnose-quality-failures.sh`` printed five of these. Each named
    ``packages/*/src`` — so none could reproduce a ``bin/`` finding, which the
    same script had just listed — and none passed ``--config``, so each resolved
    a different ruff configuration than the one that found it.
    """
    violations = []
    for path, line, statement in _statements():
        if _is_comment(statement) or not ADVICE_RE.match(statement):
            continue
        pinned = [m.group(1) for m in PINNED_RE.finditer(statement)]
        bare = [
            m.group(1)
            for m in BARE_RE.finditer(statement)
            if m.start(1) not in {p.start(1) for p in PINNED_RE.finditer(statement)}
        ]
        for tool in pinned + bare:
            violations.append(f"{path}:{line}: prints a raw {tool} invocation as advice")

    assert not violations, (
        "Printed remediation commands that re-specify a linter run:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\nName the entry point instead (./bin/fix.sh, ./bin/validate.sh, dk "
        "check). Those resolve their own targets and config, so the advice "
        "cannot drift from what the gate actually does."
    )


def test_every_path_named_in_advice_exists():
    """Advice pointing at a script that is not there is found by the reader.

    Globs are skipped: ``packages/*/src`` is a pattern, not a path, and the
    test above is what keeps those out of advice in the first place.
    """
    violations = []
    for path, line, statement in _statements():
        if _is_comment(statement) or not ADVICE_RE.match(statement):
            continue
        for match in ADVISED_PATH_RE.findall(statement):
            named = match.rstrip(_TRAILING_PUNCTUATION)
            if "*" in named or "$" in named:
                continue
            if not (ROOT / named).exists():
                violations.append(f"{path}:{line}: names {named}, which does not exist")

    assert not violations, "Printed advice naming paths that do not exist:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_the_formatter_runs_only_where_it_is_checked_or_asked_for():
    """Every script running the formatter is one of the three that should.

    The previous form of this pinned the formatter to a single opt-in command,
    on the grounds that nothing validated its output — a reformat was a diff
    nobody requested and nobody would notice being reverted, which is fine
    behind a command called ``format`` and not fine inside ``fix``, where it
    was, reachable from the tip the gate prints.

    Adopting the formatter inverts that. There is now a check, so the risk moves
    from *an unrequested rewrite* to *the fix not reaching what the check
    flags*: an entry point that formats less than ``validate.sh`` checks reports
    success and leaves the gate red, which is this file's opening paragraph with
    a different tool in it. So the three sanctioned owners are the check, its
    write side, and the named opt-in — and a fourth script running the formatter
    is a fourth answer to *which code do we format*, the same fault property 2
    rejects for target sets.

    What this cannot see is whether the three agree on their populations.
    ``test_toolchain_consistency`` holds that from the other side — and did not
    when this sentence was first written. The check ran over the *linter's*
    target set, which omits every cell whose ruff tier is deferred, so it read
    597 of 1,471 files and passed over the rest; the write side could not reach
    42 of them at all. Both halves are needed and neither implies the other: a
    fourth owner is a target set nobody keeps in step, and three owners reading
    three different sets is the same fault without the fourth script.
    """
    owners = sorted(
        {
            path
            for path, _line, statement in _statements()
            if not _is_comment(statement) and RUFF_FORMAT_RE.search(statement)
        }
    )

    assert owners == sorted(FORMATTER_OWNERS), (
        f"Scripts running ruff format: {owners or 'none'} — expected exactly "
        f"{sorted(FORMATTER_OWNERS)}.\nThe formatter belongs in the check, in "
        "the command that repairs what the check reports, and in the opt-in "
        "named after it. A fourth caller is a fourth target set nobody keeps in "
        "step; a missing one is a check with no remedy."
    )


def test_the_tool_list_is_the_one_the_resolution_guard_uses():
    """Both files ask about the same tools; only one of them should say which.

    A second list here would drift, and the drift would be silent in the
    direction that matters — a tool added there and missing here is a linter
    whose remediation path nothing checks.
    """
    assert set(TOOLS) >= {"ruff", "mypy", "pylint"}, (
        f"the shared tool list has shrunk to {TOOLS}; the linters the gate "
        "actually runs must stay in it"
    )
