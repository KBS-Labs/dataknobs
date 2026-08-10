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
4. **The formatter is contained.** ``ruff format`` is not enforced anywhere:
   no gate check runs it, so its output is unvalidated by construction.
   ``bin/fix.sh`` ran it over every target — a 422-file rewrite that nothing
   would have noticed being reverted — while being the command the gate prints.
   One explicitly-named opt-in command may run it; a command named ``fix`` may
   not, because the user asked to fix findings and would receive a repo-wide
   diff instead.
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

#: The one command allowed to run the formatter, and where it lives.
FORMATTER_OWNER = "bin/dk"

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


def test_only_one_entry_point_runs_the_formatter():
    """Formatting is not enforced here, so running it must be asked for by name.

    No gate check runs ``ruff format`` in any form, which makes its output
    unvalidated by construction: a reformat is a diff nothing requested and
    nothing would notice being reverted. That is tolerable behind a command
    called ``format``, where it is what the user asked for. It is not tolerable
    inside ``fix``, and it was there — reachable from the tip the gate prints.

    If formatting is ever adopted, this test is the place that says so: add the
    check to validation first, then widen this pin.
    """
    owners = sorted(
        {
            path
            for path, _line, statement in _statements()
            if not _is_comment(statement) and RUFF_FORMAT_RE.search(statement)
        }
    )

    assert owners == [FORMATTER_OWNER], (
        f"Scripts running ruff format: {owners or 'none'} — expected exactly "
        f"[{FORMATTER_OWNER!r}].\nNothing validates formatting, so every other "
        "caller produces an unrequested diff. Remove it, or adopt formatting as "
        "a validated check and update this pin."
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
