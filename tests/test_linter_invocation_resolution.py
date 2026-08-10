"""Reproduce-first guard: linters resolve three things against the working directory.

``ruff`` and ``pylint`` both walk up from the working directory when no config is
passed, and two further resolutions follow the same directory:

1. **The config file.** Fixed by an absolute ``--config`` / ``--rcfile``.
2. **``per-file-ignores`` keys**, which are repo-relative paths matched against
   the working directory. From anywhere else they match nothing and every
   suppression silently lifts, so the tool reports findings the gate does not
   have. Only a working directory of the root fixes this — an absolute target
   path does *not*.
3. **Target globs**, expanded by the shell at the call site. An unmatched glob
   is passed through literally, and ``ruff`` reports it as a single ``E902``
   io-error *finding* — valid JSON, exit 1, indistinguishable at a glance from a
   real result. The whole check silently did not run.

The original defect was in ``bin/dk``, and this file covered ``bin/dk`` alone on
the strength of a docstring claim that *"every other script in bin/ sidesteps
this by cd-ing to the repo root first"*. That was true when written and false
by the time anyone checked: ``bin/run-quality-checks.sh`` never ``cd``s, and it
owns the ruff invocation that writes the committed ``style-check.json``. A guard
scoped to the one file that had the defect is the shape that lets the identical
defect live one directory over — so the claim is now asserted over every script
that runs a linter rather than restated in prose.

**Two ways to be correct, and they are not interchangeable.** A script that
``cd``s to the root at the top fixes all three resolutions at once, because the
glob is expanded after the ``cd``. ``bin/dk`` cannot do that — it is explicitly
allowed to run from a subdirectory — so it dispatches through a seam that
``cd``s in a subshell, which fixes (1) and (2) but *not* (3): the caller's shell
already expanded the glob. That is why ``bin/dk`` carries the extra requirement
that its targets be ``$PROJECT_ROOT``-anchored, and other scripts do not.

Asserted per invocation rather than per file, because the original defect was
two lines in one function disagreeing about which mechanism they used.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from tests._workspace import ROOT, rel, tracked_shell_files

DK = ROOT / "bin" / "dk"

#: The linters that resolve config by walking up from the working directory.
TOOLS = ("ruff", "pylint", "mypy", "black")
_TOOL_ALT = "|".join(TOOLS)

#: ``uv run <tool>`` anywhere in a statement. Unambiguous, so it needs no
#: command-position anchor — and it is the form the workspace pins.
PINNED_RE = re.compile(rf"\buv run ({_TOOL_ALT})\b")

#: A bare ``<tool>`` in command position: statement start, or straight after an
#: operator that begins one. Anchored, because these names occur as ordinary
#: words in argument lists (``uv pip install mypy ruff``) and in prose.
BARE_RE = re.compile(rf"(?:^|\bif |\belif |&&|\|\||;|\$\()\s*({_TOOL_ALT})\b")

#: How each names its config file. A tool absent here takes no config flag.
CONFIG_FLAGS = {
    "ruff": "--config",
    "pylint": "--rcfile",
    "mypy": "--config-file",
    "black": "--config",
}

#: The single seam in ``bin/dk`` that runs a command with the root as cwd.
ROOT_RUNNER = "run_from_root"

#: A top-level ``cd`` to the repo root. Both spellings are in use and both mean
#: the same directory. Unindented on purpose: ``run-quality-checks.sh`` carries
#: three *indented* ``cd "$PROJECT_ROOT"`` lines that return from a temporary
#: ``cd`` elsewhere, and those are evidence the script believes it starts at the
#: root — not evidence that it does.
CD_TO_ROOT_RE = re.compile(r'^cd "\$(?:ROOT_DIR|PROJECT_ROOT)"\s*$', re.MULTILINE)

#: A target naming every package. Located per occurrence, because one command
#: can name several targets and anchor only some of them.
PACKAGES_GLOB_RE = re.compile(r"packages/\*")


class Invocation(NamedTuple):
    """One linter call: where it is, which tool, and the whole statement."""

    path: str
    line: int
    tool: str
    cmd: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}"


def logical_lines(text: str) -> list[tuple[int, str]]:
    r"""``(line number of the statement head, statement with continuations joined)``.

    A shell statement is not a physical line, and reading it as one is its own
    defect: this file used to match a single line and read the config flag from
    it, so splitting a long ``uv run ruff …`` across a ``\\`` reported the flag
    *absent* when it was present one line down. The first editor to hit that
    reverted the split and left a comment telling the next one not to wrap —
    a guard asserting the opposite of the truth, and a workaround standing in
    for a fix. Joining here is the fix; every predicate below reads a statement.
    """
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        head = i
        parts = [lines[i]]
        while parts[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            parts.append(lines[i])
        joined = " ".join(part.rstrip().rstrip("\\").strip() for part in parts)
        out.append((head + 1, joined))
        i += 1
    return out


def _is_advice(statement: str) -> bool:
    """A command being *printed* rather than run.

    Printed advice is wrong in its own ways and has its own guard
    (``test_remediation_paths.py``); resolving it against a working directory
    the developer chooses is not one of them.
    """
    return bool(re.match(r"^(?:echo|printf)\b", statement.lstrip()))


def invocations_in(text: str, path: str) -> list[Invocation]:
    """Every linter call in *text*, pinned form and bare form alike."""
    found: list[Invocation] = []
    for line, statement in logical_lines(text):
        stripped = statement.lstrip()
        if stripped.startswith("#") or _is_advice(statement):
            continue
        seen = {m.start(1) for m in PINNED_RE.finditer(statement)}
        for match in PINNED_RE.finditer(statement):
            found.append(Invocation(path, line, match.group(1), statement))
        for match in BARE_RE.finditer(statement):
            if match.start(1) in seen:  # the tool half of a `uv run <tool>`
                continue
            found.append(Invocation(path, line, match.group(1), statement))
    return found


def linter_scripts() -> dict[str, str]:
    """Every tracked shell script that runs a linter, as ``{path: text}``.

    Enumerated rather than declared, for the reason ``lint-shell.sh`` enumerates
    rather than reusing ``workspace_targets``: a hand-kept list answers "which
    scripts did someone think of", and the question here is "which scripts run a
    linter". Those differ exactly when a new one is added, which is when this
    guard has something to say.
    """
    scripts = {}
    for name in tracked_shell_files():
        text = (ROOT / name).read_text(encoding="utf-8")
        if invocations_in(text, name):
            scripts[name] = text
    assert scripts, "no shell script appears to run a linter — has the scan broken?"
    return scripts


def _dk_invocations() -> list[Invocation]:
    """``bin/dk``'s linter calls, with the same non-empty assertion as before.

    Every test below reports by collecting violations, and an empty collection
    is indistinguishable from a clean one — so a refactor that stopped matching
    (wrapping the call in a helper, say) would turn all of them green at once.
    """
    found = invocations_in(DK.read_text(encoding="utf-8"), rel(DK))
    assert found, f"no linter invocations matched in {rel(DK)} — restructured?"
    return found


def _unanchored_globs(cmd: str) -> list[str]:
    """Every ``packages/*`` in *cmd* that is not ``$PROJECT_ROOT``-anchored.

    Per occurrence, not per command. A command naming two targets can anchor
    one and leave the other bare, and a whole-command check would let the
    anchored one vouch for the bare one — reporting clean on a live defect.
    """
    normalized = cmd.replace('"$PROJECT_ROOT"/', "$PROJECT_ROOT/")
    return [
        normalized[max(0, m.start() - 24) : m.end()]
        for m in PACKAGES_GLOB_RE.finditer(normalized)
        if not normalized[: m.start()].endswith("$PROJECT_ROOT/")
    ]


def test_a_statement_is_read_whole_not_one_physical_line():
    """The predicate every test here rests on, pinned against constructed input.

    Against the live files it cannot fail — none currently wraps an invocation —
    so a check that only ran against them would report clean while being wrong
    about the case it exists to catch. That is the same reasoning as the glob
    test below, and the same reasoning that put a "keep it on one line" comment
    in ``bin/dk`` instead of a fix.
    """
    wrapped = 'run_from_root "Fixing" \\\n    uv run ruff check "$T" \\\n    --config "$C"\n'
    found = invocations_in(wrapped, "constructed")
    assert len(found) == 1, f"a wrapped invocation read as {len(found)} statements"
    assert "--config" in found[0].cmd, "the flag one line down read as absent"
    assert ROOT_RUNNER in found[0].cmd, "the runner one line up read as absent"
    assert found[0].line == 1, "a statement is reported at its head, not its tail"


def test_the_scan_finds_bare_invocations_not_only_pinned_ones():
    """``if ruff check …`` is an invocation. Reading only ``uv run`` misses it.

    Which is not hypothetical: ``bin/fix.sh`` — the command the gate prints on
    every failure — ran bare ``ruff``, and a scan anchored on ``uv run`` had
    nothing to say about it.
    """
    assert invocations_in('    if ruff check "$t" --config "$c"; then\n', "x")
    assert invocations_in('output=$(uv run mypy "$t" 2>&1)\n', "x")
    assert not invocations_in("    uv pip install pytest mypy ruff\n", "x"), (
        "a tool named in an argument list is not an invocation of it"
    )
    assert not invocations_in('    echo "  uv run ruff check packages/*/src"\n', "x"), (
        "printed advice is not an invocation; it has its own guard"
    )


def test_the_root_runner_actually_changes_directory():
    """The seam the ``bin/dk`` tests trust. Without this they assert a name."""
    body = DK.read_text(encoding="utf-8").split(f"{ROOT_RUNNER}() {{", 1)
    assert len(body) == 2, f"bin/dk defines no {ROOT_RUNNER}()"
    body = body[1].split("\n}", 1)[0]

    assert re.search(r'\(\s*cd "\$PROJECT_ROOT" &&', body), (
        f"{ROOT_RUNNER}() does not cd to $PROJECT_ROOT in a subshell; "
        "per-file-ignores resolve against the caller's directory"
    )


def test_every_script_that_runs_a_linter_runs_it_from_the_root():
    """``per-file-ignores`` keys are repo-relative; elsewhere they match nothing.

    An absolute ``--config`` does not cover this and neither does an absolute
    target path — only the working directory does. Two acceptable mechanisms,
    and a script needs one of them: ``cd`` to the root at the top, or dispatch
    every linter call through a seam that ``cd``s.
    """
    violations = []
    for path, text in linter_scripts().items():
        if CD_TO_ROOT_RE.search(text):
            continue
        violations += [
            f"{inv}: {inv.tool} runs from the caller's directory — "
            f"{path} neither cds to the root nor dispatches through {ROOT_RUNNER}"
            for inv in invocations_in(text, path)
            if ROOT_RUNNER not in inv.cmd
        ]

    assert not violations, "Linters run from the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_every_linter_invocation_in_dk_names_an_absolute_config():
    """Otherwise the tool silently reads whichever config is nearest ``$PWD``."""
    violations = []
    for inv in _dk_invocations():
        flag = CONFIG_FLAGS[inv.tool]
        if flag not in inv.cmd:
            violations.append(f"{inv}: {inv.tool} passes no {flag}")
        elif not re.search(rf'{re.escape(flag)}[= ]"?\$PROJECT_ROOT/', inv.cmd):
            violations.append(f"{inv}: {inv.tool} {flag} is not $PROJECT_ROOT-anchored")

    assert not violations, "Config resolved against the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_the_unanchored_glob_check_is_per_occurrence():
    """The predicate the next test trusts, pinned against constructed commands.

    Against the live file it cannot fail: every invocation there names exactly
    one target, so the mixed case below — one anchored, one bare — has nowhere
    to arise.
    """
    assert _unanchored_globs("uv run ruff check packages/*/src")
    assert not _unanchored_globs('uv run ruff check "$PROJECT_ROOT"/packages/*/src')
    assert not _unanchored_globs('uv run ruff check "$PROJECT_ROOT/packages/*/src"')
    assert _unanchored_globs('uv run ruff check "$PROJECT_ROOT"/packages/*/src packages/*/tests'), (
        "an anchored target elsewhere in the command must not vouch for a bare one"
    )


def test_every_linter_invocation_in_dk_names_absolute_targets():
    """The one resolution ``run_from_root`` does *not* fix.

    The shell expands these at the call site, before the subshell changes
    directory. A script that ``cd``s at the top has no such problem, which is
    why this requirement lands on ``bin/dk`` alone.
    """
    violations = [
        f"{inv}: {inv.tool} targets a relative glob: ...{snippet}"
        for inv in _dk_invocations()
        for snippet in _unanchored_globs(inv.cmd)
    ]

    assert not violations, "Targets resolved against the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
