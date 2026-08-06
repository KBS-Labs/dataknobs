"""Reproduce-first guard for CWD-dependent config resolution in ``bin/dk``.

``ruff`` and ``pylint`` both walk up from the working directory when no config
is passed. Every other script in ``bin/`` sidesteps that by ``cd``-ing to the
repo root first; ``bin/dk`` never ``cd``s, and its root check tests
``$PROJECT_ROOT`` rather than ``$PWD``, so running it from a subdirectory is
explicitly allowed.

That combination made ``bin/dk fix`` resolve the per-package ``[tool.ruff]``
sections instead of the root one the quality gate uses — so the documented fix
command applied rewrites the gate declines.

Three separate resolutions are involved, and passing ``--config`` fixes only
the first:

1. **The config file.** Fixed by an absolute ``--config`` / ``--rcfile``.
2. **``per-file-ignores`` keys**, which are repo-relative paths matched against
   the working directory. From anywhere else they match nothing and every
   suppression silently lifts, so the tool reports findings the gate does not
   have. Only a working directory of the root fixes this — an absolute target
   path does *not*, which is why the run goes through a ``cd``.
3. **Target globs**, expanded by the shell at the call site before any ``cd``
   takes effect, so those must be ``$PROJECT_ROOT``-anchored where they appear.

Asserted per invocation rather than per file, because the original defect was
two lines in one function disagreeing about which mechanism they used.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DK = ROOT / "bin" / "dk"

#: The linters that resolve config by walking up from the working directory.
INVOCATION_RE = re.compile(r"^\s*uv run (ruff|pylint|black|mypy)\b.*$", re.MULTILINE)

#: How each names its config file. A tool absent here takes no config flag.
CONFIG_FLAGS = {
    "ruff": "--config",
    "pylint": "--rcfile",
    "mypy": "--config-file",
    "black": "--config",
}


#: The single seam that runs a command with the repo root as working directory.
ROOT_RUNNER = "run_from_root"

#: A target naming every package. Located per occurrence, because one command
#: can name several targets and anchor only some of them.
PACKAGES_GLOB_RE = re.compile(r"packages/\*")


def _invocations() -> list[tuple[int, str, str]]:
    """``(line number, tool, full command)`` for every linter call in ``bin/dk``.

    Asserts it found something. Every test below reports by collecting
    violations, and an empty collection is indistinguishable from a clean one —
    so a refactor that stopped matching ``INVOCATION_RE`` (wrapping the call in
    a helper, say) would turn all of them green at once. This is the one place
    that can tell the difference, so it is the one place that checks.
    """
    text = DK.read_text(encoding="utf-8")
    found = [
        (text[: m.start()].count("\n") + 1, m.group(1), m.group(0).strip())
        for m in INVOCATION_RE.finditer(text)
    ]
    assert found, f"no linter invocations matched in {DK} — has it been restructured?"
    return found


def _statement_head(lines: list[str], line: int) -> str:
    """The first line of the shell statement containing 1-indexed *line*.

    Walks back over line continuations rather than guessing at a window, so
    the runner is identified by the statement it heads and not by proximity.
    """
    i = line - 1
    while i > 0 and lines[i - 1].rstrip().endswith("\\"):
        i -= 1
    return lines[i]


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


def test_the_root_runner_actually_changes_directory():
    """The seam the other two tests trust. Without this they assert a name."""
    body = DK.read_text(encoding="utf-8").split(f"{ROOT_RUNNER}() {{", 1)
    assert len(body) == 2, f"bin/dk defines no {ROOT_RUNNER}()"
    body = body[1].split("\n}", 1)[0]

    assert re.search(r'\(\s*cd "\$PROJECT_ROOT" &&', body), (
        f"{ROOT_RUNNER}() does not cd to $PROJECT_ROOT in a subshell; "
        "per-file-ignores resolve against the caller's directory"
    )


def test_every_linter_invocation_runs_from_the_root():
    """``per-file-ignores`` keys are repo-relative; elsewhere they match nothing.

    An absolute ``--config`` does not cover this and neither does an absolute
    target path — only the working directory does.
    """
    lines = DK.read_text(encoding="utf-8").splitlines()
    violations = [
        f"bin/dk:{line}: {tool} is not run through {ROOT_RUNNER}"
        for line, tool, _cmd in _invocations()
        if ROOT_RUNNER not in _statement_head(lines, line)
    ]

    assert not violations, "Linters run from the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_every_linter_invocation_names_an_absolute_config():
    """Otherwise the tool silently reads whichever config is nearest ``$PWD``."""
    violations = []
    for line, tool, cmd in _invocations():
        flag = CONFIG_FLAGS[tool]
        if flag not in cmd:
            violations.append(f"bin/dk:{line}: {tool} passes no {flag}")
        elif not re.search(rf'{re.escape(flag)}[= ]"?\$PROJECT_ROOT/', cmd):
            violations.append(f"bin/dk:{line}: {tool} {flag} is not $PROJECT_ROOT-anchored")

    assert not violations, "Config resolved against the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )


def test_the_unanchored_glob_check_is_per_occurrence():
    """The predicate the next test trusts, pinned against constructed commands.

    Against the live file it cannot fail: every invocation there names exactly
    one target, so the mixed case below — one anchored, one bare — has nowhere
    to arise. A check that only ran against the file would report clean while
    being wrong about the case it exists to catch.
    """
    assert _unanchored_globs("uv run ruff check packages/*/src")
    assert not _unanchored_globs('uv run ruff check "$PROJECT_ROOT"/packages/*/src')
    assert not _unanchored_globs('uv run ruff check "$PROJECT_ROOT/packages/*/src"')
    assert _unanchored_globs(
        'uv run ruff check "$PROJECT_ROOT"/packages/*/src packages/*/tests'
    ), "an anchored target elsewhere in the command must not vouch for a bare one"


def test_every_linter_invocation_names_absolute_targets():
    """An unmatched relative glob is passed through literally, not expanded.

    The shell expands these at the call site, before ``run_from_root`` changes
    directory — so unlike the config, this one is not fixed by the ``cd``.
    """
    violations = [
        f"bin/dk:{line}: {tool} targets a relative glob: ...{snippet}"
        for line, tool, cmd in _invocations()
        for snippet in _unanchored_globs(cmd)
    ]

    assert not violations, "Targets resolved against the caller's directory:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
