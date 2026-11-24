"""The gate's own pace: the suite pool, the clock on its lines, the syntax pass.

Each of these is invisible in a verdict. A pool that leaves slots idle, a syntax
check that starts an interpreter per file, and an inline program that has not
parsed for months all produce the right exit code, so nothing that asks only
whether the gate passed can see any of them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._workspace import ROOT, load_bin_module, rel, tracked_shell_files

HELPER = ROOT / "bin" / "gate-progress.sh"
GATE = ROOT / "bin" / "run-quality-checks.sh"
SYNTAX = ROOT / "bin" / "check-python-syntax.py"


def _bash(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    # `set -e` because the gate sets it, and a helper that only works without
    # it is one the gate cannot use.
    return subprocess.run(
        ["bash", "-c", f'set -e\nsource "{HELPER}"\n{script}', "_", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


# --------------------------------------------------------------------------
# the pool
# --------------------------------------------------------------------------


def test_a_free_slot_is_refilled_while_the_oldest_job_still_runs() -> None:
    """Waiting on the oldest job leaves every other slot idle behind a slow one.

    ``slow`` is first and holds one of two slots until ``third`` has started,
    so the two can only overlap if the slot ``quick`` gave up was refilled. No
    clock is compared: ``slow`` gives up after a bounded wait and says so.
    """
    result = _bash(
        """
        dir=$(mktemp -d)
        job() {
            case "$1" in
                slow)
                    for _ in $(seq 100); do
                        [ -e "$dir/third" ] && { echo overlapped; return 0; }
                        sleep 0.1
                    done
                    echo "third never started"
                    ;;
                *) : > "$dir/$1" ;;
            esac
        }
        run_pool 2 job slow quick third
        rm -rf "$dir"
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[0] == "overlapped", result.stdout


def test_the_pool_never_exceeds_its_limit() -> None:
    result = _bash(
        """
        dir=$(mktemp -d)
        job() {
            : > "$dir/$1"
            ls "$dir" | wc -l
            sleep 0.3
            rm "$dir/$1"
        }
        run_pool 2 job a b c d e
        rmdir "$dir"
        """
    )
    assert result.returncode == 0, result.stderr
    counts = [int(line) for line in result.stdout.split()]
    assert len(counts) == 5
    assert max(counts) == 2, counts


@pytest.mark.parametrize("failing", ["first", "middle", "last"])
def test_one_failed_job_fails_the_pool_and_stops_nothing(failing: str) -> None:
    """A job already finished when it is waited for must still report its status."""
    result = _bash(
        """
        job() { echo "ran $1"; [ "$1" != "$FAILING" ] || return 3; }
        FAILING=$1
        status=0
        run_pool 2 job first middle last || status=$?
        echo "status $status"
        """,
        failing,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert sorted(lines) == ["ran first", "ran last", "ran middle", "status 1"]


def test_an_empty_pool_succeeds() -> None:
    assert _bash("run_pool 3 job").returncode == 0


def test_an_earlier_runs_coverage_is_cleared(tmp_path: Path) -> None:
    """A run that measures nothing must not report what an earlier run measured."""
    stale = ("coverage.xml", "coverage-unit-data.xml", ".coverage", ".coverage.unit.data")
    for name in stale:
        (tmp_path / name).write_text("earlier run\n")
    (tmp_path / "quality-summary.json").write_text("{}\n")

    result = _bash('clear_coverage "$1"', str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert [path.name for path in tmp_path.iterdir()] == ["quality-summary.json"]


def test_clearing_coverage_from_an_empty_directory_succeeds(tmp_path: Path) -> None:
    result = _bash('clear_coverage "$1"', str(tmp_path))
    assert result.returncode == 0, result.stderr


def test_the_gate_clears_coverage_before_any_check_runs() -> None:
    source = GATE.read_text(encoding="utf-8")
    clear = source.find('\nclear_coverage "$OUTPUT_DIR"\n')
    assert -1 < clear < source.index("\nrecord_check() {")


def test_packages_are_ordered_by_the_size_of_their_unit_suite(tmp_path: Path) -> None:
    """Integration tests run elsewhere, so they must not count towards the order."""
    for name, unit, integration in (("small", 10, 9000), ("large", 500, 0), ("mid", 90, 0)):
        tests = tmp_path / "packages" / name / "tests"
        (tests / "integration").mkdir(parents=True)
        (tests / "test_unit.py").write_text("#" * unit)
        (tests / "integration" / "test_service.py").write_text("#" * integration)
    (tmp_path / "packages" / "untested").mkdir()

    result = _bash('largest_suite_first "$@"', str(tmp_path), "small", "untested", "large", "mid")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["large", "mid", "small", "untested"]


def test_the_gate_runs_its_unit_suites_through_the_pool() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert re.search(r"^\s*run_pool .*\|\| UNIT_TEST_STATUS=", source, re.MULTILINE)


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (59, "59s"), (60, "1m00s"), (446, "7m26s"), (3600, "1h00m00s"), (3725, "1h02m05s")],
)
def test_elapsed_time_is_readable(seconds: int, text: str) -> None:
    assert _bash('format_elapsed "$1"', str(seconds)).stdout.strip() == text
    assert load_bin_module("quality-summary").format_elapsed(seconds) == text


def test_the_stamp_carries_the_time_of_day_and_the_time_since_the_start() -> None:
    result = _bash('progress_stamp "$(( $(date +%s) - 125 ))"')
    assert re.fullmatch(r"\[\d{2}:\d{2}:\d{2} \+2m0[5-9]s\]", result.stdout.strip()), result.stdout


def test_every_status_line_the_gate_prints_is_stamped() -> None:
    source = GATE.read_text(encoding="utf-8")
    for name in ("print_status", "print_success", "print_error", "print_warning"):
        body = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", source, re.MULTILINE | re.DOTALL)
        assert body is not None, f"{name} is not defined in {rel(GATE)}"
        assert 'progress_stamp "$RUN_START"' in body[1], f"{name} prints no stamp"


# --------------------------------------------------------------------------
# the syntax pass
# --------------------------------------------------------------------------


def _syntax(*targets: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SYNTAX), *map(str, targets)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_broken_file_is_named_not_just_the_first(tmp_path: Path) -> None:
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "fine.py").write_text("x = 1\n")
    (nested / "one.py").write_text("def (:\n")
    (tmp_path / "pkg" / "two.py").write_text("x = = 1\n")
    (tmp_path / "pkg" / "nul.py").write_bytes(b"x = 1\0\n")
    (tmp_path / "pkg" / "encoding.py").write_bytes(b"x = '\xff'\n")
    # Valid to a parser and refused by the compiler, so an ast.parse-only pass
    # (or ruff's parser alone) would be a weaker check than the one replaced.
    (tmp_path / "pkg" / "compiler.py").write_text("return 1\n")

    result = _syntax(tmp_path / "pkg")
    assert result.returncode == 1
    for name in ("one.py", "two.py", "nul.py", "encoding.py", "compiler.py"):
        assert name in result.stdout, f"{name} was not reported:\n{result.stdout}"
    assert "fine.py" not in result.stdout


def test_valid_targets_pass_and_say_how_many_files_were_read(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    (tmp_path / "notes.txt").write_text("def (:\n")
    single = tmp_path / "a.py"

    result = _syntax(tmp_path, single)
    assert result.returncode == 0, result.stdout + result.stderr
    # The file named twice is read once.
    assert "2 Python files" in result.stdout


def test_a_target_that_does_not_exist_is_a_failure(tmp_path: Path) -> None:
    result = _syntax(tmp_path / "absent")
    assert result.returncode == 1
    assert "absent" in result.stdout


def test_the_syntax_step_fails_validation_and_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n")
    result = subprocess.run(
        [str(ROOT / "bin" / "validate.sh"), "-q", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    section = result.stdout.split("1. Checking Python syntax", 1)[1].split("2. Running ruff", 1)[0]
    assert "broken.py" in section, section


def test_the_syntax_step_starts_one_interpreter_not_one_per_file() -> None:
    source = (ROOT / "bin" / "validate.sh").read_text(encoding="utf-8")
    assert "-m py_compile" not in source
    assert "check-python-syntax.py" in source


# --------------------------------------------------------------------------
# programs inside shell strings
# --------------------------------------------------------------------------

#: A multi-line ``python -c`` body, up to a closing quote that opens a line.
_INLINE_PROGRAM = re.compile(
    r"""python3?\ -c\ (?P<quote>["'])\n(?P<body>.*?)\n[ \t]*(?P=quote)""",
    re.DOTALL,
)


def _inline_programs() -> list[tuple[str, str]]:
    found = []
    for path in tracked_shell_files():
        text = (ROOT / path).read_text(encoding="utf-8")
        for match in _INLINE_PROGRAM.finditer(text):
            body = match["body"]
            if match["quote"] == '"':
                # What bash removes inside double quotes. Expansions are left in
                # place: each stands inside a Python string literal.
                body = re.sub(r'\\(["\\$`])', r"\1", body)
            line = text.count("\n", 0, match.start()) + 1
            found.append((f"{rel(ROOT / path)}:{line}", body))
    return found


def test_every_inline_python_program_compiles() -> None:
    """An inline program is checked by nothing until the moment it runs.

    The gate's coverage reader was dedented along with the shell around it, and
    from then on raised ``IndentationError`` on every run. ``2>/dev/null ||
    echo "null"`` turned that into a recorded absence of coverage, so nothing
    reported it for months.
    """
    programs = _inline_programs()
    assert programs, "no inline programs found — the pattern has stopped matching"
    broken = []
    for where, body in programs:
        try:
            compile(body, where, "exec")
        except SyntaxError as exc:
            broken.append(f"{where}: {exc.msg} (line {exc.lineno} of the program)")
    assert not broken, "\n".join(broken)
