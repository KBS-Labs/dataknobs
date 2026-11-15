"""What the quality gate shows about a test suite that failed.

A red suite under an empty "Test Failure Details" block sends the reader to the
artifacts to find out what broke. The gate showed only a suite's ``FAILED``
lines, or its tail when the output matched one of four patterns, and a pytest
run can fail without producing either: an xdist run whose workers collect
different test ids aborts with ``ERROR collecting gw1``, and a fixture that
raises reports ``ERROR at setup of``. Both printed nothing.

The outputs here are produced by running pytest, with the report flags the
gate itself passes, rather than written by hand, so a pytest or xdist release
that changes what a failure looks like fails here rather than in a gate run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bin" / "test-output-detail.sh"
GATE = ROOT / "bin" / "run-quality-checks.sh"


def _bash(script: str, *args: str) -> str:
    result = subprocess.run(
        ["bash", "-c", f'source "{HELPER}"\n{script}', "_", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _gate_flags() -> list[str]:
    return _bash('printf "%s" "$GATE_PYTEST_REPORT_FLAGS"').split()


def _pytest_output(tmp_path: Path, tests: str, *extra: str) -> Path:
    """Run pytest the way the gate does over ``tests`` and keep what it wrote.

    Isolated from this repository's configuration and from the run executing
    this test: its own directory as rootdir, and none of the environment that
    pytest-cov and an outer xdist run pass to children.
    """
    (tmp_path / "test_case.py").write_text(textwrap.dedent(tests))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE", "PYTEST_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
            *_gate_flags(),
            *extra,
            "test_case.py",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode not in (0, 5), "the case was meant to fail:\n" + result.stdout
    output = tmp_path / "unit-test-output-case.txt"
    output.write_text(result.stdout + result.stderr)
    return output


def _detail(output: Path) -> str:
    return _bash('print_test_output_detail "$1"', str(output))


_WORKER_IDS_DIFFER = """
    import pytest

    def f():
        pass

    @pytest.mark.parametrize("x", [f, object()], ids=repr)
    def test_x(x):
        pass
"""

_FIXTURE_RAISES = """
    import pytest

    @pytest.fixture
    def broken():
        raise RuntimeError("fixture exploded")

    def test_needs_the_fixture(broken):
        pass
"""


def test_a_worker_collection_mismatch_shows_its_cause(tmp_path: Path) -> None:
    """The case that hid a branch's failure: ids carrying an object's address."""
    pytest.importorskip("xdist")
    output = _pytest_output(tmp_path, _WORKER_IDS_DIFFER, "-n", "2")

    detail = _detail(output)

    assert "Different tests were collected" in detail, detail


def test_a_fixture_error_names_the_test_it_stopped(tmp_path: Path) -> None:
    output = _pytest_output(tmp_path, _FIXTURE_RAISES)

    detail = _detail(output)

    assert "test_needs_the_fixture" in detail, detail


def test_a_failure_beside_an_error_shows_both(tmp_path: Path) -> None:
    """FAILED lines alone used to win, so the error beside them went unnamed."""
    output = _pytest_output(
        tmp_path,
        _FIXTURE_RAISES
        + """
    def test_that_fails():
        assert False
""",
    )

    detail = _detail(output)

    assert "FAILED" in detail and "test_that_fails" in detail, detail
    assert "ERROR" in detail and "test_needs_the_fixture" in detail, detail


def test_a_plain_failure_shows_its_failed_line(tmp_path: Path) -> None:
    output = _pytest_output(
        tmp_path,
        """
        def test_that_fails():
            assert 1 == 2
        """,
    )

    detail = _detail(output)

    assert "FAILED test_case.py::test_that_fails" in detail, detail


@pytest.mark.parametrize(
    "content",
    ["", "Traceback (most recent call last):\n  ...\nModuleNotFoundError: x\n"],
    ids=["empty", "no-pytest-summary"],
)
def test_a_failed_suite_never_shows_nothing(tmp_path: Path, content: str) -> None:
    """The helper is asked only about a suite that failed, so silence is a lie."""
    output = tmp_path / "unit-test-output-case.txt"
    output.write_text(content)

    assert _detail(output).strip()


def test_only_the_suites_that_failed_are_reported(tmp_path: Path) -> None:
    """The summary lists what each suite's verdict recorded, not every file.

    A suite that passed leaves an output file too. Reading the directory and
    asking each file whether it looks failed is how the patterns came to
    decide, so the verdict is recorded where it is produced instead.
    """
    failed = _pytest_output(tmp_path, _FIXTURE_RAISES)
    passed = tmp_path / "unit-test-output-other.txt"
    passed.write_text("1 passed in 0.01s\n")
    record = tmp_path / "failed-suites.txt"

    inline = _bash('report_failed_suite "$1" "$2"', str(failed), str(record))
    summary = _bash('print_failure_details "$1"', str(record))

    assert "test_needs_the_fixture" in inline, inline
    assert record.read_text().splitlines() == [str(failed)]
    assert "unit (case)" in summary and "test_needs_the_fixture" in summary, summary
    assert "other" not in summary, summary


def test_nothing_is_reported_when_no_suite_failed(tmp_path: Path) -> None:
    assert _bash('print_failure_details "$1"', str(tmp_path / "absent.txt")) == ""


def test_the_gate_decides_what_a_failure_shows_in_one_place() -> None:
    """Two inline copies grepped ``^FAILED`` alone after the helper learned more.

    The previous fix to this gave the summary a fallback and left the per-suite
    lines printed as each suite finished on the old rule, so the first thing a
    developer saw about a collection error was still nothing.
    """
    gate = GATE.read_text()

    assert "'^FAILED '" not in gate, (
        "bin/run-quality-checks.sh greps FAILED lines itself; route it through "
        "report_failed_suite in bin/test-output-detail.sh"
    )
    assert "print_failure_details" in gate
