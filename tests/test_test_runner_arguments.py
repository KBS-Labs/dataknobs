"""Reproduce-first guard for ``bin/test.sh`` argument handling.

A second package name landed in the catch-all arm of the argument loop, which
printed ``Unknown option:``, called ``show_usage``, and ended in ``exit 0``. So
``bin/test.sh bots common`` started no pytest process and reported success —
a green result for work that was never run, in the runner every other check
delegates to. At least two items' definition-of-done paired
``bin/validate.sh -f a b`` with a two-package test command; the first half ran
and the second half printed help.

Asserted through the script rather than by reading it, because the defect was
in what an exit code *was*, not in what the source said. Every case here is
chosen to fail before pytest starts — an unknown package, or an unknown flag —
so the guard costs a subprocess and no test run.

What this cannot cover: that a two-package invocation collects and passes.
``bin/test.sh`` runs one pytest per package in a loop, so multiple packages
never share a collection, and confirming the loop actually runs both would mean
running both suites. The probe below proves the second name is *accepted as a
package*, which is the half that was broken.
"""

from __future__ import annotations

import subprocess

from tests._workspace import ROOT

TEST_SH = ROOT / "bin" / "test.sh"

#: Names no package directory can have, so every invocation below fails at the
#: package-existence check — before services, before pytest.
MISSING = "definitely-not-a-package"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(TEST_SH), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_a_second_package_is_not_swallowed_as_an_unknown_option():
    """``bin/test.sh <pkg> <pkg>`` must reach the packages, not the usage text.

    Probed with a real package and a missing one: the run must fail *because
    the second name is not a package*, which is only reachable if the second
    positional was taken as a package name at all.
    """
    result = _run("common", MISSING)

    assert result.returncode != 0, (
        "a two-package invocation exited 0 without running anything:\n"
        f"{result.stdout}{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert MISSING in combined and "Unknown option" not in combined, (
        "the second package name reached the unknown-option arm instead of "
        f"being accepted as a package:\n{combined}"
    )


def test_an_unknown_option_is_rejected_as_an_option():
    """A misspelled flag must fail, and fail as a *flag*.

    The exit code alone was already non-zero here, but for the wrong reason:
    a leading flag fell through to the same arm that takes a package name, so
    ``--no-such-flag`` was looked up as a package and rejected for not
    existing. That is an accidental pass — it depends on no package ever
    being named like a flag — so the message is asserted too.

    Same root cause as the case above from the other side: ``show_usage``
    ended in ``exit 0``, so every path reaching it reported success, whether
    it was reached by ``--help`` or by an error.
    """
    result = _run("--no-such-flag")
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "an unknown flag exited 0:\n" + combined
    assert "Unknown option" in combined, (
        "an unknown flag was diagnosed as a missing package rather than as a "
        f"bad option:\n{combined}"
    )


def test_help_still_succeeds():
    """The one caller for which exit 0 is right, so the fix cannot overshoot."""
    result = _run("--help")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage:" in result.stdout
