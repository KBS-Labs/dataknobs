"""What ``bin/package-discovery.sh`` hands back to a script that sources it.

Two things, and it got both wrong: a status the caller can trust, and a
namespace the caller still owns.

Every script that builds, installs, tests, formats or validates this workspace
opens by asking this one which packages exist. None of them can tell a
workspace with no packages from a discovery that broke: both arrive as an empty
list, and an empty list turns every loop that follows into a no-op that reports
success having done nothing.

The callers were fixed to examine the status -- ``done < <(discover_packages)``
reports no status at all, while a captured assignment propagates one. That is
half the path, and it is the half that was missing. These pin the other half:
that ``discover_packages`` produces a status for a caller to examine.

It has to be pinned by driving the real script, because the defect lives in a
shell rule that reading the source does not show. ``set -e`` is not inherited
into a command-substitution subshell, and ``$(discover_packages)`` is the only
way to capture the function's output -- so a command failing inside it does not
abort it. Execution reaches the trailing ``echo``, which succeeds, and the
caller is handed exit 0 and an empty list. A stub standing in for the script
would reproduce the author's belief about bash rather than bash.

So the failures here are induced by shadowing a real external tool on ``PATH``.
The script under test is the shipped one, unmodified, run against the real
workspace: only its environment is hostile.

The namespace half is the same question asked of the other half of the
contract. This file is a library other scripts source, and its loop variables
were never declared local, so calling a function overwrote four names in the
caller. Nothing is bitten today -- every call site reaches it through ``$(...)``
and a subshell discards the damage -- but a direct call is the usage the
script's own help text documents, and ``for package in ...`` is what nine of
its callers do with the name it clobbers.
"""

from __future__ import annotations

import os
import subprocess

from tests._workspace import ROOT

DISCOVERY = ROOT / "bin" / "package-discovery.sh"

#: The caller shape that needs ``get_packages_in_order`` to return a status of
#: its own. Reached through ``$(...)``, so errexit is unset in that function's
#: own frame too, and a failing discovery inside it aborts nothing.
REAL_CALLER = "PACKAGES=($(get_packages_in_order))"


def _hostile(tmp_path, tool: str) -> dict[str, str]:
    """An environment where ``tool`` is found on PATH and exits non-zero."""
    stub = tmp_path / tool
    stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    return env


def _run(args: list[str], env: dict[str, str] | None = None):
    return subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, env=env, check=False
    )


def _assert_failed_deliberately(result, why: str) -> None:
    """Assert the run failed, and that it failed on purpose rather than crashed.

    Both ``basename`` tests below passed before this helper existed, on an
    unrelated ``set -u`` crash further down the same function. A non-zero exit
    is not on its own evidence that a status was propagated -- a script that
    dies of an unbound variable also exits non-zero, and it reports the same
    thing to a caller reading only the code.
    """
    assert result.returncode != 0, f"{why} (stdout={result.stdout!r})"
    assert "unbound variable" not in result.stderr, (
        "the run failed, but by crashing rather than by propagating a status; "
        f"this asserts nothing about {why} (stderr={result.stderr!r})"
    )


def _declared_packages() -> set[str]:
    """The packages the workspace actually has, read from the tree itself."""
    return {p.parent.name for p in ROOT.glob("packages/*/pyproject.toml")}


def test_the_shadow_really_shadows(tmp_path):
    """Guard the guard: the induced failure has to actually be induced.

    Every test below asserts that a run failed. If the shadowing stopped
    taking effect they would fail rather than pass, so this is not load
    bearing -- it is here so that when they do fail, the first question is
    already answered.
    """
    env = _hostile(tmp_path, "sort")
    probe = _run(["/bin/sh", "-c", "sort < /dev/null"], env=env)
    assert probe.returncode != 0, (
        "the PATH shadow did not take effect, so the runs below would be "
        "asserting against a healthy tool"
    )


def test_discovery_names_the_packages_the_workspace_has(tmp_path):
    """The healthy path, so the failure tests are not passing on a broken one.

    Derived from the tree rather than listed: a hardcoded roster would have to
    be edited by whoever adds a package, and nothing would prompt them.
    """
    result = _run([str(DISCOVERY), "list"])
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == _declared_packages()


def test_a_failing_sort_is_not_reported_as_an_empty_workspace(tmp_path):
    """The sort inside ``discover_packages`` runs in a process substitution.

    Its status is examined by nothing, so this reproduces whether or not
    errexit reaches the function: the names are collected, the sort that
    orders them fails, and the read loop it feeds simply sees no input.
    """
    result = _run([str(DISCOVERY), "list"], env=_hostile(tmp_path, "sort"))
    assert result.returncode != 0, (
        "a failing sort inside discover_packages was reported as success with "
        f"an empty package list (stdout={result.stdout!r})"
    )


def test_a_failing_basename_survives_the_substitution_boundary(tmp_path):
    """``ordered`` reaches ``discover_packages`` through ``$(...)``.

    Which is where errexit stops applying. Every name the loop collects is
    the empty string, they are dropped by the read loop that follows, and the
    function returns cleanly having found nothing.
    """
    result = _run([str(DISCOVERY), "ordered"], env=_hostile(tmp_path, "basename"))
    _assert_failed_deliberately(
        result,
        "a failing basename inside discover_packages crossing the substitution "
        "boundary as success",
    )


def test_the_ordering_relay_returns_a_status_of_its_own(tmp_path):
    """The shape fifteen call sites use, where the relay's own frame matters.

    ``ordered`` invokes ``get_packages_in_order`` at the top level of a script,
    where errexit is live and a failing capture aborts it. Every real caller
    instead reaches it through ``$(...)``, where errexit is not -- so the relay
    has to examine the status itself rather than inherit the abort.
    """
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'source "{DISCOVERY}"\n'
        f"{REAL_CALLER}\n"
        'echo "reached:${#PACKAGES[@]}"\n',
        encoding="utf-8",
    )
    driver.chmod(0o755)

    result = _run([str(driver)], env=_hostile(tmp_path, "basename"))
    _assert_failed_deliberately(
        result,
        "a failing discovery reaching a real caller as an empty package array, "
        "leaving the loop that follows it a silent no-op",
    )


def test_an_empty_workspace_is_an_empty_answer_and_not_a_crash(tmp_path):
    """The other half of "a failure must be distinguishable from no packages".

    Found by the two tests above passing before the fix, on this rather than on
    what they assert. ``discover_packages`` guards its own ``echo`` with
    ``${...[@]:-}`` because bash 3.2 treats an empty array as unset under
    ``set -u``; ``get_packages_in_order`` echoes the same shape unguarded, so
    the empty case dies there instead of returning empty.

    Run against a copy in a workspace that genuinely has no packages, since
    ``ROOT_DIR`` is derived from the script's own location and this repo will
    never be that workspace.
    """
    workspace = tmp_path / "bin" / DISCOVERY.name
    workspace.parent.mkdir(parents=True)
    (tmp_path / "packages").mkdir()
    workspace.write_bytes(DISCOVERY.read_bytes())
    workspace.chmod(0o755)

    for subcommand in ("list", "ordered"):
        result = _run([str(workspace), subcommand])
        assert result.returncode == 0, (
            f"{subcommand!r} on a workspace with no packages exited "
            f"{result.returncode} instead of reporting nothing found "
            f"(stderr={result.stderr!r})"
        )
        assert not result.stdout.split(), result.stdout


def test_sourcing_the_script_leaves_the_callers_names_alone(tmp_path):
    """A sourced function's loop variables belong to the function.

    Derived rather than listed: the probe diffs the shell's variable set across
    the calls, so a loop variable added later is caught without editing this.
    A named roster would have to be updated by whoever introduces the fifth
    leak, and nothing would prompt them.

    The canary is what keeps that from passing by measuring nothing -- a probe
    whose diff silently stopped working reports the same empty output as a
    library that leaks nothing. It deliberately leaks one name, and the
    assertion is that exactly that name comes back.
    """
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f'source "{DISCOVERY}"\n'
        "_canary() { dk_probe_canary=1; }\n"
        'before=""; after=""\n'
        "before=$(compgen -v | sort)\n"
        "discover_packages >/dev/null\n"
        "get_packages_in_order >/dev/null 2>&1\n"
        "workspace_targets >/dev/null\n"
        "_canary\n"
        "after=$(compgen -v | sort)\n"
        "comm -13 <(printf '%s\\n' \"$before\") <(printf '%s\\n' \"$after\")\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    result = _run([str(probe)])
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.split()

    assert "dk_probe_canary" in leaked, (
        "the probe did not detect a variable it leaks on purpose, so it is not "
        f"detecting anything (stdout={result.stdout!r})"
    )
    assert leaked == ["dk_probe_canary"], (
        "calling these functions overwrote names in the caller's shell: "
        f"{sorted(n for n in leaked if n != 'dk_probe_canary')}"
    )


def test_the_driver_above_matches_a_real_caller():
    """Pin the driver's shape to a caller that exists.

    A driver replicating a shape nobody uses would keep passing after the real
    callers moved on, which is the same defect one level up.
    """
    source = (ROOT / "bin" / "build-packages.sh").read_text(encoding="utf-8")
    assert REAL_CALLER in source, (
        f"no caller uses {REAL_CALLER!r} any more, so the driver above pins "
        "nothing -- re-derive it from the current callers"
    )
