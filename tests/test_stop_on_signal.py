"""What an interrupt does to a script that runs suites.

A trap whose handler returns does not stop a script: bash runs it and goes on to
the next statement. And a background job of a script starts with SIGINT ignored,
so the pool's suites never saw a Ctrl-C at all. The gate carried on through both.

The signals here are sent to a session of their own with a real terminal, since
a Ctrl-C is the terminal interrupting its whole foreground process group and
nothing else delivers it the same way.
"""

from __future__ import annotations

import contextlib
import os
import pty
import re
import select
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests._workspace import ROOT, rel, tracked_shell_files

POOL = ROOT / "bin" / "gate-progress.sh"
HELPER = ROOT / "bin" / "stop-on-signal.sh"

#: The scripts that run suites and clean up after themselves.
CALLERS = ("run-quality-checks.sh", "run-integration-tests.sh", "test.sh")

_SIGNALS = {"INT": signal.SIGINT, "TERM": signal.SIGTERM, "HUP": signal.SIGHUP}

#: Long enough to outlast every wait below, so a job still running was missed
#: rather than finished.
_JOB_SECONDS = 120

#: Run a command with its stdin as its controlling terminal. The signals are
#: restored because bash cannot trap one that was ignored when it started, and
#: this suite may itself be running as a background job.
_AS_A_TERMINAL_SESSION = """
import fcntl, os, signal, sys, termios
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(number, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
"""


def _until(condition: Callable[[], bool], seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _Session:
    """A script running as the foreground process group of its own terminal."""

    def __init__(self, tmp_path: Path, script: str) -> None:
        self.dir = tmp_path
        path = tmp_path / "script.sh"
        path.write_text(
            f'set -e\nDIR="{tmp_path}"\nsource "{POOL}"\nsource "{HELPER}"\n'
            + textwrap.dedent(script)
        )
        self._terminal, child_end = pty.openpty()

        self.process = subprocess.Popen(
            [sys.executable, "-c", _AS_A_TERMINAL_SESSION, "bash", str(path)],
            stdin=child_end,
            stdout=child_end,
            stderr=child_end,
            start_new_session=True,
        )
        os.close(child_end)
        self._output = b""

    def _drain(self) -> None:
        while select.select([self._terminal], [], [], 0)[0]:
            try:
                data = os.read(self._terminal, 4096)
            except OSError:
                return
            if not data:
                return
            self._output += data

    @property
    def output(self) -> str:
        self._drain()
        return self._output.decode(errors="replace")

    def pids(self) -> dict[str, int]:
        found = {}
        for path in self.dir.glob("*.pid"):
            text = path.read_text().strip()
            if text:
                found[path.stem] = int(text)
        return found

    def wait_for_jobs(self, count: int) -> None:
        def ready() -> bool:
            self._drain()
            return len(self.pids()) == count or self.process.poll() is not None

        _until(ready)
        assert len(self.pids()) == count, self.output

    def ctrl_c(self) -> None:
        os.write(self._terminal, b"\x03")

    def send(self, name: str) -> None:
        self.process.send_signal(_SIGNALS[name])

    def exited(self) -> bool:
        def done() -> bool:
            self._drain()
            return self.process.poll() is not None

        return _until(done)

    def survivors(self) -> list[str]:
        _until(lambda: not any(map(_alive, self.pids().values())), 2.0)
        return sorted(name for name, pid in self.pids().items() if _alive(pid))

    def close(self) -> None:
        for pid in [*self.pids().values(), self.process.pid]:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        os.close(self._terminal)
        self.process.wait(timeout=10)


_Start = Callable[[str], _Session]


@pytest.fixture
def session(tmp_path: Path) -> Iterator[_Start]:
    started: list[_Session] = []

    def start(script: str) -> _Session:
        started.append(_Session(tmp_path, script))
        return started[-1]

    yield start
    for each in started:
        each.close()


_POOL_SCRIPT = f"""
    trap 'echo cleaned up' EXIT
    stop_on_signals
    job() {{
        bash -c 'echo $$ > "$0"; exec sleep {_JOB_SECONDS}' "$DIR/$1.pid" || return $?
    }}
    run_pool 2 job a b queued || true
    echo "next stage"
"""


@pytest.mark.parametrize("how", ["ctrl-c", "INT", "TERM", "HUP"])
def test_a_signal_during_the_pool_stops_the_script_and_its_jobs(session: _Start, how: str) -> None:
    run = session(_POOL_SCRIPT)
    run.wait_for_jobs(2)

    if how == "ctrl-c":
        run.ctrl_c()
    else:
        run.send(how)

    assert run.exited(), run.output
    assert run.survivors() == []
    assert "next stage" not in run.output
    assert "queued" not in run.pids()
    assert run.output.count("cleaned up") == 1
    # Died of the signal rather than exiting with a number: that is what makes
    # bash running this script stop as well.
    expected = _SIGNALS["INT" if how == "ctrl-c" else how]
    assert run.process.returncode == -expected


def test_a_ctrl_c_while_the_last_jobs_are_awaited_stops_them(session: _Start) -> None:
    """The pool is then in `wait`, not in the poll that fills a slot."""
    run = session(_POOL_SCRIPT.replace("a b queued", "a b"))
    run.wait_for_jobs(2)
    time.sleep(0.5)
    run.ctrl_c()
    assert run.exited(), run.output
    assert run.survivors() == []
    assert "next stage" not in run.output


def test_a_job_that_ignores_the_interrupt_is_killed(session: _Start) -> None:
    run = session(
        _POOL_SCRIPT.replace("stop_on_signals", "STOP_GRACE_SECONDS=1 stop_on_signals").replace(
            "bash -c 'echo", 'bash -c \'trap "" INT TERM HUP; echo'
        )
    )
    run.wait_for_jobs(2)
    run.ctrl_c()
    assert run.exited(), run.output
    assert run.survivors() == []


def test_a_job_gets_to_finish_its_teardown(session: _Start) -> None:
    """A suite's fixtures drop the databases and indices they made."""
    run = session(
        f"""
        stop_on_signals
        job() {{
            "{sys.executable}" -c '
        import os, sys, time
        open(sys.argv[1], "w").write(str(os.getpid()))
        try:
            time.sleep({_JOB_SECONDS})
        finally:
            time.sleep(1)
            open(sys.argv[2], "w").write("done")
        ' "$DIR/a.pid" "$DIR/torn-down" || return $?
        }}
        run_pool 2 job a || true
        """
    )
    run.wait_for_jobs(1)
    run.ctrl_c()
    assert run.exited(), run.output
    assert (run.dir / "torn-down").read_text() == "done"


def test_a_ctrl_c_in_a_foreground_command_stops_the_script(session: _Start) -> None:
    run = session(
        f"""
        trap 'echo cleaned up' EXIT
        stop_on_signals
        bash -c 'echo $$ > "$0"; exec sleep {_JOB_SECONDS}' "$DIR/a.pid" || echo "recorded"
        echo "next stage"
        """
    )
    run.wait_for_jobs(1)
    run.ctrl_c()
    assert run.exited(), run.output
    assert run.survivors() == []
    assert "next stage" not in run.output
    assert run.output.count("cleaned up") == 1


def test_a_pool_job_does_not_read_the_terminal(session: _Start) -> None:
    """Outside the foreground group, a read of it would stop the job for good."""
    run = session(
        """
        stop_on_signals
        job() { if read -r line; then echo "read something"; else echo "end of input"; fi; }
        run_pool 2 job a
        echo "pool done"
        """
    )
    assert run.exited(), run.output
    assert "end of input" in run.output
    assert "pool done" in run.output


def test_a_run_nothing_interrupts_is_unchanged(session: _Start) -> None:
    run = session(
        """
        trap 'echo cleaned up' EXIT
        stop_on_signals
        job() { [ "$1" != bad ] || return 3; }
        run_pool 2 job a bad b || echo "pool failed"
        echo "next stage"
        exit 4
        """
    )
    assert run.exited(), run.output
    assert run.process.returncode == 4
    assert run.output.split() == ["pool", "failed", "next", "stage", "cleaned", "up"]


_TRAPS_A_SIGNAL = re.compile(r"^\s*trap\s+(?!-\s|--\s).*\b(?:SIG)?(?:INT|TERM|HUP)\b")


def test_no_script_handles_a_signal_its_own_way() -> None:
    """One handler, because the obvious spelling is the one that does not stop.

    ``trap cleanup EXIT INT TERM`` reads as "clean up and stop" and was written
    three times here.
    """
    offenders = [
        f"{name}:{number}: {line.strip()}"
        for name in tracked_shell_files()
        if name != rel(HELPER)
        for number, line in enumerate((ROOT / name).read_text().splitlines(), 1)
        if _TRAPS_A_SIGNAL.match(line)
    ]
    assert not offenders, "use stop_on_signals from bin/stop-on-signal.sh:\n" + "\n".join(offenders)


@pytest.mark.parametrize("line", ["trap cleanup EXIT INT TERM", "    trap 'x' SIGINT"])
def test_the_guard_recognises_the_spelling_it_refuses(line: str) -> None:
    assert _TRAPS_A_SIGNAL.match(line)


@pytest.mark.parametrize("name", CALLERS)
def test_each_suite_runner_stops_on_signals(name: str) -> None:
    text = (ROOT / "bin" / name).read_text()
    assert re.search(r"^\s*source .*stop-on-signal\.sh\"$", text, re.M)
    assert re.search(r"^\s*stop_on_signals$", text, re.M)
