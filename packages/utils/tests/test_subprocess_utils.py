"""Behavioural tests for ``run_command``."""

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from dataknobs_utils.subprocess_utils import run_command

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell semantics")


def _collect(command: str, args: list[str] | None = None) -> tuple[int, list[str]]:
    lines: list[str] = []

    def keep(line: str) -> bool:
        lines.append(line)
        return True

    rc = run_command(keep, command, args)
    return rc, lines


def _alive(pid: int) -> bool:
    """True while the process exists, including as an unreaped zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _reap_or_kill(pid: int) -> None:
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class TestRunCommand:
    def test_shell_command_lines_are_stripped(self) -> None:
        rc, lines = _collect("printf '  a  \\n\\nb\\nc'")
        assert rc == 0
        # Blank line delivered as "", final unterminated line delivered.
        assert lines == ["a", "", "b", "c"]

    def test_exit_status_returned(self) -> None:
        rc, lines = _collect("echo x; exit 3")
        assert rc == 3
        assert lines == ["x"]

    def test_unknown_shell_command(self) -> None:
        rc, _ = _collect("definitely-not-a-command-xyz 2>/dev/null")
        assert rc == 127

    def test_args_run_without_shell(self) -> None:
        rc, lines = _collect("echo", ["hello  world", "$HOME"])
        assert rc == 0
        assert lines == ["hello  world $HOME"]

    def test_empty_args_still_run_without_shell(self) -> None:
        rc, lines = _collect("pwd", [])
        assert rc == 0
        assert lines == [str(Path.cwd())]

    def test_missing_program_with_args(self) -> None:
        with pytest.raises(FileNotFoundError):
            run_command(lambda _line: True, "definitely-not-a-command-xyz", [])

    def test_utf8_output(self) -> None:
        _, lines = _collect("printf '\\303\\251\\n'")
        assert lines == ["é"]

    def test_stderr_not_captured(self, capfd: pytest.CaptureFixture[str]) -> None:
        rc, lines = _collect("echo to-err 1>&2; echo to-out")
        assert rc == 0
        assert lines == ["to-out"]
        assert "to-err" in capfd.readouterr().err

    def test_lines_delivered_as_produced(self) -> None:
        seen_at: list[float] = []

        def handle(_line: str) -> bool:
            seen_at.append(time.monotonic())
            return True

        start = time.monotonic()
        run_command(handle, "echo first; sleep 1; echo second")
        end = time.monotonic()
        assert len(seen_at) == 2
        assert seen_at[0] - start < 0.8
        assert end - seen_at[0] >= 0.8

    def test_stop_delivers_no_further_lines_and_returns_promptly(self) -> None:
        lines: list[str] = []

        def handle(line: str) -> bool:
            lines.append(line)
            return line != "2"

        start = time.monotonic()
        run_command(handle, "for i in 1 2 3 4; do echo $i; done; sleep 10; echo late")
        assert time.monotonic() - start < 5
        assert lines == ["1", "2"]

    def test_stop_returns_kill_status(self) -> None:
        rc = run_command(lambda line: line != "go", "echo go; sleep 10")
        assert rc == -signal.SIGKILL

    def test_stopped_child_is_reaped(self) -> None:
        pids: list[int] = []

        def handle(line: str) -> bool:
            pids.append(int(line))
            return False

        run_command(handle, "echo $$; sleep 10; echo late")
        try:
            assert not _alive(pids[0])
        finally:
            _reap_or_kill(pids[0])

    def test_raising_callback_kills_child_and_propagates(self) -> None:
        pids: list[int] = []

        def handle(line: str) -> bool:
            pids.append(int(line))
            raise ValueError("stop here")

        try:
            with pytest.raises(ValueError, match="stop here"):
                run_command(handle, "echo $$; sleep 10; echo late")
            assert not _alive(pids[0])
        finally:
            if pids:
                _reap_or_kill(pids[0])
