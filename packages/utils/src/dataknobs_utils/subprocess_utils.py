# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Subprocess execution utilities for running system commands.

Provides functions for executing system commands and processing their
output line-by-line with callback handlers.
"""

import logging
import subprocess
from collections.abc import Callable
from typing import List

logger = logging.getLogger(__name__)


def run_command(
    handle_line_fn: Callable[[str], bool], command: str, args: List[str] | None = None
) -> int:
    """Run a system command and process its standard output line by line.

    Each line of standard output is decoded as UTF-8, stripped of leading and
    trailing whitespace, and passed to ``handle_line_fn`` as soon as it is
    produced (a blank line arrives as "", and a final line without a newline
    is still delivered). Standard error is not captured and goes to the
    parent's standard error.

    If ``handle_line_fn`` returns a false value, the child process is killed
    (SIGKILL on POSIX), no further output is read, and the child is waited
    for; its exit status (``-9`` on POSIX) is returned. Output still held open
    by processes the child started is not waited for. If ``handle_line_fn``
    raises, the child is killed and waited for and the exception propagates.
    No child process is left running or unreaped on return.

    Args:
        handle_line_fn: Callback function that takes each output line and returns
            True to continue processing or False to kill the process immediately.
        command: With ``args=None``, a command line run by the system shell
            (``sh`` on POSIX). Otherwise, the program to execute directly.
        args: Optional list of arguments. If given (even empty), ``command`` is
            executed directly with these arguments and no shell. Defaults to
            None.

    Returns:
        int: The command's exit status.

    Raises:
        FileNotFoundError: If ``args`` is given and the program cannot be found.

    Examples:
        >>> from dataknobs_utils.subprocess_utils import run_command
        >>>
        >>> # Collect all files in the directory:
        >>> names = []
        >>> run_command(lambda x: (names.append(x), True)[1], 'ls -1')
        >>>
        >>> # Collect files until "foo" is found:
        >>> run_command(lambda x: (names.append(x), x != 'foo')[1], 'ls -1')
    """
    if args is None:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, encoding="utf-8")
    else:
        proc = subprocess.Popen([command, *args], stdout=subprocess.PIPE, encoding="utf-8")

    stdout = proc.stdout
    try:
        if stdout is not None:
            for raw_line in stdout:
                if not handle_line_fn(raw_line.strip()):
                    logger.debug("Handler stopped command early; killing pid %d", proc.pid)
                    proc.kill()
                    break
    except BaseException:
        proc.kill()
        raise
    finally:
        if stdout is not None:
            stdout.close()
        proc.wait()
    return proc.returncode
