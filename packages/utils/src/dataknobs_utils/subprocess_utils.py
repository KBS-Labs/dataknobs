"""Subprocess execution utilities for running system commands.

Provides functions for executing system commands and processing their
output line-by-line with callback handlers.
"""

import subprocess
from collections.abc import Callable
from typing import List, Union


def run_command(
    handle_line_fn: Callable[[str], bool], command: str, args: List[str] | None = None
) -> int:
    """Run a system command and process output line by line.

    Executes a command and calls a handler function for each line of output.
    The handler can signal early termination by returning False.

    Args:
        handle_line_fn: Callback function that takes each output line and returns
            True to continue processing or False to kill the process immediately.
        command: Command string with args (if args=None) or just command name
            (if args provided).
        args: Optional list of command arguments. If provided, command runs
            without shell=True. Defaults to None.

    Returns:
        int: The command's return code, or 0 if poll returns None.

    Examples:
        >>> from dataknobs_utils.subprocess_utils import run_command
        >>>
        >>> # Print all files in the directory:
        >>> run_command(lambda x: (print(x), True)[1], 'ls -1')
        >>>
        >>> # Print files until "foo" is found:
        >>> run_command(lambda x: (print(x), x!='foo')[1], 'ls -1')
    """
    the_args: Union[str, List[str]] = command
    shell = True
    if args is not None:
        the_args = [command] + args
        shell = False
    process = subprocess.Popen(the_args, stdout=subprocess.PIPE, shell=shell, encoding="utf8")
    while True:
        if process.stdout is not None:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                if not handle_line_fn(output.strip()):
                    process.kill()
                    break
        else:
            break
    rc = process.poll()
    return rc if rc is not None else 0
