import subprocess
from collections.abc import Callable
from typing import List, Union


def run_command(
    handle_line_fn: Callable[[str], bool], command: str, args: List[str] | None = None
) -> int:
    """Run a system command and do something with each line. Stop early by
    returning False from the handle_line_fn.

    Examples:
    # Print all files in the directory.
    >>> subprocess_utils.run_command(lambda x: (print(x), True)[1], 'ls -1')

    # Print files in the directory, stopping once "foo" is found.
    >>> subprocess_utils.run_command(lambda x: (print(x), x!='foo')[1], 'ls -1')

    :param handle_line_fn: A fn(output_line) that returns True to continue receiving
        lines or False to kill the process.
    :param command: A string with the command and its args or just the command
    :param args: The args for the command (if not None)
    :return: The command's return code
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
