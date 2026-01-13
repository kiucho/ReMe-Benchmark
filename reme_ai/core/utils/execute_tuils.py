"""Utility functions for executing code and shell commands.

This module provides helper functions for running Python code and shell commands,
with support for async execution and output capture.
"""

import asyncio
import contextlib
from io import StringIO


async def run_shell_command(cmd: str, timeout: float | None = 30) -> tuple[str, str, int]:
    """Execute a shell command asynchronously.

    Args:
        cmd: The shell command to execute.
        timeout: Maximum time to wait for command completion in seconds. None for no timeout.

    Returns:
        A tuple containing (stdout, stderr, return_code) as strings and integer.
    """
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if timeout:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    else:
        stdout, stderr = await process.communicate()

    return (
        stdout.decode("utf-8", errors="ignore"),
        stderr.decode("utf-8", errors="ignore"),
        process.returncode,
    )


def exec_code(code: str) -> str:
    """Execute Python code and capture the output.

    Args:
        code: The Python code string to execute.

    Returns:
        The captured stdout output, or the error message if execution fails.
    """
    try:
        redirected_output = StringIO()
        with contextlib.redirect_stdout(redirected_output):
            exec(code)

        return redirected_output.getvalue()

    except Exception as e:
        return str(e)

    except BaseException as e:
        return str(e)
