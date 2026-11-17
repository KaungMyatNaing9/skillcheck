"""Shell helpers that kill the whole process group on timeout."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        process.kill()


def run_shell(command: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a shell command in `cwd`.

    `start_new_session` puts the shell in its own process group so a timeout
    kills grandchildren (for example `sh -c "python ..."`) as well as the shell.
    The caller decides whether the command is trusted. SkillCheck task files
    and agent command templates are trusted local input, not a sandbox boundary.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from None
    return subprocess.CompletedProcess(command, process.returncode or 0, stdout, stderr)
