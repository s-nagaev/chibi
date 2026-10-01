"""Platform-aware subprocess helpers for shell command execution.

POSIX keeps the historical behavior: the child runs in a fresh session (its
own process group) and is killed via ``os.killpg``. Windows has neither
``setsid`` nor ``os.killpg``/``signal.SIGKILL``, so the child is isolated with
``CREATE_NEW_PROCESS_GROUP`` and the whole tree is force-killed with
``taskkill /F /T``.
"""

import asyncio
import os
import signal
import subprocess
import sys
from typing import Any


def get_new_process_group_kwargs() -> dict[str, Any]:
    """Build subprocess kwargs isolating the child in a new process group.

    Returns:
        ``{"start_new_session": True}`` on POSIX, ``{"creationflags":
        subprocess.CREATE_NEW_PROCESS_GROUP}`` on Windows.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Force-kill a subprocess together with all of its descendants.

    POSIX: sends ``SIGKILL`` to the whole process group, then falls back to
    killing the process itself and awaiting its exit; missing processes are
    ignored. Windows: runs ``taskkill /F /T`` against the process tree, then
    falls back to killing the process itself and always waits for it to exit.

    Args:
        process: The subprocess to terminate.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                timeout=10,
            )
        # Best-effort cleanup during a timeout: taskkill may fail because the
        # process is already gone (or the tool is unavailable). Swallowing is
        # safe here — a failed kill must not mask the original timeout error,
        # and the fallback ``process.kill()`` below still runs.
        except (OSError, subprocess.SubprocessError):
            pass

        try:
            process.kill()
        # The child may already be dead (e.g. killed by taskkill above).
        except ProcessLookupError:
            pass

        try:
            await process.wait()
        # The child may already have been reaped by the fallback kill.
        except ProcessLookupError:
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        # The whole process group may already be gone — nothing to kill.
        except ProcessLookupError:
            pass

        try:
            process.kill()
            await process.wait()
        # The child may have exited between the killpg call and here.
        except ProcessLookupError:
            pass
