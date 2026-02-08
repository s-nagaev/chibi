import os
import signal
import subprocess
import sys
from pathlib import Path

import click
from dotenv import dotenv_values


class Service:
    """Service management for the Chibi bot."""

    def __init__(self) -> None:
        """Initialize service with PID and log paths."""
        home = Path.home()
        self.pid_path = home / ".chibi" / "chibi.pid"
        self.log_path = home / ".chibi" / "logs" / "chibi.log"
        self.settings_path = home / "chibi-bot" / "settings"

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_process_running(self, pid: int | str) -> bool:
        """Check if a process with the given PID is still running.

        Args:
            pid: Process ID to check.

        Returns:
            True if process is running, False otherwise.
        """
        try:
            os.kill(int(pid), 0)  # Signal 0 doesn't kill, just checks if process exists
            return True
        except (OSError, ProcessLookupError):
            return False

    def _read_pid(self) -> int | None:
        """Read PID from PID file.

        Returns:
            PID if file exists and is valid, None otherwise.
        """
        if not os.path.exists(self.pid_path):
            return None
        try:
            with open(self.pid_path, "r") as pid_file:
                return int(pid_file.read().strip())
        except (ValueError, IOError):
            return None

    def _write_pid(self, pid: str | int) -> None:
        """Write PID to PID file atomically.

        Args:
            pid: Process ID to write.
        """
        with open(self.pid_path, "w") as pid_file:
            pid_file.write(str(pid))

    def start(self) -> None:
        """Start the bot service in background."""
        self._ensure_directories()

        pid = self._read_pid()
        if pid is not None and self._is_process_running(pid):
            click.echo(f"Chibi service is already running. PID: {click.style(str(pid), fg='green', bold=True)}")
            return None

        self.pid_path.unlink(missing_ok=True)

        envs = os.environ.copy()

        if self.settings_path.exists():
            settings = dotenv_values(self.settings_path)
            clean_settings = {k: v for k, v in settings.items() if v is not None}
            envs.update(clean_settings)

        try:
            with self.log_path.open("a") as log_file:
                process = subprocess.Popen(
                    [sys.executable, "-m", "chibi"],
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                    env=envs,
                )
                try:
                    process.wait(timeout=3)

                    if process.returncode == 1:
                        msg = "The service was stopped immediately after the start attempt."
                        click.echo(
                            f"{click.style(msg, fg='red')}\n"
                            "Please check the logs:\n"
                            f"{click.style('$ chibi logs', fg='green', bold=True)}"
                        )
                        return None
                except subprocess.TimeoutExpired:
                    pid = process.pid
                    if pid and self._is_process_running(pid):
                        self._write_pid(pid)
                        click.echo(f"Chibi service started. PID: {click.style(str(pid), fg='green', bold=True)}")
                    else:
                        click.echo(
                            f"The service seems started, but no PID was found. Please check the logs.", err=True
                        )

        except Exception as e:
            click.echo(f"{click.style('Error starting Chibi service:', fg='red', bold=True)} {e}")

    def stop(self) -> None:
        """Stop the bot service."""
        pid = self._read_pid()

        if pid is None:
            click.secho("Service is not running.", fg="yellow", bold=True)
            return

        if not self._is_process_running(pid):
            self.pid_path.unlink(missing_ok=True)
            click.secho("Service is not running (stale PID file removed).", fg="yellow", bold=True)
            return

        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Chibi service stopped. PID: {click.style(pid, fg='green', bold=True)}")
            self.pid_path.unlink(missing_ok=True)
        except ProcessLookupError:
            self.pid_path.unlink(missing_ok=True)
            click.secho("Service process not found.", fg="red", bold=True)
        except Exception as e:
            click.echo(f"{click.style('Error stopping Chibi service:', fg='red', bold=True)} {e}")

    def restart(self) -> None:
        """Restart the bot service."""
        self.stop()
        self.start()
