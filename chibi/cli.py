import os
import subprocess
import sys

import click

from chibi.config_generator import CONFIG_PATH, generate_default_config
from chibi.service import Service

service = Service()


@click.group()
def main() -> None:
    """Chibi CLI for managing the bot service."""
    pass


@main.command()
def start() -> None:
    """Start the Chibi bot service as a daemon."""
    service.start()


@main.command()
def stop() -> None:
    """Stop the running Chibi bot service."""
    service.stop()


@main.command()
def restart() -> None:
    """Restart the Chibi bot service."""
    service.restart()


@main.command()
def config() -> None:
    """Open the Chibi configuration file in the default editor."""
    if not CONFIG_PATH.exists():
        click.echo("Generating default Chibi configuration...")
        generate_default_config()

    content = ""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            click.echo(f"Error reading config: {e}", err=True)
            sys.exit(1)

    try:
        new_content = click.edit(text=content, extension=".env")
    except click.ClickException:
        click.echo("Error: No text editor found in the system.", err=True)
        click.echo("Please set the EDITOR environment variable or install vim/nano.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)

    # Сохраняем, только если были изменения
    if new_content is not None:
        # click.edit возвращает строку с \n на конце, даже если исходник был без него.
        # Можно добавить .strip(), если это критично, но для env файлов лучше оставить.
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
            click.echo("Settings updated.")
        except OSError as e:
            click.echo(f"Error saving config: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("No changes made.")


@main.command()
def logs() -> None:
    """Tail the Chibi log file."""
    log_path = service.log_path

    if not log_path.exists():
        click.echo(f"Log file {log_path} does not exist yet. Start the service first.")
        sys.exit(1)

    try:
        subprocess.call(["tail", "-n", "50", "-f", log_path.absolute()])
    except KeyboardInterrupt:
        click.echo("\nLog tailing stopped.")
    except FileNotFoundError:
        click.echo("Error: 'tail' command not found.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error tailing logs: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
