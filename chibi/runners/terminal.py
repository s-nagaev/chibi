"""Terminal runner for Chibi AI assistant - REPL interface."""

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from chibi.services.bot import handle_image_generation, handle_user_prompt
from chibi.services.task_manager import task_manager
from chibi.services.terminal_interface import TerminalInterface
from chibi.services.user import get_info, get_models_available, reset_chat_history, set_active_model

console = Console()


# Commands that the REPL supports
COMMANDS = {
    "help": "Show available commands",
    "reset": "Reset chat history",
    "model": "Show/select model (numbered menu)",
    "imagine": "Generate image from prompt (e.g., /imagine a sunset)",
    "info": "Show current user and model info",
    "quit": "Exit the REPL",
}


class TerminalRunner:
    """Terminal REPL runner for Chibi AI assistant."""

    def __init__(self, user_id: int) -> None:
        """Initialize the terminal runner.

        Args:
            user_id: The user ID to use for this session.
        """
        self.user_id = user_id
        self.interface = TerminalInterface(user_id=user_id)
        self._running = True

        # Setup history file
        history_dir = Path.home() / ".chibi" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        self._history = FileHistory(str(history_dir / f"user_{user_id}.txt"))

        # Setup prompt session
        self._setup_keys()
        self._setup_completer()

    def _setup_keys(self) -> None:
        """Setup key bindings for the prompt."""
        self._kb = KeyBindings()

        @self._kb.add("c-c")
        def interrupt(event):
            """Handle Ctrl+C - interrupt current operation."""
            console.print("\n[yellow]Interrupted. Press Ctrl+D or type /quit to exit.[/yellow]")
            event.app.current_buffer.text = ""
            event.app.redraw()

        @self._kb.add("c-d")
        def exit_event(event):
            """Handle Ctrl+D - exit the REPL."""
            console.print("\n[cyan]Goodbye![/cyan]")
            self._running = False
            event.app.exit()

    def _setup_completer(self) -> None:
        """Setup command completer for the prompt."""
        self._completer = WordCompleter(
            list(COMMANDS.keys()) + ["/" + cmd for cmd in COMMANDS.keys()],
            ignore_case=True,
        )

    def _create_session(self) -> PromptSession:
        """Create a new prompt session."""
        return PromptSession(
            message=">>> ",
            history=self._history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=self._completer,
            key_bindings=self._kb,
        )

    async def _handle_help(self) -> None:
        """Show help message."""
        console.print("\n[bold]Available commands:[/bold]")
        for cmd, desc in COMMANDS.items():
            console.print(f"  /{cmd:<10} - {desc}")
        console.print()

    async def _handle_reset(self) -> None:
        """Reset chat history."""
        console.print("[yellow]Resetting chat history...[/yellow]")
        await reset_chat_history(user_id=self.user_id, thread_id=0)
        console.print("[green]Chat history has been reset.[/green]")

    async def _handle_model_selection(self) -> None:
        """Show and handle model selection menu."""
        console.print("[cyan]Fetching available models...[/cyan]")

        available_models = await get_models_available(
            user_id=self.user_id,
            image_generation=False,
        )

        if not available_models:
            console.print("[red]No models available. Please check your API keys.[/red]")
            return

        console.print("\n[bold]Available models:[/bold]")
        for i, model in enumerate(available_models, 1):
            console.print(f"  {i}. {model.display_name} ({model.provider})")

        console.print("\n[dim]Enter number to select, or press Enter to cancel:[/dim]")

        try:
            selection_session: PromptSession[str] = PromptSession(message="> ")
            choice = await selection_session.prompt_async()
            choice = choice.strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        if not choice:
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected = available_models[idx]
                await set_active_model(interface=self.interface, model=selected)
                console.print(f"[green]Selected model: {selected.display_name} ({selected.provider})[/green]")
            else:
                console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

    async def _handle_imagine(self, prompt: str) -> None:
        """Handle image generation command."""
        if not prompt:
            console.print("[yellow]Please provide a prompt: /imagine <description>[/yellow]")
            return

        console.print(f"[cyan]Generating image: {prompt}[/cyan]")
        print("⏳ ", end="", flush=True)

        try:
            await handle_image_generation(prompt=prompt, interface=self.interface)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Image generation interrupted.[/yellow]")
            task_manager.kill_all_user_tasks(user_id=self.user_id)
        except BaseException as e:
            console.print(f"\n[red]Error generating image: {e}[/red]")
            logger.exception("Image generation error")

    async def _handle_info(self) -> None:
        """Show user and model info."""
        try:
            user_info = await get_info(user_id=self.user_id)
        except Exception:
            user_info = "N/A"

        # Get current model info (intentionally simple — fetches all models to find active)
        try:
            available_models = await get_models_available(
                user_id=self.user_id,
                image_generation=False,
            )
            active_model = None
            for model in available_models:
                if "🟢" in model.display_name:
                    active_model = f"{model.display_name} ({model.provider})"
                    break
            if not active_model and available_models:
                active_model = f"{available_models[0].display_name} ({available_models[0].provider})"
        except Exception:
            active_model = "N/A"

        console.print(f"\n[bold]User ID:[/bold] {self.user_id}")
        console.print(f"[bold]Active model:[/bold] {active_model}")
        console.print(f"[bold]User info:[/bold] {user_info}")
        console.print()

    async def _process_command(self, text: str) -> bool:
        """Process a command input.

        Returns:
            True if a command was processed, False otherwise.
        """
        text = text.strip()

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                console.print("[cyan]Goodbye![/cyan]")
                self._running = False
                return True
            elif cmd == "/help":
                await self._handle_help()
                return True
            elif cmd == "/reset":
                await self._handle_reset()
                return True
            elif cmd == "/model":
                await self._handle_model_selection()
                return True
            elif cmd == "/imagine":
                await self._handle_imagine(args)
                return True
            elif cmd == "/info":
                await self._handle_info()
                return True
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                console.print("[dim]Type /help for available commands.[/dim]")
                return True

        return False

    async def _process_message(self, text: str) -> None:
        """Process a regular message input."""
        # Set the message on the interface
        self.interface.set_last_message(text)

        # Show spinner
        print("⏳ ", end="", flush=True)

        try:
            await handle_user_prompt(interface=self.interface)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Operation interrupted.[/yellow]")
            task_manager.kill_all_user_tasks(user_id=self.user_id)
        except BaseException as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logger.exception("Error processing message")
        finally:
            # Clear last message after processing
            self.interface.set_last_message(None)

    async def run(self) -> None:
        """Run the REPL loop."""
        console.print("[bold cyan]Chibi Terminal REPL[/bold cyan]")
        console.print("[dim]Type /help for available commands, /quit to exit.[/dim]\n")

        session = self._create_session()

        while self._running:
            try:
                # Get input from user
                try:
                    text = await session.prompt_async()
                except KeyboardInterrupt:
                    console.print()
                    continue
                except EOFError:
                    console.print("\n[cyan]Goodbye![/cyan]")
                    break

                if not text.strip():
                    continue

                # Check if it's a command
                is_command = await self._process_command(text)

                if not is_command and self._running:
                    await self._process_message(text)

            except BaseException as e:
                logger.exception("Error in REPL loop")
                console.print(f"[red]Error: {e}[/red]")

        console.print("[dim]Session ended.[/dim]")


def setup_logging() -> None:
    """Setup logging for the terminal runner."""
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> <level>{message}</level>",
        level="INFO",
    )


async def run_chibi(user_id: int = 1) -> None:
    """Entry point for the terminal runner.

    Args:
        user_id: The user ID to use for this session. Default is 1.
    """
    setup_logging()

    try:
        from chibi.utils.app import log_application_settings

        log_application_settings()
    except Exception as e:
        logger.warning(f"Could not log application settings: {e}")

    runner = TerminalRunner(user_id=user_id)
    await runner.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chibi Terminal REPL")
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User ID for this session (default: 1)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_chibi(user_id=args.user_id))
    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
