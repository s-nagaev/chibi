"""CLI entrypoint for the IDE stdio runner."""

import asyncio


def run_ide() -> None:
    """Start the IDE JSONL runner using the project's async entrypoint convention."""
    import chibi.config  # noqa: F401
    from chibi.runners.ide_transport import IDEStdioRunner

    runner = IDEStdioRunner()
    exit_code = asyncio.run(runner.run())
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    run_ide()
