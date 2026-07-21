"""CLI entrypoint for the IDE stdio runner."""

import asyncio
import os

# When launched as an IDE stdio server, filesystem access and MCP stdio are
# expected to be enabled by default. setdefault() keeps explicit env overrides
# or config file values intact.
os.environ.setdefault("FILESYSTEM_ACCESS", "true")
os.environ.setdefault("ENABLE_MCP_STDIO", "true")


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
