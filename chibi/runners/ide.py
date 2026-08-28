"""CLI entrypoint for the IDE stdio runner."""

import asyncio
import io
import os
import sys
from typing import cast

# When launched as an IDE stdio server, filesystem access and MCP stdio are
# expected to be enabled by default. setdefault() keeps explicit env overrides
# or config file values intact.
os.environ.setdefault("FILESYSTEM_ACCESS", "true")
os.environ.setdefault("ENABLE_MCP_STDIO", "true")


def run_ide() -> None:
    """Start the IDE JSONL runner using the project's async entrypoint convention."""
    from chibi.config.logging import use_stderr_logging
    from chibi.utils.encoding import reconfigure_stream_utf8

    reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stdout))
    reconfigure_stream_utf8(cast(io.TextIOWrapper, sys.stderr))

    # stdout is the JSONL protocol channel: route loguru to stderr before any
    # further application code can log.
    use_stderr_logging()

    import chibi.config  # noqa: F401
    from chibi.runners.ide_transport import IDEStdioRunner

    runner = IDEStdioRunner()
    exit_code = asyncio.run(runner.run())
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    run_ide()
