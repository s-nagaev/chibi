import asyncio
from contextlib import AsyncExitStack
from typing import Any, Callable

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from chibi.services.task_manager import task_manager


class MCPManager:
    """Manages the lifecycle of MCP server connections and sessions.

    Attributes:
        _sessions: Active MCP sessions.
        _server_tasks: Background tasks maintaining the connections.
        _lock: Lock for thread-safe session management.
    """

    _sessions: dict[str, ClientSession] = {}
    _server_tasks: dict[str, asyncio.Task] = {}
    _lock: asyncio.Lock = asyncio.Lock()
    _session_tools_map: dict[str, list[str]] = {}

    @classmethod
    async def _session_lifecycle(
        cls,
        name: str,
        transport_factory: Callable[[AsyncExitStack], Any],
        ready_event: asyncio.Event,
        init_timeout: float,
    ) -> None:
        """Lifecycle manager for an MCP session running in a background task."""
        async with AsyncExitStack() as stack:
            try:
                # Initialize transport
                read, write = await transport_factory(stack)

                # Initialize session
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=init_timeout)

                # Register session
                cls._sessions[name] = session
                logger.log("TOOL", f"MCP server '{name}' connected and initialized.")
                ready_event.set()

                # Keep session alive until cancelled
                # We use a Future that never completes to hang here efficiently
                await asyncio.Future()

            except asyncio.CancelledError:
                logger.log("TOOL", f"MCP server '{name}' session cancelled, cleaning up...")
                raise
            except Exception as e:
                logger.error(f"Error in MCP server '{name}' lifecycle: {e}")
                # If we failed during init, we must signal ready_event so the waiter doesn't hang forever
                if not ready_event.is_set():
                    ready_event.set()
                raise
            finally:
                # Cleanup global registry
                if name in cls._sessions:
                    cls._sessions.pop(name, None)
                logger.log("TOOL", f"MCP server '{name}' disconnected.")

    @classmethod
    async def connect_stdio(
        cls, name: str, command: str, args: list[str], env: dict[str, str] | None = None, timeout: float = 20.0
    ) -> ClientSession:
        """Connect to an MCP server via stdio transport.

        Args:
            name: Unique name for the server session.
            command: Command to execute.
            args: Arguments for the command.
            env: Environment variables.
            timeout: Connection timeout in seconds.

        Returns:
            The connected ClientSession.

        Raises:
            RuntimeError: If connection fails or times out.
        """
        async with cls._lock:
            if name in cls._sessions:
                return cls._sessions[name]

            logger.log("TOOL", f"Connecting to MCP server '{name}' via stdio: {command} {' '.join(args)}")

            async def stdio_factory(stack: AsyncExitStack):
                server_params = StdioServerParameters(command=command, args=args, env=env)
                return await stack.enter_async_context(stdio_client(server_params))

            ready_event = asyncio.Event()

            task = task_manager.run_task(cls._session_lifecycle(name, stdio_factory, ready_event, timeout))

            if not task:
                raise RuntimeError("Failed to schedule MCP connection task")

            cls._server_tasks[name] = task

            try:
                await asyncio.wait_for(ready_event.wait(), timeout=timeout + 5.0)
            except Exception:
                task.cancel()
                cls._server_tasks.pop(name, None)
                raise

            session = cls._sessions.get(name)
            if not session:
                cls._server_tasks.pop(name, None)
                raise RuntimeError(f"Failed to initialize MCP server '{name}'")

            return session

    @classmethod
    async def connect_sse(cls, name: str, url: str, timeout: float = 20.0) -> ClientSession:
        """Connect to an MCP server via SSE transport.

        Args:
            name: Unique name for the server session.
            url: SSE endpoint URL.
            timeout: Connection timeout in seconds.

        Returns:
            The connected ClientSession.

        Raises:
            RuntimeError: If connection fails or times out.
        """
        async with cls._lock:
            if name in cls._sessions:
                return cls._sessions[name]

            logger.log("TOOL", f"Connecting to MCP server '{name}' via SSE: {url}")

            async def sse_factory(stack: AsyncExitStack):
                return await stack.enter_async_context(sse_client(url))

            ready_event = asyncio.Event()

            task = task_manager.run_task(cls._session_lifecycle(name, sse_factory, ready_event, timeout))

            if not task:
                raise RuntimeError("Failed to schedule MCP connection task")

            cls._server_tasks[name] = task

            try:
                await asyncio.wait_for(ready_event.wait(), timeout=timeout + 5.0)
            except Exception:
                task.cancel()
                cls._server_tasks.pop(name, None)
                raise

            session = cls._sessions.get(name)
            if not session:
                cls._server_tasks.pop(name, None)
                raise RuntimeError(f"Failed to initialize MCP server '{name}'")

            return session

    @classmethod
    def associate_tools_with_server(cls, server_name: str, tool_names: list[str]) -> None:
        """Register tools associated with a specific server session."""
        cls._session_tools_map[server_name] = tool_names

    @classmethod
    def pop_server_tools(cls, server_name: str) -> list[str]:
        """Remove and return tools associated with a specific server session."""
        if server_name not in cls._session_tools_map:
            logger.warning(f"No Tools registered for server {server_name}. Nothing to deregister.")
            return []
        removed_tools = cls._session_tools_map.pop(server_name)
        return removed_tools

    @classmethod
    async def disconnect(cls, name: str) -> list[str]:
        """Disconnects and cleans up an MCP session."""
        if name not in cls._sessions and name not in cls._server_tasks:
            return []

        logger.log("TOOL", f"Disconnecting from MCP server '{name}'")
        async with cls._lock:
            # Cancel the background task
            task = cls._server_tasks.pop(name, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error during MCP server '{name}' disconnect: {e}")

            # Session removal from _sessions happens in _session_lifecycle finally block,
            # but we can double check here or just trust the flow.
            # However, we need to return deregistered tools immediately.
            deregistered_tools = cls.pop_server_tools(name)

            # Ensure session is gone (in case task died before we cancelled)
            cls._sessions.pop(name, None)

            return deregistered_tools

    @classmethod
    def get_session(cls, name: str) -> ClientSession | None:
        """Get an active session by name."""
        return cls._sessions.get(name)

    @classmethod
    async def list_tools(cls, name: str):
        """List tools available on the specified server."""
        session = cls.get_session(name)
        if not session:
            raise ValueError(f"No active session for MCP server: {name}")
        return await session.list_tools()

    @classmethod
    async def call_tool(
        cls, server_name: str, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0
    ) -> CallToolResult:
        """Call a tool on the specified server.

        Args:
            server_name: The server session name.
            tool_name: The tool to call.
            arguments: Tool arguments.
            timeout: Execution timeout in seconds.

        Returns:
            The tool execution result.
        """
        session = cls.get_session(server_name)
        if not session:
            raise ValueError(f"No active session for MCP server: {server_name}")
        return await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)
