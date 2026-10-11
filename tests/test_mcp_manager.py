"""Unit tests for the MCPManager stdio connection flow with mocked SDK transports."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import ListToolsResult, Tool

from chibi.services.mcp.manager import MCPManager

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_manager_state():
    """Reset MCPManager class-level state around every test."""
    MCPManager._sessions.clear()
    MCPManager._server_tasks.clear()
    MCPManager._session_tools_map.clear()
    MCPManager._lock = asyncio.Lock()
    yield
    for task in MCPManager._server_tasks.values():
        task.cancel()
    MCPManager._sessions.clear()
    MCPManager._server_tasks.clear()
    MCPManager._session_tools_map.clear()
    MCPManager._lock = asyncio.Lock()


def _make_session() -> MagicMock:
    """Build a mock ClientSession with a successful initialize."""
    session = MagicMock(name="session")
    session.initialize = AsyncMock(return_value=None)
    return session


@asynccontextmanager
async def _fake_stdio_client(params, captured: list):
    """Fake stdio_client recording the server params and yielding mock streams."""
    captured.append(params)
    yield MagicMock(name="read_stream"), MagicMock(name="write_stream")


async def test_connect_stdio_initializes_and_registers_session():
    """connect_stdio must run the transport, initialize the session and register it."""
    captured: list = []
    session = _make_session()

    @asynccontextmanager
    async def fake_stdio_client(params):
        captured.append(params)
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.stdio_client", fake_stdio_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        connected = await MCPManager.connect_stdio("test-server", "echo", ["hello"], env={"FOO": "bar"})

    assert connected is session
    assert MCPManager.get_session("test-server") is session
    session.initialize.assert_awaited_once()
    client_session_factory.assert_called_once()

    assert len(captured) == 1
    server_params = captured[0]
    assert server_params.command == "echo"
    assert server_params.args == ["hello"]
    assert server_params.env == {"FOO": "bar"}


async def test_connect_stdio_then_list_tools():
    """Full connect -> initialize -> list_tools flow over the mocked stdio transport."""
    session = _make_session()
    session.list_tools = AsyncMock(
        return_value=ListToolsResult(tools=[Tool(name="echo", inputSchema={"type": "object"})])
    )

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.stdio_client", fake_stdio_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        await MCPManager.connect_stdio("test-server", "echo", [])

        result = await MCPManager.list_tools("test-server")

    assert [tool.name for tool in result.tools] == ["echo"]
    session.list_tools.assert_awaited_once()


async def test_connect_stdio_reuses_existing_session():
    """connect_stdio must return the already connected session without reconnecting."""
    session = _make_session()

    transport_calls: list = []

    @asynccontextmanager
    async def fake_stdio_client(params):
        transport_calls.append(params)
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.stdio_client", fake_stdio_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        first = await MCPManager.connect_stdio("test-server", "echo", [])
        second = await MCPManager.connect_stdio("test-server", "echo", [])

    assert first is second is session
    assert len(transport_calls) == 1
    client_session_factory.assert_called_once()


async def test_connect_stdio_raises_when_initialize_fails():
    """connect_stdio must fail cleanly when session initialization raises."""
    session = _make_session()
    session.initialize = AsyncMock(side_effect=RuntimeError("boom"))

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.stdio_client", fake_stdio_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        with pytest.raises(RuntimeError, match="Failed to initialize MCP server 'test-server'"):
            await MCPManager.connect_stdio("test-server", "echo", [])

    assert MCPManager.get_session("test-server") is None


async def test_disconnect_cleans_up_session_and_tools():
    """disconnect must cancel the lifecycle task, drop the session and return tools."""
    session = _make_session()

    @asynccontextmanager
    async def fake_stdio_client(params):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.stdio_client", fake_stdio_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        await MCPManager.connect_stdio("test-server", "echo", [])
        MCPManager.associate_tools_with_server("test-server", ["echo", "shout"])

        removed = await MCPManager.disconnect("test-server")

    assert removed == ["echo", "shout"]
    assert MCPManager.get_session("test-server") is None
    assert MCPManager.pop_server_tools("test-server") == []
