"""Unit tests for the MCPManager stdio connection flow with mocked SDK transports."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from mcp.types import ListToolsResult, Tool

from chibi.services.mcp.manager import MCPManager
from chibi.services.providers.tools.mcp_management import InitializeStdioMCPServer

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_manager_state():
    """Reset MCPManager class-level state around every test."""
    MCPManager._sessions.clear()
    MCPManager._server_tasks.clear()
    MCPManager._session_errors.clear()
    MCPManager._session_tools_map.clear()
    MCPManager._lock = asyncio.Lock()
    yield
    for task in MCPManager._server_tasks.values():
        task.cancel()
    MCPManager._sessions.clear()
    MCPManager._server_tasks.clear()
    MCPManager._session_errors.clear()
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
    """connect_stdio must surface the underlying initialization error, not a generic one."""
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
        with pytest.raises(RuntimeError, match="boom"):
            await MCPManager.connect_stdio("test-server", "echo", [])

    assert MCPManager.get_session("test-server") is None
    assert "test-server" not in MCPManager._server_tasks
    assert "test-server" not in MCPManager._session_errors


async def test_connect_stdio_unwraps_exception_group_from_transport():
    """A failing stdio transport wrapped in an ExceptionGroup must surface its cause."""
    inner = RuntimeError("stdio process exited with code 1")

    class FailingStdioClient:
        def __call__(self, params):
            return self

        async def __aenter__(self):
            raise ExceptionGroup("Unhandled errors in a TaskGroup", [inner])

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    with patch("chibi.services.mcp.manager.stdio_client", FailingStdioClient()):
        with pytest.raises(RuntimeError, match="stdio process exited with code 1"):
            await MCPManager.connect_stdio("test-server", "echo", [])

    assert MCPManager.get_session("test-server") is None


async def test_connect_sse_unwraps_exception_group_from_transport():
    """A failing SSE transport wrapped in an ExceptionGroup must surface its cause."""
    inner = RuntimeError("Server returned an error response")

    class FailingSseClient:
        def __call__(self, url, headers=None, **kwargs):
            return self

        async def __aenter__(self):
            raise ExceptionGroup("Unhandled errors in a TaskGroup", [inner])

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    with patch("chibi.services.mcp.manager.sse_client", FailingSseClient()):
        with pytest.raises(RuntimeError, match="Server returned an error response"):
            await MCPManager.connect_sse("test-server", "http://localhost/sse")

    assert MCPManager.get_session("test-server") is None


async def test_connect_rest_unwraps_exception_group_from_transport():
    """A failing REST transport wrapped in an ExceptionGroup must surface its cause."""
    inner = RuntimeError("401 Unauthorized")

    @asynccontextmanager
    async def failing_http_client(**kwargs):
        yield MagicMock(name="http_client")

    class FailingStreamableHttpClient:
        def __call__(self, url, http_client=None, **kwargs):
            return self

        async def __aenter__(self):
            raise ExceptionGroup("Unhandled errors in a TaskGroup", [inner])

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    with (
        patch("chibi.services.mcp.manager.create_mcp_http_client", failing_http_client),
        patch("chibi.services.mcp.manager.streamable_http_client", FailingStreamableHttpClient()),
    ):
        with pytest.raises(RuntimeError, match="401 Unauthorized"):
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert MCPManager.get_session("test-server") is None


async def test_connect_stdio_timeout_falls_back_to_generic_error():
    """A hung transport must fall back to a timeout error after the deadline."""
    never_ready = asyncio.Event()

    @asynccontextmanager
    async def hanging_stdio_client(params):
        await never_ready.wait()
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    with patch("chibi.services.mcp.manager.stdio_client", hanging_stdio_client):
        with pytest.raises(asyncio.TimeoutError):
            await MCPManager.connect_stdio("test-server", "echo", [], timeout=0.1)

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


async def test_successful_connect_stdio_keeps_lifecycle_task_for_disconnect():
    """A successful connect must keep the lifecycle task registered so disconnect can cancel it."""
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

        lifecycle_task = MCPManager._server_tasks.get("test-server")
        assert lifecycle_task is not None
        assert not lifecycle_task.done()

        await MCPManager.disconnect("test-server")

        assert lifecycle_task.cancelled()
        assert "test-server" not in MCPManager._server_tasks
        assert MCPManager.get_session("test-server") is None


async def test_successful_connect_sse_keeps_lifecycle_task_for_disconnect():
    """A successful SSE connect must keep the lifecycle task registered so disconnect can cancel it."""
    session = _make_session()

    @asynccontextmanager
    async def fake_sse_client(url, headers=None, **kwargs):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.sse_client", fake_sse_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        await MCPManager.connect_sse("test-server", "http://localhost/sse")

        lifecycle_task = MCPManager._server_tasks.get("test-server")
        assert lifecycle_task is not None

        await MCPManager.disconnect("test-server")

        assert lifecycle_task.cancelled()
        assert "test-server" not in MCPManager._server_tasks


async def test_successful_connect_rest_keeps_lifecycle_task_for_disconnect():
    """A successful REST connect must keep the lifecycle task registered so disconnect can cancel it."""
    session = _make_session()

    @asynccontextmanager
    async def fake_http_client(**kwargs):
        yield MagicMock(name="http_client")

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client=None, **kwargs):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    with (
        patch("chibi.services.mcp.manager.create_mcp_http_client", fake_http_client),
        patch("chibi.services.mcp.manager.streamable_http_client", fake_streamable_http_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    ):
        await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

        lifecycle_task = MCPManager._server_tasks.get("test-server")
        assert lifecycle_task is not None

        await MCPManager.disconnect("test-server")

        assert lifecycle_task.cancelled()
        assert "test-server" not in MCPManager._server_tasks


async def _run_stdio_tool(**kwargs) -> MagicMock:
    """Run InitializeStdioMCPServer.function with connect_stdio mocked; return the mock."""
    connect_stdio_mock = AsyncMock(return_value=MagicMock(name="session"))
    register_mock = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch.object(MCPManager, "connect_stdio", connect_stdio_mock),
        patch(
            "chibi.services.providers.tools.mcp_management.register_tools_from_mcp_session",
            register_mock,
        ),
    ):
        await InitializeStdioMCPServer.function(server_name="test-server", command="echo", args=["hello"], **kwargs)

    return connect_stdio_mock


async def test_stdio_tool_bare_secret_env_copied_to_child_env(monkeypatch):
    """A bare secret_envs name must still be copied from os.environ into the child env."""
    monkeypatch.setenv("STDIO_SECRET", "supersecret")
    connect_stdio_mock = await _run_stdio_tool(secret_envs=["STDIO_SECRET"])

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["STDIO_SECRET"] == "supersecret"


async def test_stdio_tool_secret_template_falls_back_to_env_file(tmp_path, monkeypatch):
    """Template variables for stdio must fall back to the env file."""
    monkeypatch.delenv("FILE_KEY", raising=False)
    env_file = tmp_path / "secrets.env"
    env_file.write_text("FILE_KEY=file-key\n")

    connect_stdio_mock = await _run_stdio_tool(
        secret_envs=["AUTH_HEADER=Bearer ${FILE_KEY}"],
        env_file_path=str(env_file),
    )

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["AUTH_HEADER"] == "Bearer file-key"


async def test_stdio_tool_env_file_merged_into_child_env(tmp_path):
    """The env file must remain a variable store merged into the child process env."""
    env_file = tmp_path / "vars.env"
    env_file.write_text("CHILD_VAR=child-value\n")

    connect_stdio_mock = await _run_stdio_tool(env_file_path=str(env_file))

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["CHILD_VAR"] == "child-value"


async def test_stdio_tool_missing_secret_env_raises_valueerror(monkeypatch):
    """A stdio secret_envs variable missing from both sources must raise ValueError."""
    monkeypatch.delenv("MISSING_SECRET", raising=False)

    with pytest.raises(ValueError, match="MISSING_SECRET"):
        await _run_stdio_tool(secret_envs=["MISSING_SECRET"])


async def test_stdio_tool_secret_template_file_wins_over_os_environ(tmp_path, monkeypatch):
    """Template variables for stdio must be resolved from the env file first, os.environ second."""
    monkeypatch.setenv("FILE_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("FILE_KEY=file-key\n")

    connect_stdio_mock = await _run_stdio_tool(
        secret_envs=["AUTH_HEADER=Bearer ${FILE_KEY}"],
        env_file_path=str(env_file),
    )

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["AUTH_HEADER"] == "Bearer file-key"


async def test_stdio_tool_bare_secret_name_file_wins_over_os_environ(tmp_path, monkeypatch):
    """A bare secret_envs name for stdio must also prefer the env file over os.environ."""
    monkeypatch.setenv("STDIO_SECRET", "env-secret")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("STDIO_SECRET=file-secret\n")

    connect_stdio_mock = await _run_stdio_tool(
        secret_envs=["STDIO_SECRET"],
        env_file_path=str(env_file),
    )

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["STDIO_SECRET"] == "file-secret"


async def test_stdio_tool_warning_on_divergent_env_and_file_values(tmp_path, monkeypatch):
    """A divergent runtime environment value must be shadowed by the env file with a warning."""
    monkeypatch.setenv("FILE_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("FILE_KEY=file-key\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        connect_stdio_mock = await _run_stdio_tool(
            secret_envs=["AUTH_HEADER=Bearer ${FILE_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["AUTH_HEADER"] == "Bearer file-key"
    warnings = [m for m in messages if "FILE_KEY" in m and "shadowed" in m]
    assert warnings, f"Expected a shadowing warning for FILE_KEY, got: {messages}"


async def test_stdio_tool_no_warning_when_env_and_file_values_identical(tmp_path, monkeypatch):
    """Identical values in both sources must not trigger the shadowing warning."""
    monkeypatch.setenv("FILE_KEY", "same-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("FILE_KEY=same-key\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        connect_stdio_mock = await _run_stdio_tool(
            secret_envs=["AUTH_HEADER=Bearer ${FILE_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["AUTH_HEADER"] == "Bearer same-key"
    assert not [m for m in messages if "shadowed" in m]


async def test_stdio_tool_no_secret_values_in_log_records_on_divergence(tmp_path, monkeypatch):
    """No log record produced by the divergence path may contain any secret value."""
    monkeypatch.setenv("FILE_KEY", "supersecret-env-value")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("FILE_KEY=supersecret-file-value\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="DEBUG")
    try:
        connect_stdio_mock = await _run_stdio_tool(
            secret_envs=["AUTH_HEADER=Bearer ${FILE_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    env = connect_stdio_mock.await_args.kwargs["env"]
    assert env["AUTH_HEADER"] == "Bearer supersecret-file-value"
    assert any("FILE_KEY" in m for m in messages)
    for message in messages:
        assert "supersecret-env-value" not in message
        assert "supersecret-file-value" not in message
