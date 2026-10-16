"""Unit and offline integration tests for the REST (streamable HTTP) MCP transport."""

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import TextContent, Tool

import chibi.services.providers.tools.mcp_management as mcp_management_module
from chibi.services.mcp.manager import MCPManager
from chibi.services.providers.tools.mcp_management import (
    DeinitializeMCPServer,
    InitializeRestMCPServer,
    register_tools_from_mcp_session,
)
from chibi.services.providers.tools.tool import RegisteredChibiTools

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


def _patch_rest_transport(captured: list):
    """Patch the REST transport in the manager module, recording factory calls."""
    session = _make_session()

    @asynccontextmanager
    async def fake_create_mcp_http_client(**kwargs):
        entry = {"client_kwargs": kwargs}
        captured.append(entry)
        http_client = MagicMock(name="http_client")
        entry["http_client"] = http_client
        yield http_client

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client=None, **kwargs):
        captured[-1]["url"] = url
        captured[-1]["transport_http_client"] = http_client
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    return (
        patch("chibi.services.mcp.manager.create_mcp_http_client", fake_create_mcp_http_client),
        patch("chibi.services.mcp.manager.streamable_http_client", fake_streamable_http_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    )


async def test_connect_rest_passes_url_and_headers_to_factories():
    """URL and composed headers must reach the http client factory and the transport."""
    captured: list = []
    patches = _patch_rest_transport(captured)

    with patches[0], patches[1], patches[2]:
        await MCPManager.connect_rest(
            "test-server", "http://localhost/jsonrpc", headers={"Authorization": "Bearer token"}
        )

    assert len(captured) == 1
    assert captured[0]["client_kwargs"] == {"headers": {"Authorization": "Bearer token"}}
    assert captured[0]["url"] == "http://localhost/jsonrpc"
    assert captured[0]["transport_http_client"] is captured[0]["http_client"]


async def test_connect_rest_enters_http_client_before_transport():
    """The httpx2 client must be entered into the exit stack before the transport."""
    events: list = []

    @asynccontextmanager
    async def fake_create_mcp_http_client(**kwargs):
        events.append("http_client_enter")
        yield MagicMock(name="http_client")
        events.append("http_client_exit")

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client=None, **kwargs):
        events.append("transport_enter")
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")
        events.append("transport_exit")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=_make_session())
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("chibi.services.mcp.manager.create_mcp_http_client", fake_create_mcp_http_client),
        patch("chibi.services.mcp.manager.streamable_http_client", fake_streamable_http_client),
        patch("chibi.services.mcp.manager.ClientSession", MagicMock(return_value=session_cm)),
    ):
        await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert events[:2] == ["http_client_enter", "transport_enter"]


async def test_connect_rest_initializes_and_registers_session():
    """connect_rest must run the transport, initialize the session and register it."""
    captured: list = []
    patches = _patch_rest_transport(captured)

    with patches[0], patches[1], patches[2]:
        connected = await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert connected is not None
    assert MCPManager.get_session("test-server") is connected


async def test_connect_rest_reuses_existing_session():
    """connect_rest must return the already connected session without reconnecting."""
    captured: list = []
    patches = _patch_rest_transport(captured)

    with patches[0], patches[1], patches[2]:
        first = await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")
        second = await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert first is second
    assert len(captured) == 1


async def test_connect_rest_raises_when_initialize_fails():
    """connect_rest must fail cleanly when session initialization raises."""
    session = _make_session()
    session.initialize = AsyncMock(side_effect=RuntimeError("boom"))

    @asynccontextmanager
    async def fake_create_mcp_http_client(**kwargs):
        yield MagicMock(name="http_client")

    @asynccontextmanager
    async def fake_streamable_http_client(url, http_client=None, **kwargs):
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("chibi.services.mcp.manager.create_mcp_http_client", fake_create_mcp_http_client),
        patch("chibi.services.mcp.manager.streamable_http_client", fake_streamable_http_client),
        patch("chibi.services.mcp.manager.ClientSession", MagicMock(return_value=session_cm)),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert MCPManager.get_session("test-server") is None


@asynccontextmanager
async def fake_create_mcp_http_client(**kwargs):
    """Module-level fake http client factory yielding a mock httpx2 client."""
    yield MagicMock(name="http_client")


@asynccontextmanager
async def fake_streamable_http_client(url, http_client=None, **kwargs):
    """Module-level fake streamable-HTTP transport yielding mock streams."""
    yield MagicMock(name="read_stream"), MagicMock(name="write_stream")


async def _run_rest_tool(**kwargs) -> MagicMock:
    """Run InitializeRestMCPServer.function with connect_rest mocked; return the mock."""
    connect_rest_mock: MagicMock = AsyncMock(return_value=MagicMock(name="session"))
    register_mock = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch.object(MCPManager, "connect_rest", connect_rest_mock),
        patch(
            "chibi.services.providers.tools.mcp_management.register_tools_from_mcp_session",
            register_mock,
        ),
    ):
        await InitializeRestMCPServer.function(server_name="test-server", url="http://localhost/jsonrpc", **kwargs)

    return connect_rest_mock


async def test_rest_tool_without_headers_passes_none():
    """No auth inputs at all must still connect with headers=None."""
    connect_rest_mock = await _run_rest_tool()

    connect_rest_mock.assert_awaited_once_with("test-server", "http://localhost/jsonrpc", headers=None)


async def test_rest_tool_plain_headers_pass_through(monkeypatch):
    """Plain headers must be forwarded as-is."""
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    connect_rest_mock = await _run_rest_tool(headers={"Authorization": "Bearer plain"})

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer plain"}


async def test_rest_tool_resolves_secret_envs(monkeypatch):
    """Bare secret_envs names must still be pulled from os.environ into the headers."""
    monkeypatch.setenv("MELIOUS_API_KEY", "supersecret")
    connect_rest_mock = await _run_rest_tool(secret_envs=["MELIOUS_API_KEY"])

    assert connect_rest_mock.await_args.kwargs["headers"] == {"MELIOUS_API_KEY": "supersecret"}


async def test_rest_tool_secret_template_falls_back_to_env_file(tmp_path, monkeypatch):
    """Template variables must fall back to the env file when absent from os.environ."""
    monkeypatch.delenv("MELIOUS_API_KEY", raising=False)
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    connect_rest_mock = await _run_rest_tool(
        secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
        env_file_path=str(env_file),
    )

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer file-key"}


async def test_rest_tool_env_file_is_variable_store_not_header_store(tmp_path, monkeypatch):
    """Underscore-style env-file entries must not be merged as literal headers."""
    monkeypatch.delenv("FILE_ONLY_VAR", raising=False)
    env_file = tmp_path / "vars.env"
    env_file.write_text("FILE_ONLY_VAR=file-value\nEMPTY_VAR=\n")

    connect_rest_mock = await _run_rest_tool(env_file_path=str(env_file))

    assert connect_rest_mock.await_args.kwargs["headers"] is None


async def test_rest_tool_env_file_header_like_entry_still_merged(tmp_path, monkeypatch):
    """Header-like env-file entries (hyphenated names) must stay merged for back-compat."""
    env_file = tmp_path / "headers.env"
    env_file.write_text("Authorization=Bearer from-file\nVAR_STYLE=dropped\n")

    connect_rest_mock = await _run_rest_tool(env_file_path=str(env_file))

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer from-file"}


async def test_rest_tool_precedence_headers_env_file_secrets(tmp_path, monkeypatch):
    """Precedence (lowest to highest): plain headers -> header-like env file -> secret envs."""
    monkeypatch.setenv("SHARED_SECRET", "from-env")
    env_file = tmp_path / "headers.env"
    env_file.write_text("X-File-Only=file-value\nSHARED_SECRET=from-file\n")

    connect_rest_mock = await _run_rest_tool(
        headers={"SHARED_SECRET": "from-plain", "PLAIN_ONLY": "plain-value"},
        secret_envs=["SHARED_SECRET"],
        env_file_path=str(env_file),
    )

    headers = connect_rest_mock.await_args.kwargs["headers"]
    assert headers == {
        "SHARED_SECRET": "from-file",
        "X-File-Only": "file-value",
        "PLAIN_ONLY": "plain-value",
    }


async def test_rest_tool_missing_secret_env_raises_valueerror(monkeypatch):
    """A secret_env variable missing from both sources must raise a clear ValueError."""
    monkeypatch.delenv("MISSING_SECRET", raising=False)

    with pytest.raises(ValueError, match="MISSING_SECRET"):
        await _run_rest_tool(secret_envs=["MISSING_SECRET"])


async def test_rest_tool_secret_template_file_wins_over_os_environ(tmp_path, monkeypatch):
    """Template variables must be resolved from the env file first, os.environ second."""
    monkeypatch.setenv("MELIOUS_API_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    connect_rest_mock = await _run_rest_tool(
        secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
        env_file_path=str(env_file),
    )

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer file-key"}


async def test_rest_tool_bare_secret_name_file_wins_over_os_environ(tmp_path, monkeypatch):
    """A bare secret_envs name must also prefer the env file over os.environ."""
    monkeypatch.setenv("MELIOUS_API_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    connect_rest_mock = await _run_rest_tool(
        secret_envs=["MELIOUS_API_KEY"],
        env_file_path=str(env_file),
    )

    assert connect_rest_mock.await_args.kwargs["headers"] == {"MELIOUS_API_KEY": "file-key"}


async def test_rest_tool_warning_on_divergent_env_and_file_values(tmp_path, monkeypatch):
    """A divergent runtime environment value must be shadowed by the env file with a warning."""
    monkeypatch.setenv("MELIOUS_API_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        connect_rest_mock = await _run_rest_tool(
            secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer file-key"}
    warnings = [m for m in messages if "MELIOUS_API_KEY" in m and "shadowed" in m]
    assert warnings, f"Expected a shadowing warning for MELIOUS_API_KEY, got: {messages}"


async def test_rest_tool_shadowing_warning_deduplicated_per_variable(tmp_path, monkeypatch):
    """One compose call must emit at most one shadowing warning per variable."""
    monkeypatch.setenv("MELIOUS_API_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        connect_rest_mock = await _run_rest_tool(
            secret_envs=[
                "X-A=Bearer ${MELIOUS_API_KEY}",
                "X-B=${MELIOUS_API_KEY}-${MELIOUS_API_KEY}",
            ],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    headers = connect_rest_mock.await_args.kwargs["headers"]
    assert headers == {
        "X-A": "Bearer file-key",
        "X-B": "file-key-file-key",
    }
    warnings = [m for m in messages if "MELIOUS_API_KEY" in m and "shadowed" in m]
    assert len(warnings) == 1, f"Expected exactly one shadowing warning, got: {messages}"


async def test_rest_tool_no_warning_when_env_and_file_values_identical(tmp_path, monkeypatch):
    """Identical values in both sources must not trigger the shadowing warning."""
    monkeypatch.setenv("MELIOUS_API_KEY", "same-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=same-key\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        connect_rest_mock = await _run_rest_tool(
            secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer same-key"}
    assert not [m for m in messages if "shadowed" in m]


async def test_rest_tool_no_secret_values_in_log_records_on_divergence(tmp_path, monkeypatch):
    """No log record produced by the divergence path may contain any secret value."""
    monkeypatch.setenv("MELIOUS_API_KEY", "supersecret-env-value")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=supersecret-file-value\n")

    messages: list = []
    sink_id = logger.add(messages.append, level="DEBUG")
    try:
        connect_rest_mock = await _run_rest_tool(
            secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
            env_file_path=str(env_file),
        )
    finally:
        logger.remove(sink_id)

    assert connect_rest_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer supersecret-file-value"}
    assert any("MELIOUS_API_KEY" in m for m in messages)
    for message in messages:
        assert "supersecret-env-value" not in message
        assert "supersecret-file-value" not in message


def _reload_mcp_management():
    """Reload the mcp_management module against the current application settings."""
    return importlib.reload(mcp_management_module)


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_rest_tool_register_gate_off_by_default():
    """With enable_mcp_rest off the REST tool must not register."""
    try:
        with patch(
            "chibi.config.application_settings",
            SimpleNamespace(enable_mcp_stdio=False, enable_mcp_sse=True, enable_mcp_rest=False),
        ):
            reloaded = _reload_mcp_management()
            assert reloaded.InitializeRestMCPServer.register is False
            assert reloaded.DeinitializeMCPServer.register is True
        assert "initialize_rest_mcp_server" not in RegisteredChibiTools.tools_map
    finally:
        _reload_mcp_management()
        RegisteredChibiTools.deregister_tools(["initialize_rest_mcp_server"])


@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_rest_tool_register_gate_on_with_flag():
    """With enable_mcp_rest on the REST tool must register and enable deinitialize."""
    try:
        with patch(
            "chibi.config.application_settings",
            SimpleNamespace(enable_mcp_stdio=False, enable_mcp_sse=False, enable_mcp_rest=True),
        ):
            reloaded = _reload_mcp_management()
            assert reloaded.InitializeRestMCPServer.register is True
            assert reloaded.DeinitializeMCPServer.register is True
            assert RegisteredChibiTools.tools_map["initialize_rest_mcp_server"] is reloaded.InitializeRestMCPServer
    finally:
        _reload_mcp_management()
        RegisteredChibiTools.deregister_tools(["initialize_rest_mcp_server"])


async def test_deinitialize_deregisters_rest_tool_names():
    """deinitialize_mcp_server must unregister mcp_{server}_rest_{tool} tools."""
    session = _make_session()
    session.list_tools = AsyncMock(
        return_value=MagicMock(
            tools=[Tool(name="echo", description="Echo", input_schema={"type": "object", "properties": {}})]
        )
    )
    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    chibi_tool_name = "mcp_test-server_rest_echo"
    try:
        with (
            patch("chibi.services.mcp.manager.create_mcp_http_client", fake_create_mcp_http_client),
            patch("chibi.services.mcp.manager.streamable_http_client", fake_streamable_http_client),
            patch("chibi.services.mcp.manager.ClientSession", MagicMock(return_value=session_cm)),
        ):
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")
            rest_session = MCPManager.get_session("test-server")
            assert rest_session is not None
            result = await register_tools_from_mcp_session(rest_session, server_name="test-server", transport="rest")

            assert result["status"] == "success"
            assert chibi_tool_name in RegisteredChibiTools.tools_map

            deinit_result = await DeinitializeMCPServer.function(server_name="test-server")

        assert deinit_result["status"] == "success"
        assert chibi_tool_name not in RegisteredChibiTools.tools_map
    finally:
        RegisteredChibiTools.deregister_tools([chibi_tool_name])


def _fake_jsonrpc_handler(methods: list) -> Callable[[httpx2.Request], httpx2.Response]:
    """Build a stateless plain-JSON JSON-RPC handler answering HTTP 201 / 202."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        method = body.get("method")
        request_id = body.get("id")
        methods.append(method)
        result: dict[str, object]

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fake-rest", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the message",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
        else:
            return httpx2.Response(202)

        return httpx2.Response(201, json={"jsonrpc": "2.0", "id": request_id, "result": result})

    return handler


async def test_offline_stateless_jsonrpc_server_end_to_end():
    """The real SDK streamable-HTTP path must work against a stateless plain-JSON server.

    The fake server answers with HTTP 201 (202 for notifications), plain
    application/json bodies and never sets Mcp-Session-Id.
    """
    methods: list = []
    handler = _fake_jsonrpc_handler(methods)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        async with streamable_http_client("http://fake.test/jsonrpc", http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.protocol_version == "2024-11-05"
                assert init.server_info.name == "fake-rest"

                tools = await session.list_tools()
                assert [tool.name for tool in tools.tools] == ["echo"]

                call = await session.call_tool("echo", {"message": "ping"})
                assert call.is_error is False
                first_content = call.content[0]
                assert isinstance(first_content, TextContent)
                assert first_content.text == "pong"

    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]


async def test_offline_register_tools_from_rest_session():
    """Tools from the offline REST session must register as mcp_{server}_rest_{tool}."""
    methods: list = []
    handler = _fake_jsonrpc_handler(methods)
    chibi_tool_name = "mcp_fake-rest_rest_echo"

    try:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            async with streamable_http_client("http://fake.test/jsonrpc", http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await register_tools_from_mcp_session(session, server_name="fake-rest", transport="rest")

        assert result["status"] == "success"
        assert chibi_tool_name in RegisteredChibiTools.tools_map
        assert result["message"].endswith(chibi_tool_name)
    finally:
        RegisteredChibiTools.deregister_tools([chibi_tool_name])


def _hooked_http_client() -> tuple[MagicMock, dict[str, list]]:
    """Build a mock http client with a real response event-hooks dict.

    Returns:
        The mock client and the live hooks dict the manager registers its
        recorder into.
    """
    hooks: dict[str, list] = {"response": []}
    client = MagicMock(name="http_client")
    client.event_hooks = hooks
    return client, hooks


async def _fire_response_hooks(
    hooks: dict[str, list],
    status_code: int,
    content_type: str = "application/json",
    body: bytes = b"",
) -> SimpleNamespace:
    """Invoke registered response hooks with a synthetic response.

    Args:
        hooks: The live event-hooks dict of a mock http client.
        status_code: HTTP status code of the synthetic response.
        content_type: Content type of the synthetic response.
        body: Body returned by the synthetic response's aread().

    Returns:
        The synthetic response object, so tests can assert on aread() calls.
    """
    response = SimpleNamespace(
        status_code=status_code,
        headers={"content-type": content_type},
        aread=AsyncMock(return_value=body),
    )
    for hook in hooks["response"]:
        await hook(response)
    return response


def _failing_rest_transport(error: BaseException, before_raise: Callable[[], Any] | None = None) -> type:
    """Build a fake streamable-HTTP transport class that raises after optional preparation.

    Args:
        error: The exception to raise on context entry.
        before_raise: Optional zero-argument async callable executed before
            raising (used to fire response hooks first).

    Returns:
        A transport class compatible with the manager's REST factory.
    """

    class FailingStreamableHttpClient:
        def __call__(self, url: str, http_client: Any = None, **kwargs: Any) -> "FailingStreamableHttpClient":
            return self

        async def __aenter__(self) -> tuple[MagicMock, MagicMock]:
            if before_raise is not None:
                await before_raise()
            raise error

        async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> bool:
            return False

    return FailingStreamableHttpClient


@asynccontextmanager
async def _static_http_client_factory(client: Any, **kwargs: Any):
    """Fake create_mcp_http_client always yielding the given client."""
    yield client


PLACEHOLDER_CODE = -32603
PLACEHOLDER_MESSAGE = "Server returned an error response"


async def test_connect_rest_enriches_placeholder_with_http_status_and_server_error():
    """A placeholder raised after an HTTP 401 must report the status and the server error."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    transport = _failing_rest_transport(
        error,
        before_raise=lambda: _fire_response_hooks(
            hooks, 401, "application/json", b'{"code": "AUTH_1008", "message": "Invalid API key format"}'
        ),
    )

    messages: list = []
    sink_id = logger.add(messages.append, level="ERROR")
    try:
        with (
            patch(
                "chibi.services.mcp.manager.create_mcp_http_client",
                lambda **kwargs: _static_http_client_factory(client, **kwargs),
            ),
            patch("chibi.services.mcp.manager.streamable_http_client", transport()),
        ):
            with pytest.raises(MCPError) as excinfo:
                await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")
    finally:
        logger.remove(sink_id)

    message = str(excinfo.value)
    assert excinfo.value.code == PLACEHOLDER_CODE
    assert "HTTP 401" in message
    assert "AUTH_1008" in message
    assert "Invalid API key format" in message
    assert any("HTTP 401" in m and "AUTH_1008" in m for m in messages), f"Expected enriched log line, got: {messages}"


async def test_connect_rest_preserves_error_type_and_data_when_enriching():
    """Enrichment must keep the exception type, JSON-RPC code and data intact."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE, data={"detail": "context"})
    transport = _failing_rest_transport(
        error, before_raise=lambda: _fire_response_hooks(hooks, 403, "application/json", b'{"code": "X"}')
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert isinstance(excinfo.value, MCPError)
    assert excinfo.value.code == PLACEHOLDER_CODE
    assert excinfo.value.data == {"detail": "context"}


async def test_connect_rest_redacts_bearer_token_from_error_body():
    """A server echoing a Bearer token back in the error body must not leak it."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    transport = _failing_rest_transport(
        error,
        before_raise=lambda: _fire_response_hooks(
            hooks, 401, "application/json", b'{"error": "Bearer sk-super-secret-token rejected"}'
        ),
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    message = str(excinfo.value)
    assert "sk-super-secret-token" not in message
    assert "Bearer [REDACTED]" in message


async def test_connect_rest_truncates_long_error_body_snippet():
    """The body-derived detail must be bounded to a short snippet."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    long_message = "x" * 500
    transport = _failing_rest_transport(
        error,
        before_raise=lambda: _fire_response_hooks(
            hooks, 500, "application/json", json.dumps({"message": long_message}).encode()
        ),
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    message = str(excinfo.value)
    assert "x" * 200 in message
    assert "x" * 201 not in message


async def test_connect_rest_without_recorded_outcome_keeps_placeholder_message():
    """No recorded HTTP failure means the placeholder message stays untouched."""
    client, _hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    transport = _failing_rest_transport(error)

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert str(excinfo.value) == PLACEHOLDER_MESSAGE


async def test_connect_rest_leaves_non_placeholder_mcp_error_untouched():
    """Other -32603 errors must not be rewritten even when an HTTP failure was recorded."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message="Custom tool failure")
    transport = _failing_rest_transport(
        error, before_raise=lambda: _fire_response_hooks(hooks, 401, "application/json", b'{"code": "AUTH"}')
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert str(excinfo.value) == "Custom tool failure"


async def test_connect_rest_group_wrapped_placeholder_is_enriched():
    """An ExceptionGroup-wrapped placeholder must still be unwrapped and enriched."""
    client, hooks = _hooked_http_client()
    inner = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    error = ExceptionGroup("Unhandled errors in a TaskGroup", [inner])
    transport = _failing_rest_transport(
        error,
        before_raise=lambda: _fire_response_hooks(
            hooks, 401, "application/json", b'{"code": "AUTH_1008", "message": "Invalid API key format"}'
        ),
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert excinfo.value.code == PLACEHOLDER_CODE
    assert "HTTP 401" in str(excinfo.value)
    assert "AUTH_1008" in str(excinfo.value)


async def test_connect_rest_success_with_hook_attached_unaffected():
    """Successful responses must not be buffered and must not alter a successful connect."""
    client, hooks = _hooked_http_client()
    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=_make_session())
    session_cm.__aexit__ = AsyncMock(return_value=None)

    @asynccontextmanager
    async def successful_transport(url: str, http_client: Any = None, **kwargs: Any):
        response = await _fire_response_hooks(hooks, 201, "application/json", b'{"result": {}}')
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")
        assert response.aread.await_count == 0

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", successful_transport),
        patch("chibi.services.mcp.manager.ClientSession", MagicMock(return_value=session_cm)),
    ):
        connected = await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert connected is not None
    assert MCPManager.get_session("test-server") is connected


async def test_connect_rest_does_not_buffer_sse_error_body():
    """A >= 400 text/event-stream response must never be buffered."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    transport = _failing_rest_transport(
        error, before_raise=lambda: _fire_response_hooks(hooks, 401, "text/event-stream", b"data: secret\n\n")
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    message = str(excinfo.value)
    assert "HTTP 401" in message
    assert "secret" not in message


async def test_connect_rest_binary_error_body_records_status_only():
    """A non-textual error body must contribute the status code only."""
    client, hooks = _hooked_http_client()
    error = MCPError(code=PLACEHOLDER_CODE, message=PLACEHOLDER_MESSAGE)
    transport = _failing_rest_transport(
        error, before_raise=lambda: _fire_response_hooks(hooks, 502, "application/octet-stream", b"\x00\x01\x02")
    )

    with (
        patch(
            "chibi.services.mcp.manager.create_mcp_http_client",
            lambda **kwargs: _static_http_client_factory(client, **kwargs),
        ),
        patch("chibi.services.mcp.manager.streamable_http_client", transport()),
    ):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://localhost/jsonrpc")

    assert str(excinfo.value) == f"{PLACEHOLDER_MESSAGE} (HTTP 502)"


async def test_connect_rest_surfaces_jsonrpc_error_body_at_http_400():
    """A spec-correct JSON-RPC error body at HTTP 400 must surface verbatim, with the hook attached."""
    created_clients: list[httpx2.AsyncClient] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        return httpx2.Response(
            400,
            json={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32602, "message": "Invalid params"},
            },
        )

    @asynccontextmanager
    async def fake_create_mcp_http_client(**kwargs: Any):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)
        created_clients.append(client)
        yield client

    with patch("chibi.services.mcp.manager.create_mcp_http_client", fake_create_mcp_http_client):
        with pytest.raises(MCPError) as excinfo:
            await MCPManager.connect_rest("test-server", "http://fake.test/jsonrpc")

    assert excinfo.value.code == -32602
    assert excinfo.value.message == "Invalid params"
    assert created_clients, "The transport must receive the client built by the factory"
    assert created_clients[0].event_hooks["response"], "The recorder hook must be attached to the client"
