"""Unit tests for SSE MCP auth: header composition and transport pass-through."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chibi.services.mcp.manager import MCPManager
from chibi.services.providers.tools.mcp_management import InitializeSseMCPServer

pytestmark = pytest.mark.asyncio


def _make_session() -> MagicMock:
    """Build a mock ClientSession with a successful initialize."""
    session = MagicMock(name="session")
    session.initialize = AsyncMock(return_value=None)
    return session


def _patch_sse_transport(captured: list):
    """Patch sse_client and ClientSession in the manager module, recording sse_client calls."""
    session = _make_session()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_sse_client(url, headers=None, **kwargs):
        captured.append({"url": url, "headers": headers})
        yield MagicMock(name="read_stream"), MagicMock(name="write_stream")

    session_cm = MagicMock(name="session_cm")
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    client_session_factory = MagicMock(return_value=session_cm)

    return (
        patch("chibi.services.mcp.manager.sse_client", fake_sse_client),
        patch("chibi.services.mcp.manager.ClientSession", client_session_factory),
    )


async def test_connect_sse_without_headers_keeps_back_compat():
    """connect_sse without headers must behave exactly like before: headers=None."""
    captured: list = []
    sse_patch, session_patch = _patch_sse_transport(captured)

    with sse_patch, session_patch:
        await MCPManager.connect_sse("test-server", "http://localhost/sse")

    assert len(captured) == 1
    assert captured[0]["url"] == "http://localhost/sse"
    assert captured[0]["headers"] is None


async def test_connect_sse_passes_plain_headers_to_sse_client():
    """connect_sse must forward plain headers to sse_client."""
    captured: list = []
    sse_patch, session_patch = _patch_sse_transport(captured)

    with sse_patch, session_patch:
        await MCPManager.connect_sse("test-server", "http://localhost/sse", headers={"Authorization": "Bearer token"})

    assert captured[0]["headers"] == {"Authorization": "Bearer token"}


async def _run_sse_tool(**kwargs) -> MagicMock:
    """Run InitializeSseMCPServer.function with connect_sse mocked; return the mock."""
    connect_sse_mock = AsyncMock(return_value=MagicMock(name="session"))
    register_mock = AsyncMock(return_value={"status": "success", "message": "ok"})

    with (
        patch.object(MCPManager, "connect_sse", connect_sse_mock),
        patch(
            "chibi.services.providers.tools.mcp_management.register_tools_from_mcp_session",
            register_mock,
        ),
    ):
        await InitializeSseMCPServer.function(server_name="test-server", url="http://localhost/sse", **kwargs)

    return connect_sse_mock


async def test_sse_tool_without_headers_passes_none():
    """No auth inputs at all must still connect with headers=None."""
    connect_sse_mock = await _run_sse_tool()

    connect_sse_mock.assert_awaited_once_with("test-server", "http://localhost/sse", headers=None)


async def test_sse_tool_plain_headers_pass_through(monkeypatch):
    """Plain headers must be forwarded as-is."""
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    connect_sse_mock = await _run_sse_tool(headers={"Authorization": "Bearer plain"})

    assert connect_sse_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer plain"}


async def test_sse_tool_resolves_secret_envs(monkeypatch):
    """Bare secret_envs names must still be pulled from os.environ into the headers."""
    monkeypatch.setenv("MELIOUS_API_KEY", "supersecret")
    connect_sse_mock = await _run_sse_tool(secret_envs=["MELIOUS_API_KEY"])

    assert connect_sse_mock.await_args.kwargs["headers"] == {"MELIOUS_API_KEY": "supersecret"}


async def test_sse_tool_secret_template_from_os_environ(monkeypatch):
    """A template entry must produce a substituted header from os.environ."""
    monkeypatch.setenv("MELIOUS_API_KEY", "supersecret")
    connect_sse_mock = await _run_sse_tool(secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"])

    assert connect_sse_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer supersecret"}


async def test_sse_tool_secret_template_falls_back_to_env_file(tmp_path, monkeypatch):
    """Template variables must fall back to the env file when absent from os.environ."""
    monkeypatch.delenv("MELIOUS_API_KEY", raising=False)
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    connect_sse_mock = await _run_sse_tool(
        secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
        env_file_path=str(env_file),
    )

    assert connect_sse_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer file-key"}


async def test_sse_tool_secret_template_prefers_os_environ_over_file(tmp_path, monkeypatch):
    """Template variables must be resolved from os.environ first, env file second."""
    monkeypatch.setenv("MELIOUS_API_KEY", "env-key")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("MELIOUS_API_KEY=file-key\n")

    connect_sse_mock = await _run_sse_tool(
        secret_envs=["Authorization=Bearer ${MELIOUS_API_KEY}"],
        env_file_path=str(env_file),
    )

    assert connect_sse_mock.await_args.kwargs["headers"] == {"Authorization": "Bearer env-key"}


async def test_sse_tool_secret_template_multiple_variables(tmp_path, monkeypatch):
    """Multiple ${VAR} occurrences in one template must all be substituted."""
    monkeypatch.setenv("USER_NAME", "alice")
    env_file = tmp_path / "secrets.env"
    env_file.write_text("API_TOKEN=file-token\n")

    connect_sse_mock = await _run_sse_tool(
        secret_envs=["X-Auth=Realm ${USER_NAME}, token ${API_TOKEN}"],
        env_file_path=str(env_file),
    )

    assert connect_sse_mock.await_args.kwargs["headers"] == {"X-Auth": "Realm alice, token file-token"}


async def test_sse_tool_env_file_is_variable_store_not_header_store(tmp_path, monkeypatch):
    """Underscore-style env-file entries must not be merged as literal headers."""
    monkeypatch.delenv("FILE_ONLY_VAR", raising=False)
    env_file = tmp_path / "vars.env"
    env_file.write_text("FILE_ONLY_VAR=file-value\nEMPTY_VAR=\n")

    connect_sse_mock = await _run_sse_tool(env_file_path=str(env_file))

    assert connect_sse_mock.await_args.kwargs["headers"] is None


async def test_sse_tool_env_file_header_like_entry_still_merged(tmp_path, monkeypatch):
    """Header-like env-file entries (hyphenated names) must stay merged for back-compat."""
    env_file = tmp_path / "headers.env"
    env_file.write_text("Authorization=Bearer from-file\nX-Api-Key=key-123\nVAR_STYLE=dropped\n")

    connect_sse_mock = await _run_sse_tool(env_file_path=str(env_file))

    headers = connect_sse_mock.await_args.kwargs["headers"]
    assert headers == {"Authorization": "Bearer from-file", "X-Api-Key": "key-123"}


async def test_sse_tool_env_file_lowercase_entry_not_merged(tmp_path):
    """Fully lowercase env-file entries are variable-store style and must not become headers."""
    env_file = tmp_path / "vars.env"
    env_file.write_text("token=secret-token\napikey=key-123\n")

    connect_sse_mock = await _run_sse_tool(env_file_path=str(env_file))

    assert connect_sse_mock.await_args.kwargs["headers"] is None


async def test_sse_tool_precedence_headers_env_file_secrets(tmp_path, monkeypatch):
    """Precedence (lowest to highest): plain headers -> header-like env file -> secret envs."""
    monkeypatch.setenv("SHARED_SECRET", "from-env")
    env_file = tmp_path / "headers.env"
    env_file.write_text("SHARED_SECRET=from-file\nX-File-Only=file-value\n")

    connect_sse_mock = await _run_sse_tool(
        headers={"SHARED_SECRET": "from-plain", "PLAIN_ONLY": "plain-value"},
        secret_envs=["SHARED_SECRET"],
        env_file_path=str(env_file),
    )

    headers = connect_sse_mock.await_args.kwargs["headers"]
    assert headers == {
        "SHARED_SECRET": "from-env",
        "X-File-Only": "file-value",
        "PLAIN_ONLY": "plain-value",
    }


async def test_sse_tool_missing_secret_env_raises_valueerror(monkeypatch):
    """A secret_env variable missing from both sources must raise a clear ValueError."""
    monkeypatch.delenv("MISSING_SECRET", raising=False)

    with pytest.raises(ValueError, match="MISSING_SECRET"):
        await _run_sse_tool(secret_envs=["MISSING_SECRET"])


async def test_sse_tool_missing_template_variable_does_not_leak_secrets(tmp_path, monkeypatch):
    """The error for a missing template variable must not contain resolved secret values."""
    monkeypatch.setenv("PRESENT_SECRET", "supersecret-value")
    monkeypatch.delenv("MISSING_SECRET", raising=False)

    with pytest.raises(ValueError, match="MISSING_SECRET") as exc_info:
        await _run_sse_tool(secret_envs=["Authorization=Bearer ${PRESENT_SECRET} ${MISSING_SECRET}"])

    assert "supersecret-value" not in str(exc_info.value)


async def test_sse_tool_malformed_template_entry_raises_valueerror():
    """A template entry without a header name must raise ValueError."""
    with pytest.raises(ValueError, match="must name a header"):
        await _run_sse_tool(secret_envs=["=${SOME_VAR}"])
