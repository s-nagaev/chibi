import os
from typing import Any, Callable, Coroutine, cast

from dotenv import dotenv_values
from loguru import logger
from mcp import ClientSession
from mcp.types import CallToolResult
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import application_settings
from chibi.services.mcp.manager import MCPManager
from chibi.services.providers.tools.tool import ChibiTool, RegisteredChibiTools
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.providers.utils import escape_and_truncate

JsonNode = dict[str, Any] | list[Any] | str | int | float | bool | None


def _compose_auth_headers(
    headers: dict[str, str] | None,
    secret_envs: list[str] | None,
    env_file_path: str | None,
) -> dict[str, str] | None:
    """Compose request headers for HTTP-based MCP transports.

    The final header dict is composed with the following precedence (lowest to
    highest): plain ``headers`` -> non-empty values from ``env_file_path`` ->
    values of the environment variables named in ``secret_envs``. Secret values
    are pulled from ``os.environ`` and are never logged.

    Args:
        headers: Optional plain HTTP headers sent as-is to the client.
        secret_envs: Optional list of environment variable names whose values are
            injected as headers (highest precedence).
        env_file_path: Optional path to the env file with header values.

    Returns:
        The composed header dict, or None when no inputs are provided.

    Raises:
        ValueError: If ``env_file_path`` does not exist.
        KeyError: If a variable named in ``secret_envs`` is missing from the environment.
    """
    if env_file_path:
        if not os.path.exists(env_file_path):
            raise ValueError(f"The file {env_file_path} does not exist.")
        headers = headers or {}
        envs_from_file = dotenv_values(env_file_path)
        envs_with_values_from_file = {k: v for k, v in envs_from_file.items() if v}
        if envs_with_values_from_file:
            headers.update(envs_with_values_from_file)

    if secret_envs:
        headers = headers or {}
        for secret_env in secret_envs:
            headers[secret_env] = os.environ[secret_env]

    return headers


def _clean_schema(schema: JsonNode) -> JsonNode:
    """Recursively cleans the schema for LLM provider compatibility.

    Removes 'additionalProperties', '$schema', and other meta-fields that cause validation errors.
    """
    if isinstance(schema, dict):
        # Fields to exclude from the schema
        exclude = {"additionalProperties", "$schema"}
        # Also clean nested objects in 'properties' and 'items'
        cleaned = {k: _clean_schema(v) for k, v in schema.items() if k not in exclude}
        return cleaned
    if isinstance(schema, list):
        return [_clean_schema(i) for i in schema]
    return schema


def create_wrapper(
    server_name: str, original_tool_name: str, chibi_tool_name: str
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    async def wrapper(**kwargs_inner: Any) -> dict[str, Any]:
        tool_args = {k: v for k, v in kwargs_inner.items() if k not in list(AdditionalOptions.__annotations__.keys())}

        logger.log(
            "TOOL",
            (
                f"[{kwargs_inner.get('model', 'Unknown model')}] "
                f"Calling MCP tool '{chibi_tool_name}' with args: {escape_and_truncate(tool_args)}"
            ),
        )
        res: CallToolResult = await MCPManager.call_tool(
            server_name=server_name, tool_name=original_tool_name, arguments=tool_args
        )

        if len(res.content) == 1:
            output = res.content[0].model_dump()
        else:
            output = {"result": [content.model_dump() for content in res.content]}

        logger.log(
            "TOOL",
            (
                f"[{kwargs_inner.get('model', 'Unknown model')}] "
                f"MCP tool '{chibi_tool_name}' returned: {escape_and_truncate(output)}",
            ),
        )
        return output

    return wrapper


async def register_tools_from_mcp_session(
    mcp_session: ClientSession, server_name: str, transport: str
) -> dict[str, str]:
    try:
        tools_result = await mcp_session.list_tools()

        mcp_tools = []
        for tool in tools_result.tools:
            chibi_tool_name = f"mcp_{server_name}_{transport}_{tool.name}"
            class_attributes = {
                "register": True,
                "definition": ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=chibi_tool_name,
                        description=f"MCP Server: {server_name}. {tool.description or ''}",
                        parameters=cast(dict[str, object], _clean_schema(tool.input_schema)),
                    ),
                ),
                "name": chibi_tool_name,
                "function": staticmethod(create_wrapper(server_name, tool.name, chibi_tool_name)),
            }
            type(f"MCP{server_name.capitalize()}{transport.capitalize()}Tool", (ChibiTool,), class_attributes)
            mcp_tools.append(chibi_tool_name)
    except Exception as e:
        logger.error(f"Failed to register tools for MCP server '{server_name}': {e}")
        await MCPManager.disconnect(server_name)
        return {"status": "error", "message": f"Failed to register tools: {e}"}
    MCPManager.associate_tools_with_server(server_name=server_name, tool_names=mcp_tools)

    return {
        "status": "success",
        "message": (
            f"Successfully connected to MCP server '{server_name}'. "
            f"Registered {len(mcp_tools)} tools: {', '.join(mcp_tools)}"
        ),
    }


class InitializeStdioMCPServer(ChibiTool):
    register = application_settings.enable_mcp_stdio
    name = "initialize_stdio_mcp_server"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="initialize_stdio_mcp_server",
            description="Connect to an MCP server via stdio and register its tools dynamically.",
            parameters={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Unique name for this server session (e.g., 'sqlite', 'github').",
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to start the MCP server",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments for the command",
                    },
                    "env": {
                        "type": "object",
                        "description": "Optional environment variables",
                    },
                    "secret_envs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A list of environment variables that need to be passed to the mcp server "
                            "without direct access to their contents."
                        ),
                    },
                    "env_file_path": {"type": "string", "description": "Absolute path to the env-file."},
                },
                "required": ["server_name", "command", "args"],
            },
        ),
    )

    @classmethod
    async def function(
        cls,
        server_name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        secret_envs: list[str] | None = None,
        env_file_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Connect to an MCP server via stdio and register its tools dynamically.

        Args:
            server_name: Unique name for the session.
            command: Command to start the server.
            args: Command arguments.
            env: Optional environment variables.
            secret_envs: Optional list of environment variable names.
            env_file_path: Optional path to the env file.

        Returns:
            dict[str, Any]: Success or error message.
        """
        if secret_envs:
            env = env or {}
            for secret_env in secret_envs:
                env[secret_env] = os.environ[secret_env]

        if env_file_path:
            if not os.path.exists(env_file_path):
                raise ValueError(f"The file {env_file_path} does not exist.")
            env = env or {}
            envs_from_file = dotenv_values(env_file_path)
            envs_with_values_from_file = {k: v for k, v in envs_from_file.items() if v}
            if envs_with_values_from_file:
                env.update(envs_with_values_from_file)

        session = await MCPManager.connect_stdio(name=server_name, command=command, args=args, env=env)
        return await register_tools_from_mcp_session(mcp_session=session, server_name=server_name, transport="stdio")


class InitializeSseMCPServer(ChibiTool):
    register = application_settings.enable_mcp_sse
    name = "initialize_sse_mcp_server"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="initialize_sse_mcp_server",
            description="Connect to an MCP server via SSE and register its tools dynamically.",
            parameters={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Unique name for this server session (e.g., 'sqlite', 'github').",
                    },
                    "url": {"type": "string", "description": "SSE endpoint URL (required for sse)."},
                    "headers": {
                        "type": "object",
                        "description": (
                            "Optional plain HTTP headers sent with the SSE request (e.g. Authorization). "
                            "Merged with the lowest precedence: env_file_path values override them, "
                            "secret_envs values override both."
                        ),
                    },
                    "secret_envs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A list of environment variables whose values are injected as headers "
                            "without direct access to their contents. Highest merge precedence."
                        ),
                    },
                    "env_file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the env-file with header values. "
                            "Overrides plain headers, but is overridden by secret_envs."
                        ),
                    },
                },
                "required": ["server_name", "url"],
            },
        ),
    )

    @classmethod
    async def function(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, str] | None = None,
        secret_envs: list[str] | None = None,
        env_file_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Connect to an MCP server via SSE and register its tools dynamically.

        The final header dict is composed with the following precedence (lowest to
        highest): plain ``headers`` -> non-empty values from ``env_file_path`` ->
        values of the environment variables named in ``secret_envs``. Secret values
        are pulled from ``os.environ`` and are never logged.

        Args:
            server_name: Unique name for the session.
            url: SSE endpoint URL (sse).
            headers: Optional plain HTTP headers sent as-is to the SSE client.
            secret_envs: Optional list of environment variable names whose values are
                injected as headers (highest precedence).
            env_file_path: Optional path to the env file with header values.

        Returns:
            dict[str, Any]: Success or error message.
        """
        headers = _compose_auth_headers(headers, secret_envs, env_file_path)

        try:
            session = await MCPManager.connect_sse(server_name, url, headers=headers)
        except Exception:
            raise
        return await register_tools_from_mcp_session(mcp_session=session, server_name=server_name, transport="sse")


class InitializeRestMCPServer(ChibiTool):
    register = application_settings.enable_mcp_rest
    name = "initialize_rest_mcp_server"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="initialize_rest_mcp_server",
            description=(
                "Connect to an MCP server via REST (plain JSON-RPC over HTTP POST) and register its tools "
                "dynamically. Suitable for stateless JSON-RPC endpoints that do not support SSE."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server_name": {
                        "type": "string",
                        "description": "Unique name for this server session (e.g., 'sqlite', 'github').",
                    },
                    "url": {"type": "string", "description": "JSON-RPC endpoint URL (required for rest)."},
                    "headers": {
                        "type": "object",
                        "description": (
                            "Optional plain HTTP headers sent with every request (e.g. Authorization). "
                            "Merged with the lowest precedence: env_file_path values override them, "
                            "secret_envs values override both."
                        ),
                    },
                    "secret_envs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "A list of environment variables whose values are injected as headers "
                            "without direct access to their contents. Highest merge precedence."
                        ),
                    },
                    "env_file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the env-file with header values. "
                            "Overrides plain headers, but is overridden by secret_envs."
                        ),
                    },
                },
                "required": ["server_name", "url"],
            },
        ),
    )

    @classmethod
    async def function(
        cls,
        server_name: str,
        url: str,
        headers: dict[str, str] | None = None,
        secret_envs: list[str] | None = None,
        env_file_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Connect to an MCP server via REST and register its tools dynamically.

        The transport performs one plain JSON POST per JSON-RPC message over the
        SDK streamable-HTTP client, so it works against stateless JSON-RPC
        endpoints without SSE or session negotiation. Headers are composed by
        ``_compose_auth_headers`` with the same precedence as the SSE tool.

        Args:
            server_name: Unique name for the session.
            url: JSON-RPC endpoint URL (rest).
            headers: Optional plain HTTP headers sent as-is to the client.
            secret_envs: Optional list of environment variable names whose values are
                injected as headers (highest precedence).
            env_file_path: Optional path to the env file with header values.

        Returns:
            dict[str, Any]: Success or error message.
        """
        headers = _compose_auth_headers(headers, secret_envs, env_file_path)

        session = await MCPManager.connect_rest(server_name, url, headers=headers)
        return await register_tools_from_mcp_session(mcp_session=session, server_name=server_name, transport="rest")


class DeinitializeMCPServer(ChibiTool):
    """Disconnects from an MCP server and removes its tools."""

    register = (
        application_settings.enable_mcp_stdio
        or application_settings.enable_mcp_sse
        or application_settings.enable_mcp_rest
    )
    name = "deinitialize_mcp_server"
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="deinitialize_mcp_server",
            description="Disconnect from an MCP server and unregister its tools.",
            parameters={
                "type": "object",
                "properties": {"server_name": {"type": "string", "description": "Name of the server to disconnect."}},
                "required": ["server_name"],
            },
        ),
    )

    @classmethod
    async def function(cls, server_name: str, **kwargs: Any) -> dict[str, Any]:
        """Executes the tool.

        Args:
            server_name: Name of the server to disconnect.

        Returns:
            dict[str, Any]: Success or error message.
        """
        tools_to_deregister = await MCPManager.disconnect(server_name)
        if tools_to_deregister:
            RegisteredChibiTools.deregister_tools(tools_to_deregister)

        return {
            "status": "success",
            "message": f"Disconnected from MCP server '{server_name}' and removed its tools.",
        }
