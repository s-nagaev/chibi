import os
import re
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

_SECRET_TEMPLATE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$")

SECRET_ENVS_SCHEMA_DESCRIPTION = (
    "A list of secret entries. Each entry is either 'Header-Name=template ${VAR}' — "
    "a header 'Header-Name' whose value is the template with every ${VAR} substituted "
    "from os.environ with fallback to the env file — or a bare 'VAR' name, which "
    "produces the header 'VAR: <value>' (for stdio: the child process env var 'VAR'). "
    "A variable missing from both sources raises an error naming it. "
    "Resolved values are never exposed. Highest merge precedence."
)
ENV_FILE_PATH_SCHEMA_DESCRIPTION = (
    "Absolute path to the env file. The file is a VARIABLE store used as the fallback "
    "source for secret_envs template substitution; only entries whose names look like "
    "real HTTP headers (hyphenated like 'X-Api-Key', or canonical mixed case like "
    "'Authorization') are additionally merged as literal headers. All-uppercase or "
    "underscore-style names (e.g. MELIOUS_API_KEY) are never sent as headers."
)


def _looks_like_http_header_name(name: str) -> bool:
    """Check whether a name looks like a real HTTP header name.

    The rule: a hyphenated name (``X-Api-Key``) or a canonical mixed-case single
    word (``Authorization``) is treated as an HTTP header. Anything with
    underscores (``MELIOUS_API_KEY``) or a fully uppercase single word
    (``API_TOKEN``) is treated as an environment variable and never sent as a
    header.

    Args:
        name: The name to check.

    Returns:
        True when the name looks like an HTTP header name.
    """
    if "-" in name:
        return bool(_HTTP_HEADER_NAME_PATTERN.match(name))
    if "_" in name:
        return False
    return not name.isupper()


def _load_env_file_values(env_file_path: str | None) -> dict[str, str]:
    """Load non-empty values from an env file.

    Args:
        env_file_path: Optional path to the env file.

    Returns:
        Non-empty key-value pairs from the file, or an empty dict when no path is given.

    Raises:
        ValueError: If ``env_file_path`` is provided but the file does not exist.
    """
    if not env_file_path:
        return {}
    if not os.path.exists(env_file_path):
        raise ValueError(f"The file {env_file_path} does not exist.")
    envs_from_file = dotenv_values(env_file_path) or {}
    return {k: v for k, v in envs_from_file.items() if v}


def _split_secret_entry(entry: str) -> tuple[str, str]:
    """Split a secret_envs entry into a target name and a value template.

    Args:
        entry: Either ``"Header-Name=template ${VAR}"`` or a bare ``"VAR"`` name.

    Returns:
        A tuple of the target name and the template. For a bare name the template
        is ``${VAR}`` referencing the name itself.

    Raises:
        ValueError: If the entry is an empty template form (``"=..."`` or ``"="``).
    """
    if "=" in entry:
        name, _, template = entry.partition("=")
        name = name.strip()
        if not name:
            raise ValueError("A secret_envs template entry must name a header before '='.")
        return name, template.strip()
    return entry.strip(), "${" + entry.strip() + "}"


def _resolve_secret_variable(name: str, env_file_values: dict[str, str]) -> str:
    """Resolve a variable from os.environ with fallback to the env file values.

    Args:
        name: The variable name to resolve.
        env_file_values: Non-empty values loaded from the env file.

    Returns:
        The resolved variable value.

    Raises:
        ValueError: If the variable is missing from both os.environ and the env
            file. The message names the variable but never contains any resolved
            secret values.
    """
    value = os.environ.get(name)
    if value is None:
        value = env_file_values.get(name)
    if value is None:
        raise ValueError(f"Variable '{name}' required by secret_envs is not set in the environment or the env file.")
    return value


def _expand_secret_template(template: str, env_file_values: dict[str, str]) -> str:
    """Substitute every ``${VAR}`` occurrence in a template with its resolved value.

    Args:
        template: The template string, e.g. ``"Bearer ${MELIOUS_API_KEY}"``.
        env_file_values: Non-empty values loaded from the env file.

    Returns:
        The template with all variables substituted.

    Raises:
        ValueError: If any referenced variable is missing from both os.environ
            and the env file.
    """
    return _SECRET_TEMPLATE_PATTERN.sub(
        lambda match: _resolve_secret_variable(match.group(1), env_file_values), template
    )


def _compose_auth_headers(
    headers: dict[str, str] | None,
    secret_envs: list[str] | None,
    env_file_path: str | None,
) -> dict[str, str] | None:
    """Compose request headers for HTTP-based MCP transports.

    The env file acts as a VARIABLE store: its values are the fallback source for
    ``${VAR}`` substitution in ``secret_envs`` templates. Only env-file entries whose
    names look like real HTTP headers (hyphenated like ``X-Api-Key``, or canonical
    mixed case like ``Authorization``) are additionally merged as literal headers, so
    configs that intentionally placed ``Authorization=...``-style entries in the file
    keep working. All-uppercase or underscore-style names (e.g. ``MELIOUS_API_KEY``)
    are never sent as headers.

    Each ``secret_envs`` entry is either ``"Header-Name=template ${VAR}"`` (the header
    value is the template with every ``${VAR}`` substituted) or a bare ``"VAR"`` name
    (backward compatible: the header ``VAR: <value>``). Variables are resolved from
    ``os.environ`` with fallback to the env file. Resolved secret values are never
    logged.

    Merge precedence (lowest to highest): plain ``headers`` -> header-like env-file
    entries -> ``secret_envs`` entries.

    Args:
        headers: Optional plain HTTP headers sent as-is to the client.
        secret_envs: Optional list of secret entries (templates or bare names).
        env_file_path: Optional path to the env file used as the variable store.

    Returns:
        The composed header dict, or None when no inputs are provided.

    Raises:
        ValueError: If ``env_file_path`` does not exist, a secret entry is malformed,
            or a referenced variable is missing from both os.environ and the env file.
    """
    env_file_values = _load_env_file_values(env_file_path)
    composed = dict(headers) if headers is not None else None

    header_like_entries = {name: value for name, value in env_file_values.items() if _looks_like_http_header_name(name)}
    if header_like_entries:
        composed = {**(composed or {}), **header_like_entries}

    if secret_envs:
        composed = composed or {}
        for entry in secret_envs:
            header_name, template = _split_secret_entry(entry)
            composed[header_name] = _expand_secret_template(template, env_file_values)

    return composed


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
                            "A list of secret entries passed to the child process env without direct "
                            "access to their contents. Each entry is either 'VAR=template ${VAR}' — "
                            "the child env var 'VAR' set to the template with every ${VAR} substituted "
                            "from os.environ with fallback to the env file — or a bare 'VAR' name copied "
                            "from os.environ with the same fallback. A variable missing from both sources "
                            "raises an error naming it. Resolved values are never exposed."
                        ),
                    },
                    "env_file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the env-file. Merged into the child process env as a "
                            "variable store and used as the fallback source for secret_envs templates."
                        ),
                    },
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

        ``secret_envs`` entries are resolved with the shared helper: a bare ``"VAR"``
        copies the variable into the child process env, a ``"Name=template ${VAR}"``
        entry sets the child env var ``Name`` to the substituted template. Variables
        are resolved from ``os.environ`` with fallback to ``env_file_path`` values.
        The env file itself is merged into the child env as a variable store.

        Args:
            server_name: Unique name for the session.
            command: Command to start the server.
            args: Command arguments.
            env: Optional environment variables.
            secret_envs: Optional list of secret entries (templates or bare names).
            env_file_path: Optional path to the env file used as the variable store.

        Returns:
            dict[str, Any]: Success or error message.

        Raises:
            ValueError: If ``env_file_path`` does not exist or a secret entry is
                malformed or references a missing variable.
        """
        env_file_values = _load_env_file_values(env_file_path)

        if secret_envs:
            env = env or {}
            for entry in secret_envs:
                target_name, template = _split_secret_entry(entry)
                env[target_name] = _expand_secret_template(template, env_file_values)

        if env_file_values:
            env = env or {}
            env.update(env_file_values)

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
                            "Merged with the lowest precedence: header-like env_file_path entries "
                            "override them, secret_envs values override both."
                        ),
                    },
                    "secret_envs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": SECRET_ENVS_SCHEMA_DESCRIPTION,
                    },
                    "env_file_path": {
                        "type": "string",
                        "description": ENV_FILE_PATH_SCHEMA_DESCRIPTION,
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

        Headers are composed by ``_compose_auth_headers``: the env file is a variable
        store (only header-like entries are merged as literal headers) and each
        ``secret_envs`` entry is a template or a bare variable name resolved from
        ``os.environ`` with fallback to the env file. Secret values are never logged.

        Args:
            server_name: Unique name for the session.
            url: SSE endpoint URL (sse).
            headers: Optional plain HTTP headers sent as-is to the SSE client.
            secret_envs: Optional list of secret entries (templates or bare names).
            env_file_path: Optional path to the env file used as the variable store.

        Returns:
            dict[str, Any]: Success or error message.

        Raises:
            ValueError: If ``env_file_path`` does not exist or a secret entry is
                malformed or references a missing variable.
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
                            "Merged with the lowest precedence: header-like env_file_path entries "
                            "override them, secret_envs values override both."
                        ),
                    },
                    "secret_envs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": SECRET_ENVS_SCHEMA_DESCRIPTION,
                    },
                    "env_file_path": {
                        "type": "string",
                        "description": ENV_FILE_PATH_SCHEMA_DESCRIPTION,
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
            secret_envs: Optional list of secret entries (templates or bare names).
            env_file_path: Optional path to the env file used as the variable store.

        Returns:
            dict[str, Any]: Success or error message.

        Raises:
            ValueError: If ``env_file_path`` does not exist or a secret entry is
                malformed or references a missing variable.
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
