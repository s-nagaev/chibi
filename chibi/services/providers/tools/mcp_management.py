from typing import Any, Callable, Coroutine

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

        logger.log("TOOL", f"Calling MCP tool '{chibi_tool_name}' with args: {escape_and_truncate(tool_args)}")
        res: CallToolResult = await MCPManager.call_tool(
            server_name=server_name, tool_name=original_tool_name, arguments=tool_args
        )

        if len(res.content) == 1:
            output = res.content[0].model_dump()
        else:
            output = {"result": [content.model_dump() for content in res.content]}

        logger.log("TOOL", f"MCP tool '{chibi_tool_name}' returned: {escape_and_truncate(output)}")
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
                        parameters=_clean_schema(tool.inputSchema),
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Connect to an MCP server via stdio and register its tools dynamically.

        Args:
            server_name: Unique name for the session.
            command: Command to start the server (stdio).
            args: Command arguments (stdio).
            env: Optional environment variables (stdio).

        Returns:
            dict[str, Any]: Success or error message.
        """
        session = await MCPManager.connect_stdio(server_name, command, args, env)
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Executes the tool.

        Args:
            server_name: Unique name for the session.
            url: SSE endpoint URL (sse).

        Returns:
            dict[str, Any]: Success or error message.
        """
        session = await MCPManager.connect_sse(server_name, url)
        return await register_tools_from_mcp_session(mcp_session=session, server_name=server_name, transport="sse")


class DeinitializeMCPServer(ChibiTool):
    """Disconnects from an MCP server and removes its tools."""

    register = application_settings.enable_mcp_stdio or application_settings.enable_mcp_sse
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
