from typing import Any, Unpack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions


class McpEchoTool(ChibiTool):
    """Test/demo only"""
    register = False
    name = "mcp_echo"
    definition = {
        "type": "function",
        "function": {
            "name": "mcp_echo",
            "description": "Send a message to a local MCP echo server and get the response.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "The message to echo back"}},
                "required": ["message"],
            },
        },
    }

    @classmethod
    async def function(cls, message: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, Any]:
        server_script = """
import asyncio
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo")

@mcp.tool()
def echo(message: str) -> str:
    return f"Echo from MCP: {message}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(server_script)
            server_path = f.name

        try:
            server_params = StdioServerParameters(
                command="python",
                args=[server_path],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("echo", arguments={"message": message})

                    if result.content and len(result.content) > 0:
                        content_item = result.content[0]
                        if content_item.type == "text":
                            return {"response": content_item.text}
                    return {"response": "No content returned"}

        except Exception as e:
            return {"error": f"MCP Error: {str(e)}"}
        finally:
            if os.path.exists(server_path):
                os.unlink(server_path)
