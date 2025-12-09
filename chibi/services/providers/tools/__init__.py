# flake8: noqa: F401

from .cmd import CreateFileTool, RunCommandInTerminalTool
from .common import TextToSpeechTool
from .current_date import GetCurrentDatetimeTool
from .file_editor import (
    AppendToFileTool,
    FindAndReplaceSectionTool,
    InsertAfterPatternTool,
    InsertAtLineTool,
    InsertBeforePatternTool,
    ReplaceInFileRegexTool,
    ReplaceInFileTool,
    ReplaceLinesTool,
)
from .mcp_simple import McpEchoTool
from .memory import SetUserInfoTool
from .schemas import ToolResponse
from .tool import RegisteredChibiTools, RegisteredFunctionsMap
from .web_search import DDGSWebSearchTool, GoogleSearchTool, ReadWebPageTool, SearchNewsTool

registered_functions: RegisteredFunctionsMap = RegisteredChibiTools.get_registered_functions()
tools = RegisteredChibiTools.get_tool_definitions()
