from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.config import application_settings
from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import save_thread_name

# Client frontends where the rename tool is exposed: the Telegram bot and every JSONL stdio
# client (served by the same runner). The terminal REPL is intentionally excluded via the
# explicit deregistration in TerminalRunner.run_chibi.
_RENAME_CLIENTS = frozenset({"telegram", "tui", "vscode", "pycharm", "neovim"})

# Names of the rename tools. Shared with the terminal runner so it can deregister them at
# startup: the terminal entrypoint imports this module while the client setting still holds
# its "telegram" default, so the tool registers at import time and the runner must drop it
# explicitly (see TerminalRunner.run_chibi).
RENAME_THREAD_TOOL_NAMES: tuple[str, ...] = ("rename_thread",)


class RenameThreadTool(ChibiTool):
    register = application_settings.client in _RENAME_CLIENTS
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="rename_thread",
            description=(
                "Rename the current conversation thread: the new display name is persisted and "
                "shown in the client's thread list (in Telegram forum chats, the forum topic is renamed)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "New name for the thread (1-128 chars)",
                    },
                },
                "required": ["name"],
            },
        ),
    )
    name = "rename_thread"
    allow_model_to_change_background_mode = False

    @classmethod
    async def function(cls, name: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        """Execute the thread renaming tool.

        Args:
            name: New name for the thread.
            kwargs: Additional options provided to the tool, containing user_id and other context.

        Returns:
            A dictionary containing the status and message of the operation.

        Raises:
            ToolException: If user_id is missing, interface is not found, thread name is invalid,
                or if renaming fails.
        """
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ToolException("This function requires user_id to be automatically provided.")

        interface = cls.get_interface(kwargs=kwargs)
        if not interface:
            raise ToolException("No interface found. This command requires an active conversation thread context.")

        caller_model = kwargs.get("caller_model", "unknown model")

        name = name.strip()
        if not name:
            raise ToolException("Thread name cannot be empty.")

        if len(name) > 128:
            name = name[:128]
            logger.warning(f"[{caller_model}] Thread name truncated to 128 chars for user #{user_id}")

        logger.log(
            "TOOL",
            (
                f"[{caller_model}] Renaming thread #{interface.thread_id} in chat {interface.chat_id} to "
                f"'{name}' for user #{user_id}",
            ),
        )

        success = await interface.rename_thread(new_name=name)

        if not success:
            raise ToolException(
                "Failed to rename the topic. Make sure the bot has the required permissions (can manage topics)."
            )

        await save_thread_name(storage_id=user_id, thread_id=interface.thread_id, name=name)

        return {"status": "ok", "message": f"Thread successfully renamed to '{name}'."}
