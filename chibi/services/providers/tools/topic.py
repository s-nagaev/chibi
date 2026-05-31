from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.services.providers.tools.exceptions import ToolException
from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import save_thread_name


class RenameThreadTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="rename_thread",
            description="Rename the current topic/thread in telegram app or other UI.",
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
            raise ToolException("No interface found. This command can only be used inside a Telegram forum topic.")

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

        await save_thread_name(user_id=user_id, thread_id=interface.thread_id, name=name)

        return {"status": "ok", "message": f"Thread successfully renamed to '{name}'."}
