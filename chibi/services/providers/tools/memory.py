from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions
from chibi.services.user import set_info


class SetUserInfoTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="set_user_info",
            description=(
                "Set user info that is important for YOU and YOUR job."
                "Important: this function will override the current user info!"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "new_user_info": {"type": "string", "description": "New user info."},
                },
                "required": ["new_user_info"],
            },
        ),
    )
    name = "set_user_info"

    @classmethod
    async def function(cls, new_user_info: str, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        user_id = kwargs.get("user_id")
        if not user_id:
            raise ValueError("This function requires user_id to be automatically provided.")
        logger.log("TOOL", f"Setting new user info about user #{user_id}: {new_user_info}")
        await set_info(user_id=user_id, new_info=new_user_info)
        return {"status": "ok"}
