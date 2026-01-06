import datetime
from typing import Unpack

from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition

from chibi.services.providers.tools.tool import ChibiTool
from chibi.services.providers.tools.utils import AdditionalOptions


class GetCurrentDatetimeTool(ChibiTool):
    register = True
    definition = ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_current_datetime",
            description="Get the current date and time.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    )
    name = "get_current_datetime"

    @classmethod
    async def function(cls, **kwargs: Unpack[AdditionalOptions]) -> dict[str, str]:
        logger.log("TOOL", f"[{kwargs.get('model', 'Unknown model')}] Getting current date & time")
        now = datetime.datetime.now()
        return {
            "datetime_now": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
