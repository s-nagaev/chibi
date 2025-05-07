import datetime

from loguru import logger

from chibi.services.providers.tools.schemas import ToolResponse


async def get_current_datetime() -> str:
    logger.log("TOOL", "Getting current date & time")
    now = datetime.datetime.now()
    result = {
        "status": "ok",
        "datetime_now": now.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return ToolResponse(
        tool_name="get_current_datetime",
        status="ok",
        result=result,
    ).model_dump_json()
