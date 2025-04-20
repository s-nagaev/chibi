import datetime

from loguru import logger


async def get_current_datetime():
    logger.log("TOOL", "Getting current date & time")
    now = datetime.datetime.now()
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return datetime_str
