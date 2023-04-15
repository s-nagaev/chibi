import sys
from typing import Any

from loguru import logger

config: dict[Any, Any] = {
    "handlers": [
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": "<lvl>{level}</lvl>\t| <green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> |  <lvl>{message}</lvl>",
        },
    ],
}
logger.configure(**config)
