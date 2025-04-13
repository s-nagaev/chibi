import re
from enum import Enum
from typing import Literal

from telegram import constants

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]
IMAGE_SIZE_LITERAL = Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
IMAGE_ASPECT_RATIO_LITERAL = Literal["1:1", "3:4", "4:3", "9:16", "16:9"]
MARKDOWN_V2_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"
ESCAPE_PATTERN = re.compile(r"(```(?:.|\n)*?```|`.*?`)|([" + re.escape(MARKDOWN_V2_ESCAPE_CHARS) + r"])", re.DOTALL)
SETTING_SET = "<green>SET</green>"
SETTING_UNSET = "<red>UNSET</red>"


class UserContext(Enum):
    ACTION = "ACTION"
    SELECTED_PROVIDER = "SELECTED_PROVIDER"
    ACTIVE_MODEL = "ACTIVE_MODEL"
    ACTIVE_IMAGE_MODEL = "ACTIVE_IMAGE_MODEL"
    MAPPED_MODELS = "MAPPED_MODELS"


class UserAction(Enum):
    SELECT_MODEL = "SELECT_MODEL"
    SELECT_PROVIDER = "SELECT_PROVIDER"
    SET_API_KEY = "SET_API_KEY"
    IMAGINE = "IMAGINE"
    NONE = None
