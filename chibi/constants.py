from enum import Enum
from typing import Literal

from telegram import constants


IMAGINE_ACTION = "imagine_action"
API_KEY_SET_SELECTED_PROVIDER = "api_key_set_selected_provider"

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]
IMAGE_SIZE_LITERAL = Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
IMAGE_ASPECT_RATIO_LITERAL = Literal["1:1", "3:4", "4:3", "9:16", "16:9"]


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
    NONE = "NONE"
