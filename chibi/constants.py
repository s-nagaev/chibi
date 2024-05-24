from enum import Enum

from telegram import constants

SET_API_KEY = "set_api_key"
IMAGINE_ACTION = "imagine_action"

GROUP_CHAT_TYPES = [constants.ChatType.GROUP, constants.ChatType.SUPERGROUP]
PERSONAL_CHAT_TYPES = [constants.ChatType.SENDER, constants.ChatType.PRIVATE]


class SupportedProviders(Enum):
    ANTHROPIC = "ANTHROPIC"
    MISTRALAI = "MISTRALAI"
    OPENAI = "OPENAI"
