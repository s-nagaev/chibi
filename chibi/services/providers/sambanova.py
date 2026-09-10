from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class SambaNova(OpenAIFriendlyProvider):
    api_key = gpt_settings.sambanova_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.sambanova.ai/v1"
    name = "SambaNova"

    default_model = "MiniMax-M3"
    default_moderation_model = "gpt-oss-120b"
    default_vision_model = "gemma-4-31B-it"
