from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class DeepSeek(OpenAIFriendlyProvider):
    api_key = gpt_settings.deepseek_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    name = "DeepSeek"
    base_url = "https://api.deepseek.com"

    default_model = "deepseek-v4-pro"
    default_moderation_model = "deepseek-v4-flash"
    default_vision_model = "deepseek-v4-flash-vision-exp"
