from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class MoonshotAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.moonshotai_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.moonshot.ai/v1"
    name = "MoonshotAI"
    model_name_keywords = ["moonshot", "kimi"]
    model_name_keywords_exclude = ["vision"]
    default_model = "kimi-k2.6"
    default_moderation_model = "kimi-k2.6"
    default_vision_model = "kimi-k2.6"
    temperature = 1
