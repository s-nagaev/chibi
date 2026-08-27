from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Melious(OpenAIFriendlyProvider):
    api_key = gpt_settings.melious_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.melious.ai/v1"
    name = "Melious"
    model_name_keywords_exclude = ["whisper", "flux", "voxtral", "embedding", "image", "pixtral", "multilingual"]

    default_model = "kimi-k2.6"
    default_moderation_model = "deepseek-v4-flash"
    default_vision_model = "kimi-k2.6"
    temperature = 1
