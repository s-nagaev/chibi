from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Lyceum(OpenAIFriendlyProvider):
    api_key = gpt_settings.lyceum_key
    chat_ready = True
    vision_ready = True

    base_url = "https://api.lyceum.technology/openai/v1"
    name = "Lyceum"
    model_name_keywords_exclude = ["whisper", "embedding", "image", "flux", "speech", "tts", "moderation"]

    default_model = "kimi-k3"
    default_vision_model = "kimi-k3"
    temperature = 1
