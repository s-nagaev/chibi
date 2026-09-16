from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Lyceum(OpenAIFriendlyProvider):
    api_key = gpt_settings.lyceum_key
    chat_ready = True
    vision_ready = True
    moderation_ready = True

    base_url = "https://api.lyceum.technology/openai/v1"
    name = "Lyceum"
    model_name_keywords_exclude = ["whisper", "embedding", "image", "flux", "speech", "tts", "moderation"]

    default_model = "z-ai/glm-5.3-flash"
    default_vision_model = "z-ai/glm-5.3-flash"
    default_moderation_model = "z-ai/glm-5.3-flash"
    temperature = 0.6
