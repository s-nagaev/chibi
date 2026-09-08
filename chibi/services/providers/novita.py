from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Novita(OpenAIFriendlyProvider):
    api_key = gpt_settings.novita_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.novita.ai/openai"
    name = "Novita"

    default_model = "zai-org/glm-5.3-flash"
    default_moderation_model = "zai-org/glm-5.3-flash"
    default_vision_model = "zai-org/glm-5.3-flash"
    # temperature = 1
