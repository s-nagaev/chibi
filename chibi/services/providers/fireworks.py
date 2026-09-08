from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Fireworks(OpenAIFriendlyProvider):
    api_key = gpt_settings.fireworks_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.fireworks.ai/inference/v1"
    name = "Fireworks"

    default_model = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    default_moderation_model = "accounts/fireworks/models/glm-5p3-flash"
    default_vision_model = "accounts/fireworks/models/glm-5p3-flash"
