from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Baseten(OpenAIFriendlyProvider):
    api_key = gpt_settings.baseten_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://inference.baseten.co/v1"
    name = "Baseten"

    default_model = "openai/gpt-oss-120b"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
