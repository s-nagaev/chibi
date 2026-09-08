from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Parasail(OpenAIFriendlyProvider):
    api_key = gpt_settings.parasail_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.parasail.io/v1"
    name = "Parasail"

    default_model = "zai-org/GLM-5.3-Flash"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
