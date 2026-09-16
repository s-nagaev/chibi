from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Featherless(OpenAIFriendlyProvider):
    api_key = gpt_settings.featherless_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.featherless.ai/v1"
    name = "Featherless"

    default_model = "zai-org/GLM-5.3-Flash"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
