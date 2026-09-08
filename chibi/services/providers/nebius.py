from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Nebius(OpenAIFriendlyProvider):
    api_key = gpt_settings.nebius_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.tokenfactory.nebius.com/v1/"
    name = "Nebius"

    default_model = "nvidia/Nemotron-3_5-Lightning"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
    # temperature = 1
