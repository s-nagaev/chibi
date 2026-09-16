from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class DeepInfra(OpenAIFriendlyProvider):
    api_key = gpt_settings.deepinfra_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.deepinfra.com/v1/openai"
    name = "Deepinfra"

    default_model = "zai-org/GLM-5.3-Flash"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
