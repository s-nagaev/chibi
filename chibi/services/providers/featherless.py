from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Featherless(OpenAIFriendlyProvider):
    api_key = gpt_settings.featherless_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.featherless.ai/v1"
    name = "Featherless"

    default_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    default_moderation_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    default_vision_model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # temperature = 1
