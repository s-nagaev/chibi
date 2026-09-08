from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Parasail(OpenAIFriendlyProvider):
    api_key = gpt_settings.parasail_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.parasail.io/v1"
    name = "Parasail"

    default_model = "parasail-deepseek-r1"
    default_moderation_model = "parasail-deepseek-r1"
