from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class GreenPT(OpenAIFriendlyProvider):
    api_key = gpt_settings.greenpt_key
    chat_ready = True
    vision_ready = True

    base_url = "https://api.greenpt.ai/v1"
    name = "GreenPT"
    model_name_keywords_exclude = ["embedding", "rerank"]

    default_model = "glm-5.2"
    default_vision_model = "kimi-k3"
