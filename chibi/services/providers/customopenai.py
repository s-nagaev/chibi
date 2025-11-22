from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class CustomOpenAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.customopenai_key
    chat_ready = True

    name = "CustomOpenAI"
    base_url = gpt_settings.customopenai_url
    default_model = ""
