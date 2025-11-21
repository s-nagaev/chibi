from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class DeepSeek(OpenAIFriendlyProvider):
    api_key = gpt_settings.deepseek_key
    chat_ready = True

    name = "DeepSeek"
    model_name_keywords = ["deepseek"]
    base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
