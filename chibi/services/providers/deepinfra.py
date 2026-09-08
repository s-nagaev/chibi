from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class DeepInfra(OpenAIFriendlyProvider):
    api_key = gpt_settings.deepinfra_key
    chat_ready = True
    moderation_ready = False
    vision_ready = True

    base_url = "https://api.deepinfra.com/v1/openai"
    name = "Deepinfra"

    default_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    default_vision_model = "google/gemma-3-4b-it"
    # temperature = 1
