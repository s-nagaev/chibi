from openai import NOT_GIVEN

from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class MoonshotAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.moonshotai_key
    chat_ready = True
    moderation_ready = True

    base_url = "https://api.moonshot.cn/v1"
    name = "MoonshotAI"
    model_name_keywords = ["moonshot", "kimi"]
    model_name_keywords_exclude = ["vision"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_model = "kimi-latest"
    default_moderation_model = "kimi-k2-turbo-preview"
    temperature = 1
