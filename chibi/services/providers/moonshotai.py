from openai import NOT_GIVEN

from chibi.services.providers.provider import OpenAIFriendlyProvider


class MoonshotAI(OpenAIFriendlyProvider):
    base_url = "https://api.moonshot.cn/v1"
    name = "MoonshotAI"
    model_name_keywords = ["moonshot", "kimi"]
    model_name_keywords_exclude = ["vision"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_model = "kimi-latest"
    temperature = 0.3
