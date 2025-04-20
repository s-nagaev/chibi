from openai import NOT_GIVEN

from chibi.services.providers.provider import OpenAIFriendlyProvider


class Grok(OpenAIFriendlyProvider):
    base_url = "https://api.x.ai/v1"
    name = "Grok"
    model_name_keywords = ["grok"]
    model_name_keywords_exclude = ["vision", "image"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_image_model = "grok-2-image-1212"
    default_model = "grok-3-beta"
