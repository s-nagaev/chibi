from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Cheaperinference(OpenAIFriendlyProvider):
    api_key = gpt_settings.cheaperinference_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True
    image_generation_ready = True

    base_url = "https://api.cheaperinference.com/v1"
    name = "Cheaper Inference"

    model_name_keywords_exclude = ["imagine", "nano", "image", "embedding", "image", "pixtral", "multilingual"]

    default_model = "glm-5.2"
    default_moderation_model = "gpt-5.6-luna"
    default_vision_model = "gpt-5.6-luna"
    default_image_model = "nano-banana-pro"
    temperature = 1
