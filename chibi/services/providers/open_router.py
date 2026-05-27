from openai import NOT_GIVEN, omit

from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenRouter(OpenAIFriendlyProvider):
    api_key = gpt_settings.open_router_key
    chat_ready = True
    image_generation_ready = False
    moderation_ready = False
    vision_ready = False

    base_url = "https://openrouter.ai/api/v1"
    name = "OpenRouter"
    # model_name_keywords = ["grok"]
    # model_name_keywords_exclude = ["vision", "imag"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    # default_image_model = "ernie-5.0"
    # default_model = "grok-4-1-fast-reasoning"
    # default_moderation_model = "grok-4-1-fast-non-reasoning"
    # default_vision_model = "grok-4-1-fast-reasoning"
    presence_penalty = NOT_GIVEN
    frequency_penalty = omit
    image_n_choices = 1

    def get_model_display_name(self, model_name: str) -> str:
        original_provider, model_name = model_name.split("/")
        return f"{model_name} / {original_provider}"
