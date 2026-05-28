from openai import NOT_GIVEN, omit

from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenRouter(OpenAIFriendlyProvider):
    api_key = gpt_settings.open_router_key
    chat_ready = True
    moderation_ready = True

    name = "OpenRouter"
    base_url = "https://openrouter.ai/api/v1"
    default_model = "google/gemini-3.5-flash"
    default_moderation_model = "google/gemini-3.1-flash-lite"
    presence_penalty = NOT_GIVEN
    frequency_penalty = omit
    image_n_choices = 1

    def get_model_display_name(self, model_name: str) -> str:
        original_provider, model_name = model_name.split("/")
        return f"{model_name} / {original_provider}"
