from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Entrim(OpenAIFriendlyProvider):
    """OpenAI-compatible adapter for the Entrim inference service (https://entrim.ai)."""

    api_key = gpt_settings.entrim_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.entrim.ai/v1"
    name = "Entrim"

    default_model = "deepseek-ai/DeepSeek-V4.1-Flash"
    default_moderation_model = "zai-org/GLM-5.3-Flash"
    default_vision_model = "zai-org/GLM-5.3-Flash"
