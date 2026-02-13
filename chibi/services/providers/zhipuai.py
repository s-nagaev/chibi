from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class ZhipuAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.zai_key
    chat_ready = True
    moderation_ready = True

    name = "ZhipuAI"
    model_name_keywords = ["glm"]
    base_url = "https://api.z.ai/api/paas/v4/"
    default_model = "glm-5"
    default_moderation_model = "glm-4-32b-0414-128k"

    def get_model_display_name(self, model_name: str) -> str:
        return model_name.upper()
