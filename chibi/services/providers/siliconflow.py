from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class SiliconFlow(OpenAIFriendlyProvider):
    api_key = gpt_settings.siliconflow_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    base_url = "https://api.siliconflow.com/v1"
    name = "Siliconflow"

    default_model = "deepseek-ai/DeepSeek-V4-Flash"
    default_moderation_model = "deepseek-ai/DeepSeek-V4-Flash"
    default_vision_model = "zai-org/GLM-5V-Turbo"
