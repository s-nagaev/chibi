from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Cloudflare(OpenAIFriendlyProvider):
    api_key = gpt_settings.cloudflare_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True

    name = "Cloudflare"
    model_name_keywords = ["@cf", "@hf"]
    base_url = f"https://api.cloudflare.com/client/v4/accounts/${gpt_settings.cloudflare_account_id}/ai/v1"

    default_model = "@cf/zai-org/glm-5.3-flash"
    default_moderation_model = "@cf/zai-org/glm-5.3-flash"
    default_vision_model = "@cf/zai-org/glm-5.3-flash"
