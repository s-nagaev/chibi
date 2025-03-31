from chibi.services.providers.provider import OpenAIFriendlyProvider


class DeepSeek(OpenAIFriendlyProvider):
    name = "DeepSeek"
    model_name_keywords = ["deepseek"]
    base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
