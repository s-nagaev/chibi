from chibi.services.providers.provider import OpenAIFriendlyProvider


class Alibaba(OpenAIFriendlyProvider):
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    name = "Alibaba"
    model_name_keywords = ["qwen"]
    default_model = "qwen-plus"
