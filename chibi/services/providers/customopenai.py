from chibi.config import gpt_settings
from chibi.services.providers.openai_compatible import create_openai_compatible

CustomOpenAI = create_openai_compatible(
    name="CustomOpenAI",
    api_key=gpt_settings.customopenai_key,
    base_url=gpt_settings.customopenai_url,
    class_name="CustomOpenAI",
    dynamic=False,
)
