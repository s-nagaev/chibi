from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Together(OpenAIFriendlyProvider):
    api_key = gpt_settings.together_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True
    tts_ready = True
    stt_ready = True

    base_url = "https://api.together.ai/v1"
    name = "Together"

    default_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    default_moderation_model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    default_vision_model = "Qwen/Qwen3.5-9B"
    default_tts_model = "hexgrad/Kokoro-82M"
    default_tts_voice = "af_heart"
    default_stt_model = "openai/whisper-large-v3"
