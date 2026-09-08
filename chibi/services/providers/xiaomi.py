from chibi.config import gpt_settings
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Xiaomi(OpenAIFriendlyProvider):
    api_key = gpt_settings.xiaomi_mimo_key
    chat_ready = True
    moderation_ready = True
    vision_ready = True
    tts_ready = True
    voice_ready = True

    base_url = "https://api.xiaomimimo.com/v1"
    name = "Xiaomi"

    model_name_keywords_exclude = ["tts", "asr"]

    default_model = "mimo-v2.5"
    default_moderation_model = "mimo-v2.5"
    default_vision_model = "mimo-v2.5"
    default_tts_model = "mimo-v2.5-tts"
    default_stt_model = "mimo-v2.5-asr"
    default_tts_voice = "Chloe"
    # temperature = 1
