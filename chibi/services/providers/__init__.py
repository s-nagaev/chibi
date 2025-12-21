# flake8: noqa: F401

from chibi.services.providers.alibaba import Alibaba
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.cloudflare import Cloudflare
from chibi.services.providers.customopenai import CustomOpenAI
from chibi.services.providers.deepseek import DeepSeek
from chibi.services.providers.eleven_labs import ElevenLabs
from chibi.services.providers.gemini_native import Gemini
from chibi.services.providers.grok import Grok
from chibi.services.providers.mistralai_native import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI
from chibi.services.providers.provider import RegisteredProviders

registered_providers = {
    "Alibaba": Alibaba,
    "Anthropic": Anthropic,
    "Cloudflare": Cloudflare,
    "DeepSeek": DeepSeek,
    "ElevenLabs": ElevenLabs,
    "Gemini": Gemini,
    "Grok": Grok,
    "MistralAI": MistralAI,
    "MoonshotAI": MoonshotAI,
    "OpenAI": OpenAI,
    "CustomOpenAI": CustomOpenAI,
}
