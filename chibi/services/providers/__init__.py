from chibi.services.providers.alibaba import Alibaba
from chibi.services.providers.anthropic import Anthropic
from chibi.services.providers.cloudflare import Cloudflare
from chibi.services.providers.deepseek import DeepSeek
from chibi.services.providers.gemini import Gemini
from chibi.services.providers.grok import Grok
from chibi.services.providers.mistralai import MistralAI
from chibi.services.providers.moonshotai import MoonshotAI
from chibi.services.providers.openai import OpenAI

registered_providers = {
    "Alibaba": Alibaba,
    "Anthropic": Anthropic,
    "Cloudflare": Cloudflare,
    "DeepSeek": DeepSeek,
    "Gemini": Gemini,
    "Grok": Grok,
    "MistralAI": MistralAI,
    "MoonshotAI": MoonshotAI,
    "OpenAI": OpenAI,
}
