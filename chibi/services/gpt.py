from openai import AsyncOpenAI, AuthenticationError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice

from chibi.config import gpt_settings


async def get_chat_response(
    api_key: str,
    messages: list[ChatCompletionMessageParam],
    model: str,
    temperature: float = gpt_settings.temperature,
    max_tokens: int = gpt_settings.max_tokens,
    presence_penalty: float = gpt_settings.presence_penalty,
    frequency_penalty: float = gpt_settings.frequency_penalty,
    timeout: int = gpt_settings.timeout,
) -> tuple[str, CompletionUsage | None]:
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        # timeout=timeout,
    )
    if len(response.choices) == 0:
        return "", None

    choices: list[Choice] = response.choices
    answer = choices[0]
    usage = response.usage

    return answer.message.content or "", usage


async def get_images_by_prompt(api_key: str, prompt: str) -> list[str]:
    client = AsyncOpenAI(api_key=api_key)
    response = await client.images.generate(
        prompt=prompt,
        quality=gpt_settings.image_quality,
        model=gpt_settings.dall_e_model,
        size=gpt_settings.image_size,
        n=gpt_settings.image_n_choices,
    )
    return [image.url for image in response.data if image.url]


async def api_key_is_valid(api_key: str) -> bool:
    client = AsyncOpenAI(api_key=api_key)
    try:
        await client.completions.create(model="text-davinci-003", prompt="This is a test", max_tokens=5)
    except AuthenticationError:
        return False
    except Exception:
        raise
    return True


async def retrieve_available_models(api_key: str, include_gpt4: bool) -> list[str]:
    client = AsyncOpenAI(api_key=api_key)
    all_models = await client.models.list()

    if include_gpt4:
        return sorted([model.id for model in all_models.data if "gpt" in model.id])
    return sorted([model.id for model in all_models.data if "gpt-3" in model.id])
