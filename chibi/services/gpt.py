import openai
from openai.error import AuthenticationError

from chibi.config import gpt_settings


async def get_chat_response(
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float = gpt_settings.temperature,
    max_tokens: int = gpt_settings.max_tokens,
    presence_penalty: float = gpt_settings.presence_penalty,
    frequency_penalty: float = gpt_settings.frequency_penalty,
    timeout: int = gpt_settings.timeout,
) -> tuple[str, dict[str, int]]:
    response = await openai.ChatCompletion.acreate(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        timeout=timeout,
    )
    if len(response.choices) == 0:
        return "", {}

    answer = response.choices[0]["message"]["content"].strip()
    usage = dict(response.usage)

    return answer, usage


async def get_images_by_prompt(api_key: str, prompt: str) -> list[str]:
    response = await openai.Image.acreate(
        api_key=api_key, prompt=prompt, size=gpt_settings.image_size, n=gpt_settings.image_n_choices
    )
    return [result["url"] for result in response["data"]]


async def api_key_is_valid(api_key: str) -> bool:
    try:
        await openai.Completion.acreate(
            model="text-davinci-003", prompt="This is a test", max_tokens=5, api_key=api_key
        )
    except AuthenticationError:
        return False
    except Exception:
        raise
    return True


async def retrieve_available_models(api_key: str, include_gpt4: bool) -> list[str]:
    all_models = await openai.Model.alist(api_key=api_key)
    if include_gpt4:
        return [model["id"] for model in all_models.data if "gpt" in model["id"]]
    return [model["id"] for model in all_models.data if "gpt-3" in model["id"]]
