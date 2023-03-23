import openai

from chibi.config import gpt_settings


async def get_chat_response(api_key: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
    response = await openai.ChatCompletion.acreate(
        api_key=api_key,
        model=gpt_settings.model,
        messages=messages,
        temperature=gpt_settings.temperature,
        max_tokens=gpt_settings.max_tokens,
        presence_penalty=gpt_settings.presence_penalty,
        frequency_penalty=gpt_settings.frequency_penalty,
        timeout=gpt_settings.timeout,
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
