from openai import AsyncOpenAI, AuthenticationError
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion import Choice

from chibi.config import gpt_settings
from chibi.services.provider import Provider


class OpenAI(Provider):
    @property
    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.token)

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float = gpt_settings.temperature,
        max_tokens: int = gpt_settings.max_tokens,
        presence_penalty: float = gpt_settings.presence_penalty,
        frequency_penalty: float = gpt_settings.frequency_penalty,
        timeout: int = gpt_settings.timeout,
    ) -> tuple[str, CompletionUsage | None]:
        system_message = ChatCompletionSystemMessageParam(role="system", content=gpt_settings.assistant_prompt)

        dialog = [system_message] + messages

        response = await self.client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=timeout,
        )
        if len(response.choices) == 0:
            return "", None

        choices: list[Choice] = response.choices
        answer = choices[0]
        usage = response.usage

        return answer.message.content or "", usage

    async def get_available_models(self) -> list[str]:
        openai_models = await self.client.models.list()

        if gpt_settings.models_whitelist:
            allowed_model_names = [
                model.id for model in openai_models.data if model.id in gpt_settings.models_whitelist
            ]
        else:
            allowed_model_names = [model.id for model in openai_models.data]

        if gpt_settings.gpt4_enabled:
            return sorted([model for model in allowed_model_names if "gpt" in model])
        return sorted([model for model in allowed_model_names if "gpt-3" in model])

    async def api_key_is_valid(self) -> bool:
        try:
            await self.client.completions.create(model="text-davinci-003", prompt="This is a test", max_tokens=5)
        except AuthenticationError:
            return False
        except Exception:
            raise
        return True

    async def get_images(self, prompt: str) -> list[str]:
        response = await self.client.images.generate(
            prompt=prompt,
            quality=gpt_settings.image_quality,
            model=gpt_settings.dall_e_model,
            size=gpt_settings.image_size,
            n=gpt_settings.image_n_choices,
        )
        return [image.url for image in response.data if image.url]
