from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from chibi.config import gpt_settings
from chibi.exceptions import NoAccountIDSetError
from chibi.schemas.app import ChatResponseSchema, UsageSchema
from chibi.schemas.cloudflare import (
    ChatCompletionResponseSchema,
    ModelsSearchResponseSchema,
)
from chibi.services.providers.provider import RestApiFriendlyProvider


class Cloudflare(RestApiFriendlyProvider):
    name = "Cloudflare"
    model_name_keywords = ["@cf", "@hf"]
    default_model = "@cf/meta/llama-3.2-3b-instruct"
    base_url = f"https://api.cloudflare.com/client/v4/accounts/{gpt_settings.cloudflare_account_id}/ai"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
        }

    async def _get_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        system_prompt: str | None = None,
    ) -> ChatResponseSchema:
        if not gpt_settings.cloudflare_account_id:
            raise NoAccountIDSetError

        url = f"{self.base_url}/run/{model}"

        system_message = {"role": "system", "content": system_prompt}

        dialog = (
            [system_message] + [dict(m) for m in messages]
            if system_message not in messages
            else [dict(m) for m in messages]
        )
        data = {"messages": dialog}
        response = await self._request(method="POST", url=url, data=data)

        response_data = ChatCompletionResponseSchema(**response.json())
        if response_data.success:
            answer_data = response_data.result
            answer = answer_data.response
            usage = UsageSchema(
                completion_tokens=answer_data.usage.completion_tokens,
                prompt_tokens=answer_data.usage.prompt_tokens,
                total_tokens=answer_data.usage.total_tokens,
            )
        else:
            answer = ""
            usage = None

        return ChatResponseSchema(answer=answer, provider=self.name, model=model, usage=usage)

    async def get_available_models(self, image_generation: bool = False) -> list[str]:
        if image_generation:
            return []

        if not gpt_settings.cloudflare_account_id:
            logger.error("No Cloudflare account ID set. Please, check the CLOUDFLARE_ACCOUNT_ID env value.")
            return []

        url = f"{self.base_url}/models/search"
        params = {"task": "Text Generation"}

        response = await self._request(method="GET", url=url, params=params)

        data = response.json()
        response_data = ModelsSearchResponseSchema(**data)

        all_models = [model.name for model in response_data.result]

        if gpt_settings.models_whitelist:
            allowed_model_names = [model for model in all_models if model in gpt_settings.models_whitelist]
        else:
            allowed_model_names = all_models

        return sorted(allowed_model_names)
