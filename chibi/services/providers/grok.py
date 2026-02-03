from openai import NOT_GIVEN

from chibi.config import gpt_settings
from chibi.schemas.app import ModelChangeSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider


class Grok(OpenAIFriendlyProvider):
    api_key = gpt_settings.grok_key
    chat_ready = True
    image_generation_ready = True
    moderation_ready = True

    base_url = "https://api.x.ai/v1"
    name = "Grok"
    model_name_keywords = ["grok"]
    model_name_keywords_exclude = ["vision", "image"]
    image_quality = NOT_GIVEN
    image_size = NOT_GIVEN
    default_image_model = "grok-2-image-1212"
    default_model = "grok-4-1-fast-reasoning"
    default_moderation_model = "grok-4-1-fast-non-reasoning"
    presence_penalty = NOT_GIVEN
    frequency_penalty = NOT_GIVEN
    image_n_choices = 1

    async def get_available_models(self, image_generation: bool = False) -> list[ModelChangeSchema]:
        models = await super().get_available_models(image_generation=image_generation)

        if not image_generation:
            return models

        # For some reason we stopped getting a grok-2-image-1212 model from the API. But it still works.
        if not models:
            models.append(
                ModelChangeSchema(
                    provider=self.name, name="grok-2-image-1212", display_name="Grok 2 Image", image_generation=True
                )
            )
        return models
