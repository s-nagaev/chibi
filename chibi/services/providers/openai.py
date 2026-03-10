import base64
from io import BytesIO

from loguru import logger
from openai import NOT_GIVEN, Omit, omit
from openai.types import ImagesResponse, ReasoningEffort
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_content_part_param import File, FileFile
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

from chibi.config import gpt_settings
from chibi.constants import OPENAI_TTS_INSTRUCTIONS
from chibi.schemas.app import VisionResultSchema
from chibi.services.providers.provider import OpenAIFriendlyProvider


class OpenAI(OpenAIFriendlyProvider):
    api_key = gpt_settings.openai_key
    chat_ready = True
    tts_ready = True
    stt_ready = True
    image_generation_ready = True
    moderation_ready = True
    vision_ready = True
    ocr_ready = True

    name = "OpenAI"
    model_name_prefixes = ["gpt", "o1", "o3", "o4"]
    model_name_keywords_exclude = ["audio", "realtime", "transcribe", "tts", "image"]
    base_url = "https://api.openai.com/v1"
    max_tokens = NOT_GIVEN
    default_model = "gpt-5.2"
    default_image_model = "dall-e-3"
    default_moderation_model = "gpt-5-mini"
    default_stt_model = "whisper-1"
    default_tts_model = "gpt-4o-mini-tts"
    default_tts_voice = "nova"
    default_vision_model = "gpt-4.1-mini"
    default_ocr_model = "gpt-4.1-mini"

    async def transcribe(self, audio: BytesIO, model: str | None = None) -> str:
        model = model or self.default_stt_model
        logger.info(f"Transcribing audio with model {model}...")
        response = await self.client.audio.transcriptions.create(
            model=model,
            file=("voice.ogg", audio.getvalue()),
        )
        if response:
            logger.info(f"Transcribed text: {response.text}")
            return response.text
        raise ValueError("Could not transcribe audio message")

    async def speech(self, text: str, voice: str | None = None, model: str | None = None) -> bytes:
        voice = voice or self.default_tts_voice
        model = model or self.default_tts_model
        logger.info(f"Recording a voice message with model {model}...")
        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions=OPENAI_TTS_INSTRUCTIONS,
        )
        return await response.aread()

    @classmethod
    def is_image_ready_model(cls, model_name: str) -> bool:
        return "dall-e" in model_name

    async def _get_image_generation_response(self, prompt: str, model: str) -> ImagesResponse:
        return await self.client.images.generate(  # type: ignore
            model=model,
            prompt=prompt,
            n=gpt_settings.image_n_choices if "dall-e-2" in model else 1,
            quality=self.image_quality,
            size=self.image_size if "dall-e-3" in model else NOT_GIVEN,
            timeout=gpt_settings.timeout,
            response_format="url",
        )

    def get_model_display_name(self, model_name: str) -> str:
        if "dall" in model_name:
            return model_name.replace("dall-e-", "DALL·E ")

        model_display_name = super().get_model_display_name(model_name=model_name)

        if "Gpt" in model_display_name:
            model_display_name = model_display_name.replace("Gpt ", "GPT-")
        return model_display_name

    def get_reasoning_effort_value(self, model_name: str) -> ReasoningEffort | Omit | None:
        if "chat" in model_name:
            return omit
        if "gpt-5" in model_name:
            return "medium"
        return omit

    def _get_temperature_value(self, model_name: str) -> float | Omit:
        if model_name.startswith("o"):
            return omit
        if model_name.startswith("gpt-5"):
            return omit
        return getattr(self, "temperature", gpt_settings.temperature)

    async def ocr(self, pdf: bytes, model: str | None = None) -> VisionResultSchema:
        """Extract text from a PDF document using OpenAI's vision/OCR capabilities.

        Args:
            pdf: The PDF file content as bytes.
            model: The model to use for OCR. Defaults to default_vision_model.

        Returns:
            VisionResultSchema containing the extracted text and descriptions.
        """
        model = model or self.default_ocr_model
        logger.info(f"[{self.name}] Extracting text from PDF with model {model}...")

        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf).decode("utf-8")
        file_data = f"data:application/pdf;base64,{pdf_base64}"

        # Build the message content with the PDF file
        content: list[ChatCompletionContentPartTextParam | File] = [
            File(
                type="file",
                file=FileFile(
                    filename="document.pdf",
                    file_data=file_data,
                ),
            ),
            ChatCompletionContentPartTextParam(
                type="text",
                text="Extract all text from this PDF. Provide a short description and full text content.",
            ),
        ]

        # Use parse() for structured output with Pydantic models
        response: ParsedChatCompletion[VisionResultSchema] = await self.get_client().chat.completions.parse(
            model=model,
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=content,
                )
            ],
            response_format=VisionResultSchema,
            # max_tokens=4096,
        )

        if not response.choices:
            from chibi.exceptions import ServiceResponseError

            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: empty response",
            )

        result = response.choices[0].message
        if not result or not result.parsed:
            from chibi.exceptions import ServiceResponseError

            raise ServiceResponseError(
                provider=self.name,
                model=model,
                detail="Could not extract text from PDF: empty parsed result",
            )

        logger.info(f"[{self.name}] PDF text extracted successfully: {result.parsed.short_description}...")
        return result.parsed
