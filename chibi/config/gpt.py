from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from chibi.constants import BASE_PROMPT, FILESYSTEM_ACCESS_PROMPT, IMAGE_ASPECT_RATIO_LITERAL, IMAGE_SIZE_LITERAL


class GPTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alibaba_key: str | None = Field(alias="ALIBABA_API_KEY", default=None)
    anthropic_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)
    cloudflare_account_id: str | None = Field(alias="CLOUDFLARE_ACCOUNT_ID", default=None)
    cloudflare_key: str | None = Field(alias="CLOUDFLARE_API_KEY", default=None)
    customopenai_key: str | None = Field(alias="CUSTOMOPENAI_API_KEY", default=None)
    customopenai_url: str = Field(alias="CUSTOMOPENAI_URL", default="http://localhost:1234/v1")
    deepseek_key: str | None = Field(alias="DEEPSEEK_API_KEY", default=None)
    gemini_key: str | None = Field(alias="GEMINI_API_KEY", default=None)
    grok_key: str | None = Field(alias="GROK_API_KEY", default=None)
    mistralai_key: str | None = Field(alias="MISTRALAI_API_KEY", default=None)
    moonshotai_key: str | None = Field(alias="MOONSHOTAI_API_KEY", default=None)
    openai_key: str | None = Field(alias="OPENAI_API_KEY", default=None)
    suno_key: str | None = Field(alias="SUNO_API_ORG_API_KEY", default=None)

    frequency_penalty: float = Field(default=0)
    max_tokens: int = Field(default=32000)
    presence_penalty: float = Field(default=0)
    temperature: float = Field(default=1)

    backoff_factor: float = Field(default=0.5)
    retries: int = Field(default=3)
    timeout: int = Field(default=600)

    system_prompt: str = Field(alias="ASSISTANT_PROMPT", default=BASE_PROMPT)

    image_generations_monthly_limit: int = Field(alias="IMAGE_GENERATIONS_LIMIT", default=0)
    image_n_choices: int = Field(default=1)
    image_quality: Literal["standard", "hd"] = Field(default="standard")
    image_size: IMAGE_SIZE_LITERAL = Field(default="1024x1024")
    image_aspect_ratio: IMAGE_ASPECT_RATIO_LITERAL = Field(default="16:9")
    image_size_nano_banana: Literal["1K", "2K", "4K"] = Field(default="2K")
    image_size_imagen: Literal["1K", "2K"] = Field(default="2K")

    default_model: str | None = Field(default=None)
    default_provider: str | None = Field(default=None)

    stt_provider: str | None = Field(default=None)
    stt_model: str | None = Field(default=None)
    tts_provider: str | None = Field(default=None)
    tts_model: str | None = Field(default=None)

    filesystem_access: bool = Field(default=False)
    image_generations_whitelist_raw: str | None = Field(alias="IMAGE_GENERATIONS_WHITELIST", default=None)
    max_conversation_age_minutes: int = Field(default=360)
    max_history_tokens: int = Field(default=64000)
    models_whitelist_raw: str | None = Field(alias="MODELS_WHITELIST", default=None)
    proxy: str | None = Field(default=None)
    public_mode: bool = Field(default=False)
    show_llm_thoughts: bool = Field(default=False)

    google_search_api_key: str | None = Field(default=None)
    google_search_cx: str | None = Field(default=None)

    elevenlabs_api_key: str | None = Field(alias="ELEVEN_LABS_API_KEY", default=None)

    @property
    def google_search_client_set(self) -> bool:
        return bool(self.google_search_api_key) and bool(self.google_search_cx)

    @property
    def assistant_prompt(self) -> str:
        if self.filesystem_access:
            return self.system_prompt + FILESYSTEM_ACCESS_PROMPT
        return self.system_prompt

    @property
    def models_whitelist(self) -> list[str]:
        return [x.strip() for x in self.models_whitelist_raw.split(",")] if self.models_whitelist_raw else []

    @property
    def image_generations_whitelist(self) -> list[str]:
        return (
            [x.strip() for x in self.image_generations_whitelist_raw.split(",")]
            if self.image_generations_whitelist_raw
            else []
        )

    @property
    def messages_ttl(self) -> int:
        return self.max_conversation_age_minutes * 60


@lru_cache()
def _get_gpt_settings() -> GPTSettings:
    return GPTSettings()


gpt_settings = _get_gpt_settings()
