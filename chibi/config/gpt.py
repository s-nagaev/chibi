from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from chibi.constants import IMAGE_ASPECT_RATIO_LITERAL, get_llm_prompt


class GPTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alibaba_key: str | None = Field(alias="ALIBABA_API_KEY", default=None)
    anthropic_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)
    open_router_key: str | None = Field(alias="OPENROUTER_API_KEY", default=None)
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
    elevenlabs_api_key: str | None = Field(alias="ELEVEN_LABS_API_KEY", default=None)
    minimax_api_key: str | None = Field(alias="MINIMAX_API_KEY", default=None)
    zhipuai_key: str | None = Field(alias="ZHIPUAI_API_KEY", default=None)

    frequency_penalty: float = Field(default=0)
    max_tokens: int = Field(default=32000)
    presence_penalty: float = Field(default=0)
    temperature: float = Field(default=1)

    backoff_factor: float = Field(default=0.5)
    retries: int = Field(default=3)
    timeout: int = Field(default=180)

    image_generations_monthly_limit: int = Field(alias="IMAGE_GENERATIONS_LIMIT", default=0)
    image_n_choices: int = Field(default=1, ge=1, le=4)
    image_quality: Literal["standard", "hd", "low", "medium", "high", "auto"] = Field(default="auto")
    image_aspect_ratio: IMAGE_ASPECT_RATIO_LITERAL = Field(default="16:9")
    image_size_nano_banana: Literal["1K", "2K", "4K"] = Field(default="2K")
    image_size_imagen: Literal["1K", "2K"] = Field(default="2K")
    image_size_alibaba: str = "1664*928"
    image_size_openai: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "1536x1024"

    default_model: str | None = Field(default=None)
    default_provider: str | None = Field(default=None)

    stt_provider: str | None = Field(default=None)
    stt_model: str | None = Field(default=None)

    tts_provider: str | None = Field(default=None)
    tts_model: str | None = Field(default=None)
    tts_voice: str | None = Field(default=None)

    vision_provider: str | None = Field(default=None)
    vision_model: str | None = Field(default=None)

    ocr_provider: str | None = Field(default=None)

    moderation_provider: str | None = Field(default=None)
    moderation_model: str | None = Field(default=None)

    supervisor_enabled: bool = Field(
        default=False,
        description=(
            "Enable the Supervisor workflow guard for agent scenarios. When False (default), "
            "the Supervisor is not invoked at all. When True, every tool call (pre-execution) "
            "and every final LLM response is checked against the configured Supervisor model; "
            "on Intervene, execution is blocked or the response is regenerated, up to "
            "max_supervisor_retries. Independent from the Moderator (safety)."
        ),
    )
    supervisor_provider: str | None = Field(
        default=None,
        description=(
            "Optional explicit provider name for the Supervisor (e.g. 'openai', 'anthropic'). "
            "Falls back to moderation_provider when unset, and to the default "
            "RegisteredProviders.first_moderation_ready mechanism at runtime."
        ),
    )
    supervisor_model: str | None = Field(
        default=None,
        description=(
            "Optional explicit model id for the Supervisor (e.g. 'gpt-5-mini'). "
            "Falls back to moderation_model when unset, and to the provider's "
            "default_moderation_model at runtime."
        ),
    )
    max_supervisor_retries: int = Field(
        default=2,
        description=(
            "Maximum number of times the Supervisor may ask the model to "
            "regenerate a final answer after an 'intervene' verdict. After "
            "this many retries the last answer is returned as-is (fail-open)."
        ),
    )

    max_conversation_age_minutes: int = Field(default=360)
    max_history_tokens: int = Field(default=64000)

    image_generations_whitelist_raw: str | None = Field(alias="IMAGE_GENERATIONS_WHITELIST", default=None)
    models_whitelist_raw: str | None = Field(alias="MODELS_WHITELIST", default=None)
    models_blacklist_raw: str | None = Field(alias="MODELS_BLACKLIST", default=None)
    proxy: str | None = Field(default=None)
    public_mode: bool = Field(default=False)
    show_llm_thoughts: bool = Field(default=False)

    filesystem_access: bool = Field(default=False)
    allow_delegation: bool = Field(default=True)
    llm_role_raw: str | None = Field(alias="LLM_ROLE", default=None)
    delegate_task_timeout: int | None = Field(default=None)
    tools_whitelist_raw: str | None = Field(alias="TOOLS_WHITELIST", default=None)

    google_search_api_key: str | None = Field(default=None)
    google_search_cx: str | None = Field(default=None)

    @property
    def google_search_client_set(self) -> bool:
        return bool(self.google_search_api_key) and bool(self.google_search_cx)

    @property
    def llm_role(self) -> str | None:
        """Return the free-form role/persona text configured via ``LLM_ROLE``.

        The value is arbitrary operator-supplied prose; there is no fixed set of
        roles. Returns the stripped text, or ``None`` when unset/blank so the
        default agent behavior is preserved.
        """
        if not self.llm_role_raw or not self.llm_role_raw.strip():
            return None
        return self.llm_role_raw.strip()

    @property
    def assistant_prompt(self) -> str:
        return get_llm_prompt(
            filesystem_access=self.filesystem_access,
            allow_delegation=self.allow_delegation,
            role=self.llm_role,
        )

    @property
    def models_whitelist(self) -> list[str]:
        return [x.strip() for x in self.models_whitelist_raw.split(",")] if self.models_whitelist_raw else []

    @property
    def models_blacklist(self) -> list[str]:
        return [x.strip() for x in self.models_blacklist_raw.split(",")] if self.models_blacklist_raw else []

    @property
    def image_generations_whitelist(self) -> list[str]:
        return (
            [x.strip() for x in self.image_generations_whitelist_raw.split(",")]
            if self.image_generations_whitelist_raw
            else []
        )

    @property
    def tools_whitelist(self) -> list[str]:
        return [x.strip() for x in self.tools_whitelist_raw.split(",")] if self.tools_whitelist_raw else []

    @property
    def messages_ttl(self) -> int:
        return self.max_conversation_age_minutes * 60

    @property
    def supervisor_provider_resolved(self) -> str | None:
        """Return the Supervisor provider name resolved at the config level.

        Resolution order: explicit ``supervisor_provider`` if set, otherwise fall
        back to ``moderation_provider``, otherwise ``None``. Further resolution
        (e.g. via ``RegisteredProviders.first_moderation_ready``) is performed
        at runtime by the Supervisor-resolution layer and is not handled here.

        Returns:
            The resolved provider identifier, or ``None`` when neither
            ``supervisor_provider`` nor ``moderation_provider`` is configured.
        """
        if self.supervisor_provider:
            return self.supervisor_provider
        if self.moderation_provider:
            return self.moderation_provider
        return None

    @property
    def supervisor_model_resolved(self) -> str | None:
        """Return the Supervisor model id resolved at the config level.

        Resolution order: explicit ``supervisor_model`` if set, otherwise fall
        back to ``moderation_model``, otherwise ``None``. Further resolution
        (e.g. via the provider's ``default_moderation_model``) is performed at
        runtime by the Supervisor-resolution layer and is not handled here.

        Returns:
            The resolved model identifier, or ``None`` when neither
            ``supervisor_model`` nor ``moderation_model`` is configured.
        """
        if self.supervisor_model:
            return self.supervisor_model
        if self.moderation_model:
            return self.moderation_model
        return None


@lru_cache()
def _get_gpt_settings() -> GPTSettings:
    return GPTSettings()


gpt_settings: GPTSettings = _get_gpt_settings()
